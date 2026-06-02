# cross_analysis_pipeline_split.py
# Split-mode: each panel saved as an individual figure file.
# Based on cross_analysis_pipeline.py, refactored for single-panel output.
#
# Usage:
#   python cross_analysis_pipeline_split.py results.csv transition.csv output_dir
#
# Output naming: Fig_01a, Fig_01b, ... Fig_10f (40 individual panels)

import warnings
warnings.filterwarnings('ignore')

import re, ast
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

import seaborn as sns
import networkx as nx
from sklearn.preprocessing import LabelEncoder

# ═══════════════════════════════════════════════════════════════════════════════
# Nature Communications colour palette
# ═══════════════════════════════════════════════════════════════════════════════
NC = {
    'blue':   '#4477AA', 'cyan':   '#66CCEE', 'green':  '#228833',
    'yellow': '#CCBB44', 'red':    '#EE6677', 'purple': '#AA3377',
    'grey':   '#BBBBBB', 'orange': '#E07030', 'teal':   '#3D8B8B',
    'navy':   '#223F5E',
}
NC_LIST = [NC[k] for k in ['blue','red','green','yellow','purple','cyan','orange','teal','grey']]
NC_DIV  = ['#4477AA','#77AADD','#BBCCEE','#F7F7F7','#FFCCCC','#EE9988','#BB4444']

STAGE_COLORS = {
    'M1': NC['blue'], 'M2': NC['green'],
    'M3': NC['orange'], 'M4': NC['red'],
}
ACTION_COLORS = {
    'a_next':    NC['blue'],
    'a_eval':    NC['cyan'],
    'a_mitig':   NC['yellow'],
    'a_verify':  NC['teal'],
    'a_escalate':NC['purple'],
    'a_abstain': NC['grey'],
    'a_accept':  NC['green'],
    'a_reject':  NC['red'],
    'a_defer':   NC['navy'],
    'a_redact':  NC['orange'],
}
DECISION_COLORS = {
    'verified':   NC['green'],
    'rejected':   NC['red'],
    'abstained':  NC['grey'],
    'escalated':  NC['purple'],
    'uncertain':  NC['orange'],
}
PAPER_OUTPUTS = ['verified', 'rejected', 'uncertain', 'abstained', 'escalated']
RISK_COLORS = ['#4477AA','#66CCEE','#EE6677','#AA3377']
RISK_ORDER  = ['minimal','limited','high','prohibited']
STAGES      = ['M1','M2','M3','M4']
ACTIONS     = ['a_next','a_eval','a_mitig','a_verify','a_escalate','a_abstain','a_accept','a_reject','a_defer','a_redact']

# ═══════════════════════════════════════════════════════════════════════════════
# Global matplotlib (Nature Communications style) — larger fonts for readability
# ═══════════════════════════════════════════════════════════════════════════════
matplotlib.rcParams.update({
    'font.family':        'Arial',
    'font.size':           11,
    'axes.titlesize':      12,
    'axes.labelsize':      14,
    'xtick.labelsize':     12,
    'ytick.labelsize':     12,
    'legend.fontsize':     10,
    'axes.linewidth':      0.8,
    'axes.spines.top':     False,
    'axes.spines.right':   False,
    'xtick.major.width':   0.8,
    'ytick.major.width':   0.8,
    'xtick.major.size':    4,
    'ytick.major.size':    4,
    'lines.linewidth':     1.5,
    'figure.dpi':          150,
    'savefig.dpi':         300,
    'savefig.bbox':        'tight',
    'pdf.fonttype':        42,
})

_FIG_DIR = [Path('figures_pipeline_split')]
_FIG_DIR[0].mkdir(exist_ok=True)

def savefig(name):
    plt.savefig(_FIG_DIR[0] / f'{name}.pdf')
    plt.savefig(_FIG_DIR[0] / f'{name}.png', dpi=300)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# 0. Load & Prepare Data
# ═══════════════════════════════════════════════════════════════════════════════
def load_data(results_csv='results_pipeline.csv'):
    df = pd.read_csv(results_csv)
    print(f'Loaded {len(df):,} episodes')

    le = LabelEncoder()
    for col in ['final_stage','max_stage','final_action','dominant_risk','paper_output']:
        if col in df.columns:
            df[col+'_enc'] = le.fit_transform(df[col].astype(str))

    if 'paper_output' not in df.columns:
        def _derive_output(row):
            fa = str(row.get('final_action', ''))
            if fa == 'a_accept':   return 'verified'
            elif fa == 'a_reject': return 'rejected'
            elif fa == 'a_abstain': return 'abstained'
            elif fa == 'a_escalate': return 'escalated'
            else: return 'uncertain'
        df['paper_output'] = df.apply(_derive_output, axis=1)

    # Derived columns for correlation
    g_cols = ['g_minimal','g_limited','g_high','g_prohibited']
    available_g = [c for c in g_cols if c in df.columns]
    if available_g:
        df['g_max'] = df[available_g].max(axis=1)
        g_arr = df[available_g].values
        entropy = -np.sum(g_arr * np.log(g_arr + 1e-8), axis=1)
        df['g_entropy'] = entropy / np.log(len(available_g))

    df['reward_per_step'] = df['total_reward'] / df['total_steps'].replace(0, 1)
    stage_map = {'M1': 1, 'M2': 2, 'M3': 3, 'M4': 4}
    df['max_stage_ordinal'] = df['max_stage'].map(stage_map).fillna(0)
    df['stage_progressed'] = (df['max_stage_ordinal'] > 1).astype(int)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: extract stage-action pairs from action_path
# ═══════════════════════════════════════════════════════════════════════════════
def _extract_stage_action_pairs(action_paths):
    all_pairs = []
    for path in action_paths:
        nodes = [n.strip() for n in str(path).split('->')]
        for i, node in enumerate(nodes):
            if node.startswith('a_') and node != 'a_next' and i > 0:
                prev = nodes[i-1]
                if prev in STAGES:
                    all_pairs.append((prev, node))
    return all_pairs


def _build_sa_pct(all_pairs, relevant_actions):
    sa_counts = defaultdict(Counter)
    for s, a in all_pairs:
        sa_counts[s][a] += 1
    sa_pct = pd.DataFrame(
        {s: {a: sa_counts[s].get(a,0)/max(sum(sa_counts[s].values()),1)*100
             for a in relevant_actions} for s in STAGES}
    ).T.reindex(STAGES)
    return sa_counts, sa_pct


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 panels — Overview Statistics (8 individual panels)
# ═══════════════════════════════════════════════════════════════════════════════
def fig01a_terminal_outputs(df):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    cnt = df['paper_output'].value_counts().reindex(PAPER_OUTPUTS, fill_value=0)
    ax.pie(cnt[cnt > 0].values, labels=cnt[cnt > 0].index,
           colors=[DECISION_COLORS.get(k, '#aaa') for k in cnt[cnt > 0].index],
           autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10},
           wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'})
    savefig('Fig_01a')

def fig01b_final_stage(df):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    cnt_fs = df['final_stage'].value_counts().reindex(STAGES, fill_value=0)
    ax.bar(cnt_fs.index, cnt_fs.values,
           color=[STAGE_COLORS[s] for s in cnt_fs.index], edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Count')
    savefig('Fig_01b')

def fig01c_max_stage(df):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    cnt_ms = df['max_stage'].value_counts().reindex(STAGES, fill_value=0)
    ax.bar(cnt_ms.index, cnt_ms.values,
           color=[STAGE_COLORS[s] for s in cnt_ms.index], edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Count')
    savefig('Fig_01c')

def fig01d_dominant_risk(df):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    cnt_dr = df['dominant_risk'].value_counts().reindex(RISK_ORDER, fill_value=0)
    ax.bar(cnt_dr.index, cnt_dr.values, color=RISK_COLORS, edgecolor='white', linewidth=0.5)
    ax.set_xticklabels(RISK_ORDER, rotation=20, ha='right')
    ax.set_ylabel('Count')
    savefig('Fig_01d')

def fig01e_reward_by_output(df):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    for out, col in DECISION_COLORS.items():
        sub = df[df['paper_output'] == out]['total_reward']
        if len(sub) > 0:
            ax.hist(sub, bins=60, alpha=0.65, color=col, label=out, density=True)
    ax.set_xlabel('Total reward')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    savefig('Fig_01e')

def fig01f_steps_per_episode(df):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.hist(df['total_steps'], bins=30, color=NC['blue'], edgecolor='white', linewidth=0.3)
    ax.axvline(df['total_steps'].median(), color=NC['red'], lw=1.2, ls='--',
               label=f"Median={df['total_steps'].median():.0f}")
    ax.set_xlabel('Total steps')
    ax.set_ylabel('Count')
    ax.legend()
    savefig('Fig_01f')

def fig01g_total_action_calls(df):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    action_totals = {}
    for a_prefix in ['n_eval','n_mitig','n_verify','n_escalate','n_abstain',
                      'n_accept','n_reject','n_defer','n_redact']:
        if a_prefix in df.columns:
            a_name = 'a_' + a_prefix[2:]
            action_totals[a_name] = df[a_prefix].sum()
    sorted_a = sorted(action_totals, key=lambda x: -action_totals[x])
    ax.barh(sorted_a, [action_totals[a] for a in sorted_a],
            color=[ACTION_COLORS[a] for a in sorted_a], edgecolor='white', linewidth=0.3)
    ax.set_xlabel('Count')
    savefig('Fig_01g')

def fig01h_stages_visited(df):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    if 'n_stages_visited' in df.columns:
        ax.hist(df['n_stages_visited'], bins=range(1, 6), align='left', rwidth=0.8,
                color=NC['teal'], edgecolor='white', linewidth=0.3)
        ax.set_xlabel('# Stages visited')
        ax.set_ylabel('Count')
    savefig('Fig_01h')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 panels — Stage vs Outcomes (8 individual boxplots)
# ═══════════════════════════════════════════════════════════════════════════════
def _fig02_boxplot(df, col, label, filename):
    fig, ax = plt.subplots(figsize=(4, 3.5))
    data = [df[df['final_stage'] == m][col].dropna().values for m in STAGES]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops={'color':'white','linewidth':1.5},
                    whiskerprops={'linewidth':0.8}, capprops={'linewidth':0.8},
                    flierprops={'marker':'o','markersize':1.5,'alpha':0.3,'linewidth':0})
    for patch, m in zip(bp['boxes'], STAGES):
        patch.set_facecolor(STAGE_COLORS[m])
        patch.set_alpha(0.85)
    ax.set_xticklabels(STAGES)
    ax.set_ylabel(label)
    ax.set_xlabel('Final stage')
    savefig(filename)

def fig02a_total_reward(df):
    _fig02_boxplot(df, 'total_reward', 'Total reward', 'Fig_02a')

def fig02b_total_steps(df):
    _fig02_boxplot(df, 'total_steps', 'Steps', 'Fig_02b')

def fig02c_n_verify(df):
    _fig02_boxplot(df, 'n_verify', '# Verify', 'Fig_02c')

def fig02d_n_escalate(df):
    _fig02_boxplot(df, 'n_escalate', '# Escalate', 'Fig_02d')

def fig02e_n_defer(df):
    _fig02_boxplot(df, 'n_defer', '# Defer', 'Fig_02e')

def fig02f_n_redact(df):
    _fig02_boxplot(df, 'n_redact', '# Redact', 'Fig_02f')

def fig02g_g_minimal(df):
    _fig02_boxplot(df, 'g_minimal', 'g minimal', 'Fig_02g')

def fig02h_g_prohibited(df):
    _fig02_boxplot(df, 'g_prohibited', 'g prohibited', 'Fig_02h')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 panels — Action Distribution (3 individual panels)
# ═══════════════════════════════════════════════════════════════════════════════
def fig03_action_data(df):
    """Return sa_counts, sa_pct for action distribution panels."""
    relevant_actions = ['a_eval','a_mitig','a_verify','a_escalate',
                        'a_defer','a_redact','a_abstain','a_accept','a_reject']
    pairs = _extract_stage_action_pairs(df['action_path'])
    return _build_sa_pct(pairs, relevant_actions)

def fig03a_action_stack(df):
    sa_counts, sa_pct = fig03_action_data(df)
    relevant_actions = list(sa_pct.columns)
    fig, ax = plt.subplots(figsize=(6, 4))
    bottom = np.zeros(len(STAGES))
    for act in relevant_actions:
        vals = sa_pct[act].fillna(0).values
        ax.bar(STAGES, vals, bottom=bottom,
               color=ACTION_COLORS[act], alpha=0.88,
               label=act.replace('a_',''), edgecolor='white', linewidth=0.4)
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v > 5:
                ax.text(i, b + v/2, f'{v:.0f}%', ha='center', va='center',
                        fontsize=8, color='white', fontweight='bold')
        bottom += vals
    ax.set_ylim(0, 100)
    ax.set_ylabel('Proportion (%)')
    ax.set_xlabel('Current stage')
    ax.legend(loc='lower center', frameon=False, fontsize=10, ncol=5,
              bbox_to_anchor=(0.5, 1.00))
    savefig('Fig_03a')

def fig03b_action_counts(df):
    sa_counts, _ = fig03_action_data(df)
    relevant_actions = ['a_eval','a_mitig','a_verify','a_escalate',
                        'a_defer','a_redact','a_abstain','a_accept','a_reject']
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(STAGES))
    w = 0.11
    for k, act in enumerate(relevant_actions):
        vals = [sa_counts[s].get(act, 0) for s in STAGES]
        ax.bar(x + (k - 4)*w, vals, w, color=ACTION_COLORS[act], alpha=0.88,
               label=act.replace('a_',''), edgecolor='white', linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(STAGES)
    ax.set_ylabel('Total call count')
    ax.set_xlabel('Current stage')
    ax.legend(loc='lower center', frameon=False, fontsize=10, ncol=5,
              bbox_to_anchor=(0.5, 1.00))
    savefig('Fig_03b')

def fig03c_action_heatmap(df):
    _, sa_pct = fig03_action_data(df)
    relevant_actions = ['a_eval','a_mitig','a_verify','a_escalate',
                        'a_defer','a_redact','a_abstain','a_accept','a_reject']
    fig, ax = plt.subplots(figsize=(6, 4))
    nc_seq = LinearSegmentedColormap.from_list('seq', ['#F7FBFF','#08519C'])
    sns.heatmap(sa_pct[relevant_actions].fillna(0), ax=ax, cmap=nc_seq, annot=True, fmt='.1f',
                annot_kws={'size':9}, linewidths=0.5, linecolor='white',
                cbar_kws={'shrink':0.7, 'label':'%'})
    ax.set_xlabel('Action')
    ax.set_ylabel('Current stage')
    ax.set_xticklabels([a.replace('a_','') for a in relevant_actions], rotation=30)
    savefig('Fig_03c')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 panels — Action Radar (4 individual radars)
# ═══════════════════════════════════════════════════════════════════════════════
def fig04_radar_data(df):
    relevant_actions = ['a_eval','a_mitig','a_verify','a_escalate',
                        'a_defer','a_redact','a_abstain','a_accept','a_reject']
    pairs = _extract_stage_action_pairs(df['action_path'])
    _, sa_pct = _build_sa_pct(pairs, relevant_actions)
    return sa_pct, relevant_actions

def fig04_radar_single(sa_pct, relevant_actions, stage_idx, stage_name, filename):
    N = len(relevant_actions)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    max_pct = sa_pct.max().max()
    y_max = min(100, max(25, int(np.ceil(max_pct / 5) * 5 + 5)))
    y_ticks = [y_max // 3, y_max * 2 // 3, y_max]
    y_tick_labels = [str(t) for t in y_ticks]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw={'projection':'polar'})
    vals = sa_pct.loc[stage_name, relevant_actions].fillna(0).values.tolist()
    vals += vals[:1]
    ax.plot(angles, vals, color=STAGE_COLORS[stage_name], lw=2.0)
    ax.fill(angles, vals, color=STAGE_COLORS[stage_name], alpha=0.20)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([a.replace('a_','') for a in relevant_actions], fontsize=9)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels, fontsize=7)
    ax.set_ylim(0, y_max)
    ax.spines['polar'].set_linewidth(0.5)
    savefig(filename)

def fig04a_radar_m1(df):
    sa_pct, relevant = fig04_radar_data(df)
    fig04_radar_single(sa_pct, relevant, 0, 'M1', 'Fig_04a')

def fig04b_radar_m2(df):
    sa_pct, relevant = fig04_radar_data(df)
    fig04_radar_single(sa_pct, relevant, 1, 'M2', 'Fig_04b')

def fig04c_radar_m3(df):
    sa_pct, relevant = fig04_radar_data(df)
    fig04_radar_single(sa_pct, relevant, 2, 'M3', 'Fig_04c')

def fig04d_radar_m4(df):
    sa_pct, relevant = fig04_radar_data(df)
    fig04_radar_single(sa_pct, relevant, 3, 'M4', 'Fig_04d')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 panels — Transition Heatmaps (6 individual heatmaps per action)
# ═══════════════════════════════════════════════════════════════════════════════
def fig05_heatmap_data(transition_csv):
    raw = pd.read_csv(transition_csv)
    def collapse_stage(s):
        s = str(s)
        for out in PAPER_OUTPUTS:
            if s == out: return out.capitalize()
        return s if s in STAGES else s
    raw['next_collapsed'] = raw['next_stage'].apply(collapse_stage)
    raw['current_collapsed'] = raw['current_stage'].apply(
        lambda x: x if x in STAGES else x)
    agg = raw.groupby(['current_collapsed','action','next_collapsed'], as_index=False)['count'].sum()
    sa_total = agg.groupby(['current_collapsed','action'])['count'].sum().rename('sa_total')
    agg = agg.join(sa_total, on=['current_collapsed','action'])
    agg['prob'] = agg['count'] / agg['sa_total']
    return agg

def fig05_action_heatmap(agg, action_name, filename):
    sub = agg[agg['action'] == action_name]
    all_next = sorted(agg['next_collapsed'].unique())
    next_order = [n for n in STAGES if n in all_next] + \
                 [n for n in ['Verified','Rejected','Abstained','Escalated'] if n in all_next]
    frm = [m for m in STAGES if m in sub['current_collapsed'].values]
    to  = [n for n in next_order if n in sub['next_collapsed'].values]
    if not frm or not to:
        plt.close('all'); return

    mat = pd.DataFrame(0.0, index=frm, columns=to)
    for _, row in sub.iterrows():
        if row['current_collapsed'] in frm and row['next_collapsed'] in to:
            mat.loc[row['current_collapsed'], row['next_collapsed']] += row['prob']
    mat = mat.div(mat.sum(axis=1).replace(0, 1), axis=0)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    nc_yr = LinearSegmentedColormap.from_list('yr', ['#FFFFD4','#FED976','#FEB24C','#FD8D3C','#E31A1C'])
    sns.heatmap(mat, ax=ax, cmap=nc_yr, vmin=0, vmax=1,
                annot=True, fmt='.2f', annot_kws={'size': 9},
                linewidths=0.5, linecolor='white',
                cbar_kws={'shrink': 0.7, 'label': 'Probability'})
    for label in ax.get_xticklabels():
        t = label.get_text()
        if t in STAGE_COLORS:    label.set_color(STAGE_COLORS[t])
        elif t == 'Verified':    label.set_color(NC['green'])
        elif t == 'Rejected':    label.set_color(NC['red'])
        elif t == 'Abstained':   label.set_color(NC['grey'])
        elif t == 'Escalated':   label.set_color(NC['purple'])
    for label in ax.get_yticklabels():
        label.set_color(STAGE_COLORS.get(label.get_text(), '#333'))
    ax.set_title(action_name.replace('a_','').capitalize(), fontsize=10,
                 color=ACTION_COLORS.get(action_name, '#333'), fontweight='bold')
    ax.set_xlabel('Next state')
    ax.set_ylabel('Current stage')
    ax.tick_params(axis='x', rotation=30)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    savefig(filename)

def fig05a_eval(transition_csv):
    agg = fig05_heatmap_data(transition_csv)
    fig05_action_heatmap(agg, 'a_eval', 'Fig_05a')

def fig05b_mitig(transition_csv):
    agg = fig05_heatmap_data(transition_csv)
    fig05_action_heatmap(agg, 'a_mitig', 'Fig_05b')

def fig05c_verify(transition_csv):
    agg = fig05_heatmap_data(transition_csv)
    fig05_action_heatmap(agg, 'a_verify', 'Fig_05c')

def fig05d_escalate(transition_csv):
    agg = fig05_heatmap_data(transition_csv)
    fig05_action_heatmap(agg, 'a_escalate', 'Fig_05d')

def fig05e_defer(transition_csv):
    agg = fig05_heatmap_data(transition_csv)
    fig05_action_heatmap(agg, 'a_defer', 'Fig_05e')

def fig05f_redact(transition_csv):
    agg = fig05_heatmap_data(transition_csv)
    fig05_action_heatmap(agg, 'a_redact', 'Fig_05f')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6 panels — Aggregate Stage Transition Matrix (2 panels)
# ═══════════════════════════════════════════════════════════════════════════════
def fig06_aggregate_data(transition_csv):
    raw = pd.read_csv(transition_csv)
    def collapse_stage(s):
        s = str(s)
        for out in PAPER_OUTPUTS:
            if s == out: return out.capitalize()
        return s if s in STAGES else s
    raw['next_collapsed'] = raw['next_stage'].apply(collapse_stage)
    raw['current_collapsed'] = raw['current_stage'].apply(
        lambda x: x if x in STAGES else x)
    total_flow = raw.groupby(['current_collapsed','next_collapsed'])['count'].sum().unstack(fill_value=0)
    all_next = sorted(raw['next_collapsed'].unique())
    row_order = [s for s in STAGES if s in total_flow.index]
    col_order = [n for n in (STAGES + ['Verified','Rejected','Abstained','Escalated']) if n in all_next]
    total_flow = total_flow.reindex(index=row_order, columns=col_order, fill_value=0)
    flow_pct = total_flow.div(total_flow.sum(axis=1), axis=0) * 100
    return total_flow, flow_pct

def fig06a_aggregate_count(transition_csv):
    total_flow, _ = fig06_aggregate_data(transition_csv)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    nc_blues = LinearSegmentedColormap.from_list('blues', ['#F7FBFF','#2171B5'])
    sns.heatmap(total_flow, ax=ax, cmap=nc_blues, annot=True, fmt='d',
                annot_kws={'size': 9}, linewidths=0.5, linecolor='white', cbar_kws={'shrink': 0.7})
    ax.set_xlabel('Next state')
    ax.set_ylabel('Current stage')
    ax.tick_params(axis='x', rotation=30)
    savefig('Fig_06a')

def fig06b_aggregate_pct(transition_csv):
    _, flow_pct = fig06_aggregate_data(transition_csv)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    nc_blues = LinearSegmentedColormap.from_list('blues', ['#F7FBFF','#2171B5'])
    sns.heatmap(flow_pct, ax=ax, cmap=nc_blues, annot=True, fmt='.1f',
                annot_kws={'size': 9}, linewidths=0.5, linecolor='white',
                cbar_kws={'shrink': 0.7, 'label': '%'})
    ax.set_xlabel('Next state')
    ax.set_ylabel('Current stage')
    ax.tick_params(axis='x', rotation=30)
    savefig('Fig_06b')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 7 — Force-Directed Stage-Output Graph (single panel, already individual)
# ═══════════════════════════════════════════════════════════════════════════════
def fig07_force_directed_outputs(transition_csv='transition_pipeline.csv', min_prob=0.005):
    raw = pd.read_csv(transition_csv)

    def classify_next(s):
        s = str(s)
        for out in PAPER_OUTPUTS:
            if s == out: return out
        return s if s in STAGES else s
    raw['next_class'] = raw['next_stage'].apply(classify_next)

    agg = raw.groupby(['current_stage','action','next_class'], as_index=False)['count'].sum()
    sa  = agg.groupby(['current_stage','action'])['count'].sum().rename('sa_total')
    agg = agg.join(sa, on=['current_stage','action'])
    agg['prob'] = agg['count'] / agg['sa_total']

    bundle = agg.groupby(['current_stage','next_class'], as_index=False).agg(
        total_count=('count','sum'), actions=('action', lambda x: list(x)),
        probs=('prob', lambda x: list(x)))
    def _dominant(sub):
        best = sub.loc[sub['count'].idxmax()]
        return best['action'], best['prob']
    dom = {(cm, nm): _dominant(grp)
           for (cm, nm), grp in agg.groupby(['current_stage','next_class'])}
    bundle['action'] = bundle.apply(
        lambda r: dom[(r['current_stage'], r['next_class'])][0], axis=1)
    bundle['prob']   = bundle.apply(
        lambda r: dom[(r['current_stage'], r['next_class'])][1], axis=1)
    bundle = bundle[bundle['prob'] >= min_prob]

    G = nx.DiGraph()
    for _, row in bundle.iterrows():
        G.add_node(row['current_stage'])
        G.add_node(row['next_class'])
        G.add_edge(row['current_stage'], row['next_class'],
                   action=row['action'], prob=float(row['prob']),
                   total_count=int(row['total_count']),
                   weight=float(np.log1p(row['total_count'])))

    pos = {}
    inner_r = 5.5
    outer_r = 13.0
    x_stretch = 1.5
    for m, deg in zip(STAGES, [0, 90, 180, 270]):
        pos[m] = np.array([x_stretch * inner_r * np.cos(np.deg2rad(deg)),
                           inner_r * np.sin(np.deg2rad(deg))])

    output_groups = {
        'verified': (10, 70), 'abstained': (80, 120),
        'rejected': (130, 190), 'escalated': (200, 260), 'uncertain': (270, 340),
    }
    output_colors = {
        'verified': NC['green'], 'rejected': NC['red'],
        'abstained': '#999999', 'escalated': NC['purple'], 'uncertain': NC['orange'],
    }
    for out_name, (a0, a1) in output_groups.items():
        nodes = [n for n in G.nodes() if str(n) == out_name]
        if not nodes: continue
        ang = np.deg2rad((a0 + a1) / 2)
        pos[nodes[0]] = np.array([x_stretch * outer_r * np.cos(ang), outer_r * np.sin(ang)])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor('#FAFAFA')
    ax.axis('off')

    from matplotlib.patches import Ellipse
    for r, ls, a in [(inner_r, '-', 0.04), (outer_r, '--', 0.03)]:
        ax.add_patch(Ellipse((0, 0), 2*x_stretch*r, 2*r, fill=False, ec='#999', lw=0.5,
                              ls=ls, alpha=a, zorder=0))

    edge_slots = defaultdict(list)
    for u, v, data in sorted(G.edges(data=True), key=lambda e: e[2]['total_count']):
        if u not in pos or v not in pos: continue
        pu, pv = np.array(pos[u]), np.array(pos[v])
        diff = pv - pu; dist = np.linalg.norm(diff)
        if dist < 1e-6: continue
        perp = np.array([-diff[1], diff[0]]) / dist
        col  = ACTION_COLORS.get(data['action'], NC['grey'])
        lw   = np.interp(data['total_count'], [50, 50000], [0.5, 4.5])
        alpha_v = np.interp(data['prob'], [0.01, 0.5], [0.20, 0.80])
        direction = np.arctan2(diff[1], diff[0])
        slot_key  = (u, round(direction, 2))
        slot_idx  = len(edge_slots[slot_key])
        edge_slots[slot_key].append((u, v))
        base  = 0.35 if (u in STAGES and v in STAGES) else 0.15
        sign  = 1 if slot_idx % 2 == 0 else -1
        rad   = sign * (base + slot_idx * 0.08)
        waypoint = (pu + pv) / 2.0 + perp * rad * dist * 0.60
        path = matplotlib.path.Path(
            [pu, waypoint, pv],
            [matplotlib.path.Path.MOVETO, matplotlib.path.Path.CURVE3, matplotlib.path.Path.CURVE3])
        ax.add_patch(matplotlib.patches.PathPatch(
            path, edgecolor=col, facecolor='none', lw=lw, alpha=alpha_v, zorder=2, capstyle='round'))
        tip = pv - (diff / dist) * 0.12
        ax.annotate("", xy=pv, xytext=tip,
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=lw*0.5,
                                    alpha=alpha_v, mutation_scale=8), zorder=3)
        if data['prob'] >= 0.08:
            lp = pu * 0.25 + waypoint * 0.50 + pv * 0.25
            ax.text(lp[0], lp[1], f"{data['prob']:.2f}", fontsize=7,
                    ha='center', va='center', color=col, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.18', fc='white',
                              ec=col, lw=0.4, alpha=0.88), zorder=5)

    for n in G.nodes():
        if n not in pos: continue
        x, y = pos[n]
        if n in STAGES:
            r, fc, fs = 0.80, STAGE_COLORS[n], 9
        elif n in output_colors:
            r, fc, fs = 0.56, output_colors[n], 7.5
        else: continue
        ax.add_patch(plt.Circle((x, y), r + 0.16, color=fc, alpha=0.08, ec='none', zorder=3))
        ax.add_patch(plt.Circle((x, y), r, color=fc, ec='white', lw=1.5, zorder=4, alpha=0.94))
        ax.text(x, y, n, fontsize=fs, fontweight='bold', ha='center', va='center', color='white', zorder=5)

    acts_in_graph = sorted({d['action'] for _, _, d in G.edges(data=True)})
    handles = [mpatches.Patch(color=ACTION_COLORS[a], alpha=0.85, label=a.replace('a_','').capitalize())
               for a in acts_in_graph]
    ax.legend(handles=handles, loc='lower center', frameon=True, fontsize=8,
              title='Action', title_fontsize=8.5, framealpha=0.90,
              ncol=min(4, len(handles)), bbox_to_anchor=(0.5, 0.06))
    ax.set_xlim(-20, 22)
    ax.set_ylim(-14, 14)
    ax.set_aspect('equal')
    plt.tight_layout(pad=0.05)
    savefig('Fig_07')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 8 — All-Path Cluster Graph (single panel, already individual)
# ═══════════════════════════════════════════════════════════════════════════════
def fig08_all_paths_cluster(results_csv='results_pipeline.csv', min_count=1):
    df = pd.read_csv(results_csv)
    all_trans = []
    for path in df['action_path']:
        nodes = [n.strip() for n in str(path).split('->')]
        for i in range(len(nodes) - 2):
            frm = nodes[i]; act = nodes[i+1]; to = nodes[i+2]
            if not act.startswith('a_'): continue
            # Skip S0 (removed from pipeline model)
            if frm == 'S0' or to == 'S0': continue
            frm_c = frm if frm in STAGES else (
                'Verified' if frm == 'verified' else 'Rejected' if frm == 'rejected' else
                'Abstained' if frm == 'abstained' else 'Escalated' if frm == 'escalated' else
                'Uncertain' if frm == 'uncertain' else frm)
            to_c  = to if to in STAGES else (
                'Verified' if to == 'verified' else 'Rejected' if to == 'rejected' else
                'Abstained' if to == 'abstained' else 'Escalated' if to == 'escalated' else
                'Uncertain' if to == 'uncertain' else to)
            all_trans.append((frm_c, act, to_c))
    tc = Counter(all_trans)
    G = nx.DiGraph()
    for (frm, act, to), cnt in tc.items():
        if cnt < min_count: continue
        G.add_node(frm); G.add_node(to)
        G.add_edge(frm, to, action=act, count=cnt, weight=float(np.log1p(cnt)))

    pos = {}
    # Stages on top row
    for m, x_off in zip(STAGES, [-3.0, -1.0, 1.0, 3.0]):
        pos[m] = np.array([x_off, 2.0])
    # Terminal outputs on bottom row
    for label, x_val, y_val in [('Verified', -4.0, -2.0), ('Escalated', -2.0, -2.0),
                                  ('Abstained', 0.0, -2.0), ('Uncertain', 2.0, -2.0),
                                  ('Rejected', 4.0, -2.0)]:
        if label in G.nodes(): pos[label] = np.array([x_val, y_val])
    fixed = [k for k in pos]
    pos = nx.spring_layout(G, seed=42, k=2.0, iterations=300, pos=pos, fixed=fixed)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor('#FAFAFA')
    ax.axis('off')

    edge_slots = defaultdict(list)
    max_count = max(tc.values()) if tc else 1
    for u, v, data in sorted(G.edges(data=True), key=lambda e: e[2]['count']):
        pu, pv = np.array(pos[u]), np.array(pos[v])
        diff = pv - pu; dist = np.linalg.norm(diff)
        if dist < 1e-6: continue
        perp = np.array([-diff[1], diff[0]]) / dist
        act = data['action']; col = ACTION_COLORS.get(act, NC['grey'])
        cnt = data['count']
        lw = np.interp(cnt, [min_count, max_count], [1.5, 10.0])
        alpha_v = np.interp(cnt, [min_count, max_count], [0.30, 0.85])
        direction = np.arctan2(diff[1], diff[0])
        slot_key = (u, round(direction, 2))
        slot_idx = len(edge_slots[slot_key])
        edge_slots[slot_key].append((u, v))
        base = 0.48 if (u in STAGES and v in STAGES) else 0.20
        sign = 1 if slot_idx % 2 == 0 else -1
        rad = sign * (base + slot_idx * 0.10)
        waypoint = (pu + pv) / 2.0 + perp * rad * dist * 0.55
        path = matplotlib.path.Path(
            [pu, waypoint, pv],
            [matplotlib.path.Path.MOVETO, matplotlib.path.Path.CURVE3, matplotlib.path.Path.CURVE3])
        ax.add_patch(matplotlib.patches.PathPatch(
            path, edgecolor=col, facecolor='none', lw=lw, alpha=alpha_v, zorder=1, capstyle='round'))
        tip = pv - (diff / dist) * 0.14
        ax.annotate("", xy=pv, xytext=tip,
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=lw*0.5,
                                    alpha=alpha_v, mutation_scale=12), zorder=2)
        if cnt >= 1:
            mid = pu * 0.25 + waypoint * 0.45 + pv * 0.30
            ax.text(mid[0], mid[1], f"{cnt:,}", fontsize=7, ha='center', va='center',
                    color=col, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.10', fc='white', ec=col, lw=0.5, alpha=0.85), zorder=5)

    for n in G.nodes():
        x, y = pos[n]
        if n in STAGES:
            r, fc, fs = 0.32, STAGE_COLORS[n], 10
        elif n == 'Verified':
            r, fc, fs = 0.36, NC['green'], 9
        elif n == 'Rejected':
            r, fc, fs = 0.36, NC['red'], 9
        elif n == 'Escalated':
            r, fc, fs = 0.36, NC['purple'], 9
        elif n == 'Abstained':
            r, fc, fs = 0.36, '#999999', 9
        elif n == 'Uncertain':
            r, fc, fs = 0.36, NC['orange'], 9
        else:
            r, fc, fs = 0.28, '#999999', 7
        ax.add_patch(plt.Circle((x, y), r + 0.07, color=fc, alpha=0.08, ec='none', zorder=3))
        ax.add_patch(plt.Circle((x, y), r, color=fc, ec='white', lw=1.5, zorder=4, alpha=0.94))
        ax.text(x, y, n, fontsize=fs, fontweight='bold', ha='center', va='center', color='white', zorder=5)
        out_total = sum(d['count'] for _, _, d in G.out_edges(n, data=True))
        in_total  = sum(d['count'] for _, _, d in G.in_edges(n, data=True))
        node_total = out_total if out_total > 0 else in_total
        if node_total > 0:
            label = f"{node_total/1000:.0f}k" if node_total >= 1000 else str(node_total)
            ax.text(x, y - r - 0.20, label, fontsize=9, ha='center', va='top',
                    color=fc, alpha=0.7, zorder=4)

    acts_in_graph = sorted({d['action'] for _, _, d in G.edges(data=True)})
    handles = [mpatches.Patch(color=ACTION_COLORS[a], alpha=0.85, label=a.replace('a_','').capitalize())
               for a in acts_in_graph]
    ax.legend(handles=handles, loc='lower center', frameon=True, fontsize=8.5,
              title='Action (edge colour)', title_fontsize=9.5, framealpha=0.92,
              ncol=len(handles), bbox_to_anchor=(0.5, -0.04))
    ax.set_xlim(-6.0, 6.0)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect('equal')
    plt.tight_layout(pad=0.05)
    savefig('Fig_08')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 9 — Spearman Correlation (single panel, already individual)
# ═══════════════════════════════════════════════════════════════════════════════
def fig09_correlation(df):
    corr_cols = []
    for col in ['total_reward', 'total_steps', 'reward_per_step']:
        if col in df.columns: corr_cols.append(col)
    for col in ['n_stages_visited', 'max_stage_ordinal', 'stage_progressed']:
        if col in df.columns: corr_cols.append(col)
    for a_short in ['verify','escalate','accept','reject','defer','redact','abstain']:
        col = f'n_{a_short}'
        if col in df.columns and df[col].nunique() > 1: corr_cols.append(col)
    for col in ['g_max', 'g_entropy']:
        if col in df.columns: corr_cols.append(col)
    for col in ['escalated_ever', 'input_risk_score']:
        if col in df.columns: corr_cols.append(col)

    corr_labels = {
        'total_reward': 'Total\nReward', 'total_steps': 'Total\nSteps',
        'reward_per_step': 'Reward\nper Step', 'n_stages_visited': '#Stages\nVisited',
        'max_stage_ordinal': 'Max\nStage', 'stage_progressed': 'Stage\nProgressed',
        'n_verify': '#Verify', 'n_escalate': '#Escalate', 'n_accept': '#Accept',
        'n_reject': '#Reject', 'n_defer': '#Defer', 'n_redact': '#Redact',
        'n_abstain': '#Abstain', 'g_max': 'g_max\n(Confidence)',
        'g_entropy': 'g_entropy\n(Uncertainty)', 'escalated_ever': 'Escalated\nEver',
        'input_risk_score': 'Input\nRisk Score',
    }
    labels = [corr_labels.get(c, c) for c in corr_cols]
    corr_df = df[corr_cols].dropna()
    rho, pval = stats.spearmanr(corr_df)
    sig_mask = pval > 0.05

    n_params = len(labels)
    fig_size = max(10, n_params * 0.65)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9))
    nc_div_cmap = LinearSegmentedColormap.from_list('NC_div', NC_DIV)
    sns.heatmap(
        pd.DataFrame(rho, index=labels, columns=labels),
        ax=ax, cmap=nc_div_cmap, center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.4, linecolor='white',
        annot=True, fmt='.2f', annot_kws={'size': 8},
        cbar_kws={'shrink': 0.65, 'label': "Spearman's ρ"})
    for i in range(len(labels)):
        for j in range(len(labels)):
            if sig_mask[i, j] and i != j:
                ax.text(j+0.5, i+0.5, '×', ha='center', va='center',
                        fontsize=9, color='#888888', alpha=0.5)
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.tick_params(axis='y', rotation=0,  labelsize=12)
    plt.tight_layout()
    savefig('Fig_09')


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 10 panels — Input Feature Impact (6 individual panels)
# ═══════════════════════════════════════════════════════════════════════════════
def fig10_feature_impact_single(df, fcol, label, filename):
    fig, ax = plt.subplots(figsize=(5, 4))
    ct = pd.crosstab(df[fcol], df['paper_output'], normalize='index')
    ct = ct.reindex(columns=PAPER_OUTPUTS, fill_value=0)
    x = np.arange(len(ct.index))
    w = 0.15
    for k, out in enumerate(PAPER_OUTPUTS):
        vals = ct[out].values if out in ct.columns else np.zeros(len(ct.index))
        ax.bar(x + (k - 2)*w, vals, w, color=DECISION_COLORS[out],
               alpha=0.85, label=out, edgecolor='white', linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(ct.index, rotation=25, ha='right', fontsize=14)
    ax.set_ylabel('Proportion')
    ax.legend(loc='lower center', frameon=False, fontsize=10, ncol=3,
              bbox_to_anchor=(0.5, 1.00))
    savefig(filename)

def fig10a_data_type(df):
    fig10_feature_impact_single(df, 'input_d', 'Data type (d)', 'Fig_10a')

def fig10b_biometric(df):
    fig10_feature_impact_single(df, 'input_b', 'Biometric (b)', 'Fig_10b')

def fig10c_risk_flag(df):
    fig10_feature_impact_single(df, 'input_r', 'Risk flag (r)', 'Fig_10c')

def fig10d_compliance(df):
    fig10_feature_impact_single(df, 'input_c', 'Compliance (c)', 'Fig_10d')

def fig10e_jurisdiction(df):
    fig10_feature_impact_single(df, 'input_j', 'Jurisdiction (j)', 'Fig_10e')

def fig10f_metadata(df):
    fig10_feature_impact_single(df, 'input_m', 'Metadata quality (m)', 'Fig_10f')


# ═══════════════════════════════════════════════════════════════════════════════
# Run All
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import sys

    results_csv    = sys.argv[1] if len(sys.argv) > 1 else 'results_pipeline.csv'
    transition_csv = sys.argv[2] if len(sys.argv) > 2 else 'transition_pipeline.csv'
    output_dir     = sys.argv[3] if len(sys.argv) > 3 else 'figures_pipeline_split'

    _FIG_DIR[0] = Path(output_dir)
    _FIG_DIR[0].mkdir(exist_ok=True)

    print("=" * 60)
    print("Cross-Border AI Governance — Individual Panel Generation")
    print("=" * 60)

    df = load_data(results_csv)

    # Fig_01 panels (8)
    print("[01/40] Fig_01a — Terminal outputs..."); fig01a_terminal_outputs(df)
    print("[02/40] Fig_01b — Final stage..."); fig01b_final_stage(df)
    print("[03/40] Fig_01c — Max stage..."); fig01c_max_stage(df)
    print("[04/40] Fig_01d — Dominant risk..."); fig01d_dominant_risk(df)
    print("[05/40] Fig_01e — Reward by output..."); fig01e_reward_by_output(df)
    print("[06/40] Fig_01f — Steps per episode..."); fig01f_steps_per_episode(df)
    print("[07/40] Fig_01g — Total action calls..."); fig01g_total_action_calls(df)
    print("[08/40] Fig_01h — Stages visited..."); fig01h_stages_visited(df)

    # Fig_02 panels (8)
    print("[09/40] Fig_02a — Total reward..."); fig02a_total_reward(df)
    print("[10/40] Fig_02b — Total steps..."); fig02b_total_steps(df)
    print("[11/40] Fig_02c — #Verify..."); fig02c_n_verify(df)
    print("[12/40] Fig_02d — #Escalate..."); fig02d_n_escalate(df)
    print("[13/40] Fig_02e — #Defer..."); fig02e_n_defer(df)
    print("[14/40] Fig_02f — #Redact..."); fig02f_n_redact(df)
    print("[15/40] Fig_02g — g_minimal..."); fig02g_g_minimal(df)
    print("[16/40] Fig_02h — g_prohibited..."); fig02h_g_prohibited(df)

    # Fig_03 panels (3)
    print("[17/40] Fig_03a — Action stack..."); fig03a_action_stack(df)
    print("[18/40] Fig_03b — Action counts..."); fig03b_action_counts(df)
    print("[19/40] Fig_03c — Action heatmap..."); fig03c_action_heatmap(df)

    # Fig_04 panels (4)
    print("[20/40] Fig_04a — Radar M1..."); fig04a_radar_m1(df)
    print("[21/40] Fig_04b — Radar M2..."); fig04b_radar_m2(df)
    print("[22/40] Fig_04c — Radar M3..."); fig04c_radar_m3(df)
    print("[23/40] Fig_04d — Radar M4..."); fig04d_radar_m4(df)

    # Fig_05 panels (6)
    print("[24/40] Fig_05a — Eval..."); fig05a_eval(transition_csv)
    print("[25/40] Fig_05b — Mitig..."); fig05b_mitig(transition_csv)
    print("[26/40] Fig_05c — Verify..."); fig05c_verify(transition_csv)
    print("[27/40] Fig_05d — Escalate..."); fig05d_escalate(transition_csv)
    print("[28/40] Fig_05e — Defer..."); fig05e_defer(transition_csv)
    print("[29/40] Fig_05f — Redact..."); fig05f_redact(transition_csv)

    # Fig_06 panels (2)
    print("[30/40] Fig_06a — Aggregate count..."); fig06a_aggregate_count(transition_csv)
    print("[31/40] Fig_06b — Aggregate proportion..."); fig06b_aggregate_pct(transition_csv)

    # Fig_07 (1)
    print("[32/40] Fig_07 — Force-directed outputs..."); fig07_force_directed_outputs(transition_csv)

    # Fig_08 (1)
    print("[33/40] Fig_08 — All-path cluster..."); fig08_all_paths_cluster(results_csv)

    # Fig_09 (1)
    print("[34/40] Fig_09 — Spearman correlation..."); fig09_correlation(df)

    # Fig_10 panels (6)
    print("[35/40] Fig_10a — Data type..."); fig10a_data_type(df)
    print("[36/40] Fig_10b — Biometric..."); fig10b_biometric(df)
    print("[37/40] Fig_10c — Risk flag..."); fig10c_risk_flag(df)
    print("[38/40] Fig_10d — Compliance..."); fig10d_compliance(df)
    print("[39/40] Fig_10e — Jurisdiction..."); fig10e_jurisdiction(df)
    print("[40/40] Fig_10f — Metadata quality..."); fig10f_metadata(df)

    print("\n" + "=" * 60)
    print(f"All 40 individual panels generated in {_FIG_DIR[0]}/")
    print("=" * 60)
