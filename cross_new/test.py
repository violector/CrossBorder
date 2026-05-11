import torch
import numpy as np
import pandas as pd
import ast
import random
from collections import defaultdict

from new_cross import DQNAgent, CrossBorderDQNMDP, StateEncoder, RiskClassifier


# ─── 推理用 Agent（纯利用，不探索）────────────────────────────────────────────

class InferenceAgent:
    def __init__(self, state_dim, action_dim, model_path):
        self.agent = DQNAgent(state_dim, action_dim)
        self.agent.load(model_path)
        self.agent.epsilon = 0.0  # 关闭探索，纯利用

    def act(self, state_vec, valid_actions):
        return self.agent.act(state_vec, valid_actions=valid_actions)


# ─── 扩展 MDP：增加 reset_with_pool 接口 ─────────────────────────────────────

class TestMDP(CrossBorderDQNMDP):
    def reset_with_pool(self, candidate_pool):
        """用指定候选池初始化 episode（用于 CSV 测试）"""
        self.candidate_pool = list(candidate_pool)
        self.current_candidate_idx = -1
        self.current_model_state = None
        self.episode_true_tiers = {}
        self.phase = 'select'
        return self._s0_state()


# ─── 将 CSV 行映射为候选模型类型 ─────────────────────────────────────────────

def infer_model_type(row):
    """根据特征匹配打分，推断最接近的模板类型 M1~M4"""
    templates = {
        'M1': {'d': 'structured',  'b': 'none',           'r': 'false',     'c': 'EU_only',       'm': 'optimal'},
        'M2': {'d': 'multimodal',  'b': 'remote_id',      'r': 'high-risk', 'c': 'adequacy',      'm': 'suboptimal'},
        'M3': {'d': 'multimodal',  'b': 'remote_id',      'r': 'high-risk', 'c': 'non_compliant', 'm': 'suboptimal'},
        'M4': {'d': 'multimodal',  'b': 'categorisation', 'r': 'high-risk', 'c': 'adequacy',      'm': 'insufficient'},
    }
    scores = {mt: sum(1 for f in templates[mt] if row.get(f) == templates[mt][f])
              for mt in templates}
    return max(scores, key=scores.get)


# ─── 辅助：状态字典 → 可读 key（用于转移统计）────────────────────────────────

def state_to_key(s):
    return (f"d={s.get('d','?')},b={s.get('b','?')},r={s.get('r','?')},"
            f"c={s.get('c','?')},k={s.get('k','?')},m={s.get('m','?')},"
            f"mt={s.get('model_type','?')}")

def state_to_model(s):
    """仅提取模型名称"""
    return s.get('model_type', '?')


# ─── 主测试函数 ───────────────────────────────────────────────────────────────

def run_test(csv_path='transformed_data.csv',
             model_path='agent_v1.pth',
             results_path='results_v1.csv',
             transition_path='transition_v1.csv',
             pool_size=3,
             seed=42):

    np.random.seed(seed)
    random.seed(seed)

    # 读取并预处理 CSV
    df = pd.read_csv(csv_path)
    df['m_metrics_parsed'] = df['m_metrics'].apply(
        lambda v: ast.literal_eval(str(v)) if pd.notna(v) else [0.8, 0.8, 0.8, 0.1])
    df['model_type_inferred'] = df.apply(infer_model_type, axis=1)
    print(f"📂 已读取 {len(df)} 条测试数据: {csv_path}")

    # 初始化环境与 Agent
    env = TestMDP()
    agent = InferenceAgent(env.state_dim, env.action_dim, model_path)

    transition_counts = defaultdict(int)
    from_action_totals = defaultdict(int)
    results_rows = []
    transition_rows = []

    # 按 pool_size 分批，每批作为一个 episode
    batches = [df.iloc[i:i + pool_size].to_dict('records')
               for i in range(0, len(df), pool_size)]
    print(f"🧪 共 {len(batches)} 个 episode（每批最多 {pool_size} 条）\n")

    # CSV 原始列名（去掉预处理时新增的派生列）
    raw_input_cols = [c for c in df.columns
                      if c not in ('m_metrics_parsed', 'model_type_inferred')]

    for ep_idx, batch in enumerate(batches):
        candidate_pool = [row['model_type_inferred'] for row in batch]

        # 每个候选槽位对应的原始输入行（按 pool_size 对齐，不足时补空）
        input_slots = []
        for slot_idx in range(pool_size):
            if slot_idx < len(batch):
                row = batch[slot_idx]
                input_slots.append({c: row.get(c, '') for c in raw_input_cols})
            else:
                input_slots.append({c: '' for c in raw_input_cols})

        state_dict = env.reset_with_pool(candidate_pool)
        state_vec = env.encoder.encode(state_dict)

        total_reward, steps, done = 0.0, 0, False
        g_history, final_action, final_model, final_state = [], None, None, None
        # 决策链：交替记录 状态节点 和 动作节点，起点为 s0
        chain = ['s0']

        while not done and steps < 80:
            valid = env.get_valid_actions(state_dict)
            action_idx = agent.act(state_vec, valid)
            action_name = env.actions[action_idx]

            curr_key = state_to_key(state_dict)
            curr_model = state_to_model(state_dict)
            curr_g = state_dict.get('g', [0.25] * 4)[:]

            next_dict, reward, done = env.step(state_dict, action_idx)
            next_vec = env.encoder.encode(next_dict)

            # ── 确定 next 节点的展示名称 ─────────────────────────────────────
            if action_name in ('a_accept', 'a_reject'):
                decision_word  = 'Accept' if action_name == 'a_accept' else 'Reject'
                decision_model = state_dict.get('model_type', '?')
                sub            = env._select_subtype(
                    decision_model,
                    'accept' if action_name == 'a_accept' else 'reject')
                terminal_label = f"{decision_word}_{decision_model}_{sub}"
                next_key   = terminal_label
                next_model = terminal_label
            else:
                next_key   = state_to_key(next_dict)
                next_model = state_to_model(next_dict)

            # ── 更新转移统计 ──────────────────────────────────────────────────
            tk = (curr_key, action_name, next_key)
            transition_counts[tk] += 1
            from_action_totals[(curr_key, action_name)] += 1

            transition_rows.append({
                'episode':            ep_idx + 1,
                'step':               steps + 1,
                'current_state':      curr_key,
                'current_model':      curr_model,
                'action':             action_name,
                'next_state':         next_key,
                'next_model':         next_model,
                'reward':             round(reward, 4),
                'current_k':          state_dict.get('k', '?'),
                'next_k':             next_dict.get('k', '?'),
                'current_eval_count': state_dict.get('eval_count', 0),
                'next_eval_count':    next_dict.get('eval_count', 0),
                'g_before':           str([round(x, 4) for x in curr_g]),
                'g_after':            str([round(x, 4) for x in next_dict.get('g', [0.25] * 4)]),
                'done':               done,
            })

            # ── 决策链：动作节点 -> 下一状态节点 ────────────────────────────
            # a_next 后状态节点显示选中的模型名；其余动作显示 next_model
            if action_name == 'a_next':
                next_node = next_dict.get('model_type', '?')
            else:
                next_node = next_model  # 普通状态用模型名，终端用 Accept_M2_b
            chain.append(action_name)
            chain.append(next_node)

            total_reward += reward
            steps += 1

            if action_name in ('a_accept', 'a_reject'):
                final_action = action_name
                final_model  = decision_model
                final_state  = terminal_label
                g_history.append(curr_g)

            state_vec, state_dict = next_vec, next_dict

        final_g = g_history[-1] if g_history else state_dict.get('g', [0.25] * 4)

        # 基础输出字段
        result_row = {
            'episode':        ep_idx + 1,
            'candidate_pool': str(candidate_pool),
            'initial_model':  candidate_pool[0],
            'final_model':    final_model or '(none)',
            'final_state':    final_state or '(timeout)',
            'action_path':    ' -> '.join(chain),
            'final_action':   final_action or '(timeout)',
            'total_reward':   round(total_reward, 4),
            'total_steps':    steps,
            'g_final':        str([round(x, 4) for x in final_g]),
            'g_minimal':      round(final_g[0], 4),
            'g_limited':      round(final_g[1], 4),
            'g_high':         round(final_g[2], 4),
            'g_banned':       round(final_g[3], 4),
            'dominant_risk':  ['minimal', 'limited', 'high', 'banned'][int(np.argmax(final_g))],
        }

        # 追加每个候选槽位的原始输入字段，列名加 input_N_ 前缀
        for slot_idx, slot_data in enumerate(input_slots):
            prefix = f'input_{slot_idx + 1}_'
            for col, val in slot_data.items():
                result_row[f'{prefix}{col}'] = val

        results_rows.append(result_row)

        if (ep_idx + 1) % 50 == 0 or ep_idx == 0:
            print(f"  Episode {ep_idx + 1:4d}: pool={candidate_pool}, "
                  f"path={ ' -> '.join(chain) }, "
                  f"final_state={final_state}, reward={total_reward:.1f}")

    # 保存 results_v1.csv
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 决策结果已保存: {results_path}  ({len(results_df)} 行)")

    # 保存 transition_v1.csv（含转移概率、current_model、next_model）
    trans_rows = []
    for (curr, act, nxt), cnt in sorted(transition_counts.items(), key=lambda x: -x[1]):
        # 从 transition_rows 中提取对应的 current_model / next_model 示例值
        match = next(
            (r for r in transition_rows
             if r['current_state'] == curr and r['action'] == act and r['next_state'] == nxt),
            None)
        curr_model_ex = match['current_model'] if match else '?'
        next_model_ex = match['next_model']    if match else '?'
        trans_rows.append({
            'current_state':           curr,
            'current_model':           curr_model_ex,
            'action':                  act,
            'next_state':              nxt,
            'next_model':              next_model_ex,
            'count':                   cnt,
            'total_from_state_action': from_action_totals[(curr, act)],
            'transition_probability':  round(cnt / from_action_totals[(curr, act)], 4),
        })
    pd.DataFrame(trans_rows).to_csv(transition_path, index=False, encoding='utf-8-sig')
    print(f"✅ 转移统计已保存: {transition_path}  ({len(trans_rows)} 行)\n")

    # 摘要
    print("=" * 60)
    print("📊 决策摘要")
    print("=" * 60)
    for act, cnt in results_df['final_action'].value_counts().items():
        print(f"  {act}: {cnt} ({cnt/len(results_df)*100:.1f}%)")
    print(f"\n平均总奖励: {results_df['total_reward'].mean():.2f}")
    print(f"平均步数:   {results_df['total_steps'].mean():.2f}")
    print(f"\n主导风险分布:")
    for risk, cnt in results_df['dominant_risk'].value_counts().items():
        print(f"  {risk}: {cnt} ({cnt/len(results_df)*100:.1f}%)")


if __name__ == '__main__':
    run_test(
        csv_path='comprehensive_test_data.csv',
        model_path='agent_v3.pth',
        results_path='results_v8.csv',
        transition_path='transition_v8.csv',
        pool_size=3,
        seed=42,
    )