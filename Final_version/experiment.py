import torch
import numpy as np
import pandas as pd
import ast
import random
from collections import defaultdict

from new_crossf import DQNAgent, CrossBorderDQNMDP, StateEncoder, RiskClassifier


# ─── 推理用 Agent（纯利用，不探索）────────────────────────────────────────────

class InferenceAgent:
    def __init__(self, state_dim, action_dim, model_path):
        self.agent = DQNAgent(state_dim, action_dim)
        self.agent.load(model_path)
        self.agent.epsilon = 0.0

    def act(self, state_vec, valid_actions):
        return self.agent.act(state_vec, valid_actions=valid_actions)


# ─── 扩展 MDP：暴露 reset_with_input 接口 ────────────────────────────────────

class TestMDP(CrossBorderDQNMDP):
    pass   # reset_with_input 已在基类实现，直接继承即可


# ─── 辅助函数 ─────────────────────────────────────────────────────────────────

def state_to_key(s):
    return (f"d={s.get('d','?')},b={s.get('b','?')},r={s.get('r','?')},"
            f"c={s.get('c','?')},k={s.get('k','?')},m={s.get('m','?')},"
            f"mt={s.get('model_type','?')}")

def state_to_model(s):
    return s.get('model_type', '?')


# ─── 主测试函数 ───────────────────────────────────────────────────────────────

def run_test(csv_path='transformed_data.csv',
             model_path='agent_v1.pth',
             results_path='results_v1.csv',
             transition_path='transition_v1.csv',
             top_n=4,
             seed=42):

    np.random.seed(seed)
    random.seed(seed)

    # 读取 CSV，解析 m_metrics
    df = pd.read_csv(csv_path)
    df['m_metrics_parsed'] = df['m_metrics'].apply(
        lambda v: ast.literal_eval(str(v)) if pd.notna(v) else [0.8, 0.8, 0.8, 0.1])
    print(f"📂 已读取 {len(df)} 条输入数据: {csv_path}")
    print(f"⚙️  top_n={top_n}（每条输入最多尝试 {top_n} 个候选模型）")

    env   = TestMDP()
    agent = InferenceAgent(env.state_dim, env.action_dim, model_path)

    # 原始输入列（不含预处理派生列）
    raw_input_cols = [c for c in df.columns if c != 'm_metrics_parsed']

    transition_counts  = defaultdict(int)
    from_action_totals = defaultdict(int)
    results_rows       = []
    transition_rows    = []

    print(f"🧪 共 {len(df)} 个 episode（每条输入数据独立一个 episode）\n")

    for ep_idx, raw_row in enumerate(df.to_dict('records')):

        # 用该行特征构建候选池（相似度降序排列的4个模型）
        input_features = {c: raw_row.get(c, '') for c in ['d', 'b', 'r', 'c', 'm']}
        candidate_pool = env.build_candidate_pool(input_features)

        state_dict = env.reset_with_input(input_features, top_n=top_n)
        state_vec  = env.encoder.encode(state_dict)

        total_reward, steps, done = 0.0, 0, False
        g_history, final_action, final_model, final_state = [], None, None, None
        chain = ['s0']

        while not done and steps < 80:
            valid      = env.get_valid_actions(state_dict)
            action_idx = agent.act(state_vec, valid)
            action_name = env.actions[action_idx]

            curr_key   = state_to_key(state_dict)
            curr_model = state_to_model(state_dict)
            curr_g     = state_dict.get('g', [0.25] * 4)[:]

            next_dict, reward, done = env.step(state_dict, action_idx)
            next_vec  = env.encoder.encode(next_dict)

            # 终단标签
            # a_accept：始终独立生成 Accept_Mx_x
            # a_reject 中途：独立生成 Reject_Mx_x（回到 s0，next_dict 无 final_label）
            # a_reject 耗尽：优先使用 MDP 已生成的 final_label，保持子类一致
            if action_name in ('a_accept', 'a_reject'):
                decision_word  = 'Accept' if action_name == 'a_accept' else 'Reject'
                decision_model = state_dict.get('model_type', '?')
                mdp_label      = next_dict.get('final_label')   # 仅耗尽 reject 时非 None
                if mdp_label:
                    terminal_label = mdp_label
                else:
                    sub = env._select_subtype(
                        decision_model,
                        'accept' if action_name == 'a_accept' else 'reject')
                    terminal_label = f"{decision_word}_{decision_model}_{sub}"
                next_key   = terminal_label
                next_model = terminal_label
            else:
                next_key   = state_to_key(next_dict)
                next_model = state_to_model(next_dict)

            # 转移统计
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
                'g_after':            str([round(x, 4) for x in next_dict.get('g', [0.25]*4)]),
                'done':               done,
            })

            # 决策链
            if action_name == 'a_next':
                chain.append(action_name)
                chain.append(next_dict.get('model_type', '?'))
            else:
                chain.append(action_name)
                chain.append(next_model)

            total_reward += reward
            steps += 1

            if action_name in ('a_accept', 'a_reject'):
                final_action = action_name
                final_model  = decision_model
                final_state  = terminal_label
                g_history.append(curr_g)

            state_vec, state_dict = next_vec, next_dict

        final_g = g_history[-1] if g_history else state_dict.get('g', [0.25] * 4)

        # 构建结果行：基础输出 + 原始输入字段
        result_row = {
            'episode':        ep_idx + 1,
            'candidate_pool': str(candidate_pool),
            'initial_model': candidate_pool[0],
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
            'dominant_risk':  ['minimal','limited','high','banned'][int(np.argmax(final_g))],
        }
        # 追加原始输入字段
        for col in raw_input_cols:
            result_row[f'input_{col}'] = raw_row.get(col, '')

        results_rows.append(result_row)

        if (ep_idx + 1) % 100 == 0 or ep_idx == 0:
            print(f"  Episode {ep_idx + 1:5d}: pool={candidate_pool} | "
                  f"path={' -> '.join(chain)} | "
                  f"final={final_state} | reward={total_reward:.1f}")

    # 保存 results
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 决策结果已保存: {results_path}  ({len(results_df)} 行)")

    # 保存 transition（含转移概率）
    trans_rows = []
    for (curr, act, nxt), cnt in sorted(transition_counts.items(), key=lambda x: -x[1]):
        match = next(
            (r for r in transition_rows
             if r['current_state'] == curr and r['action'] == act and r['next_state'] == nxt),
            None)
        trans_rows.append({
            # 'current_state':           curr,
            'current_model':           match['current_model'] if match else '?',
            'action':                  act,
            # 'next_state':              nxt,
            'next_model':              match['next_model'] if match else '?',
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
        results_path='results_v4.csv',
        transition_path='transition_v4.csv',
        top_n=4,
        seed=42,
    )