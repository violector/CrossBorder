import pandas as pd
import torch
import ast
import os
import numpy as np
from collections import defaultdict

from new_cross import DQNAgent, CrossBorderDQNMDP, StateEncoder, RiskClassifier


def infer_model_type_from_row(row: dict) -> str:
    b = row.get('b', 'none')
    c = row.get('c', 'EU_only')
    d = row.get('d', 'structured')
    r = row.get('r', 'false')

    if b == 'categorisation':
        return 'M4'
    if c in ('non_compliant', 'inadequate_SCC'):
        return 'M3'
    if d in ('multimodal', 'video') or r == 'high-risk':
        return 'M2'
    return 'M1'


class TransitionTracker:
    def __init__(self):
        self.counts = defaultdict(lambda: defaultdict(int))

    def record(self, s, a, ns):
        self.counts[(s, a)][ns] += 1

    def get_probabilities(self) -> pd.DataFrame:
        rows = []
        for (s, a), next_states in self.counts.items():
            total = sum(next_states.values())
            for ns, count in next_states.items():
                rows.append({
                    'current_state': s,
                    'action':        a,
                    'next_state':    ns,
                    'count':         count,
                    'probability':   round(count / total, 4),
                })
        return pd.DataFrame(rows)

    def save_to_csv(self, filename: str):
        df = self.get_probabilities()
        df.to_csv(filename, index=False)
        print(f"✅ 转移概率矩阵已保存至: {filename}")


def run_csv_test(input_csv: str, output_csv: str,
                 model_path: str, num_rows: int = None):

    # ── 1. 初始化环境与 Agent ──────────────────────────────
    env     = CrossBorderDQNMDP()
    encoder = StateEncoder()
    agent   = DQNAgent(state_dim=env.state_dim, action_dim=env.action_dim)

    if os.path.exists(model_path):
        agent.load(model_path)
        agent.model.eval()
        agent.epsilon = 0.0
        print(f"✓ 已加载模型: {model_path}  (state_dim={env.state_dim})")
        print(f"✓ 评估模式 epsilon=0.0")
    else:
        print(f"❌ 找不到模型文件: {model_path}")
        return

    # ── 2. 读取数据 ────────────────────────────────────────
    print("\n📂 读取数据集...")
    df = pd.read_csv(input_csv)
    print(f"✓ 总行数: {len(df)}")
    print(f"✓ 列名:   {list(df.columns)}")

    required_cols = ['d', 'b', 'r', 'c', 'k', 'm', 'm_metrics']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"❌ 缺少必需列: {missing}")
        return

    print("\n📊 数据样本（前3行）:")
    print(df.head(3).to_string())
    print()

    print("📈 特征分布:")
    for col in ['d', 'b', 'r', 'c', 'm']:
        print(f"  {col}: {dict(df[col].value_counts())}")
    print()

    if num_rows is not None:
        df = df.head(num_rows)
        print(f"⚙️  限制测试行数: {num_rows}\n")

    results = []
    tracker = TransitionTracker()
    tier_names = ['minimal', 'limited', 'high', 'banned']
    print(f"🚀 开始推理测试...\n")

    # ── 3. 逐行推理（每行 = 独立单候选 episode）─────────────
    for index, row in df.iterrows():

        # ── 3a. 解析 m_metrics ────────────────────────────
        try:
            raw = row['m_metrics']
            m_metrics = ast.literal_eval(raw) if isinstance(raw, str) else raw
            if not isinstance(m_metrics, list):
                m_metrics = [0.8, 0.8, 0.8, 0.1]
            elif len(m_metrics) != 4:
                m_metrics = (m_metrics + [0.1] * 4)[:4]
        except Exception as e:
            print(f"⚠️  行 {index}: m_metrics 解析失败，使用默认值。{e}")
            m_metrics = [0.8, 0.8, 0.8, 0.1]

        # ── 3b. 推断 model_type 与初始 g ──────────────────
        model_type = infer_model_type_from_row(row)
        try:
            initial_g = RiskClassifier.compute_initial_g(
                row['d'], row['b'], row['r'], row['c'], row['m'])
        except Exception as e:
            print(f"⚠️  行 {index}: g 值计算失败，使用均匀分布。{e}")
            initial_g = [0.25, 0.25, 0.25, 0.25]

        # ── 3c. 初始化单候选 episode ──────────────────────
        env.candidate_pool        = [model_type]
        env.current_candidate_idx = -1
        env.current_model_state   = None
        env.episode_true_tiers    = {}
        env.phase                 = 'select'

        state_dict = env._s0_state()
        state_dict, _, _ = env.step(state_dict, action_idx=0)  # a_next

        # 用 CSV 真实特征覆盖模板特征
        state_dict.update({
            'd':                    row['d'],
            'b':                    row['b'],
            'r':                    row['r'],
            'c':                    row['c'],
            'k':                    row['k'],
            'm':                    row['m'],
            'm_metrics':            m_metrics,
            'g':                    initial_g[:],
            'eval_count':           0,
            'mitig_count':          0,
            'model_type':           model_type,
            'candidates_remaining': 0,
        })

        # ── 3d. 决策循环 ──────────────────────────────────
        done                = False
        step_count          = 0
        total_reward        = 0.0
        actions_taken       = []
        # state_history 每项为 (model_type_str, g_list, k_str)
        state_history       = [(model_type, initial_g[:], row['k'])]

        # 决策结果记录（在 accept/reject 执行前快照）
        final_decision      = 'No Decision'
        final_model_type    = model_type
        final_subtype       = 'N/A'
        final_g_at_decision = initial_g[:]
        final_k_at_decision = row['k']
        prev_model          = model_type

        while not done and step_count < 20:
            state_vec     = encoder.encode(state_dict)
            valid_actions = env.get_valid_actions(state_dict)

            with torch.no_grad():
                q = agent.model(
                    torch.FloatTensor(state_vec).unsqueeze(0))[0].clone()
                for a_idx in range(env.action_dim):
                    if a_idx not in valid_actions:
                        q[a_idx] = -float('inf')
                action_idx = int(torch.argmax(q).item())

            action_name = env.actions[action_idx]
            actions_taken.append(action_name)

            # ★ 在执行 accept/reject 前，快照当前状态（此时 model_type 是真实的）
            if action_name in ('a_accept', 'a_reject'):
                final_model_type    = state_dict.get('model_type', model_type)
                final_g_at_decision = state_dict.get('g', initial_g)[:]
                final_k_at_decision = state_dict.get('k', 'final')

            next_state_dict, reward, done = env.step(state_dict, action_idx)
            total_reward += reward

            # ★ accept/reject 执行后，重新采样 subtype（与环境内部逻辑一致）
            if action_name in ('a_accept', 'a_reject'):
                decision_key   = 'accept' if action_name == 'a_accept' else 'reject'
                final_subtype  = env._select_subtype(final_model_type, decision_key)
                final_decision = 'Accept' if action_name == 'a_accept' else 'Reject'

            # tracker 记录转移
            if done:
                next_model = final_model_type
            else:
                if env.phase == 'select':
                    next_model = 's0'
                else:
                    next_model = next_state_dict.get('model_type', prev_model)
            tracker.record(prev_model, action_name, next_model)

            # state_history 记录完整三元组
            state_history.append((
                next_model,
                next_state_dict.get('g', [0.25] * 4)[:],
                next_state_dict.get('k', 'final'),
            ))

            prev_model  = next_model
            state_dict  = next_state_dict
            step_count += 1

        # ── 3e. 整理结果 ──────────────────────────────────
        res = row.to_dict()

        # 初始信息
        res['initial_model_type'] = model_type
        res['initial_g']          = str([f'{x:.3f}' for x in initial_g])
        res['initial_k']          = row['k']

        # 动作序列
        res['actions_sequence'] = ' -> '.join(actions_taken)

        # ★ state_trajectory：格式 M4[ban0.98|k=final] -> ...
        traj_parts = []
        for (mt, g, k) in state_history:
            dominant = tier_names[int(np.argmax(g))][:3]
            conf     = max(g)
            traj_parts.append(f"{mt}[{dominant}{conf:.2f}|k={k}]")
        res['state_trajectory'] = ' -> '.join(traj_parts)

        res['total_steps']  = step_count
        res['total_reward'] = round(total_reward, 2)

        # ★ 最终状态使用决策前快照（而非 _terminal_state 占位值）
        res['final_model_type'] = final_model_type
        res['final_g']          = str([f'{x:.3f}' for x in final_g_at_decision])
        res['final_k']          = final_k_at_decision

        # ★ 决策结果
        res['decision']      = final_decision
        # terminal_subtype 格式：M4_a
        res['terminal_subtype'] = (f"{final_model_type}_{final_subtype}"
                                   if final_subtype != 'N/A' else 'N/A')
        # decision_full 格式：Accept_M4_a / Reject_M2_b / No Decision
        res['decision_full'] = (f"{final_decision}_{final_model_type}_{final_subtype}"
                                if final_decision != 'No Decision' else 'No Decision')

        results.append(res)

        # 进度显示
        if (index + 1) % 1000 == 0:
            recent = [r['total_reward'] for r in results[-1000:]]
            print(f"  已处理 {index + 1} 条 | 近1000条平均奖励: {np.mean(recent):.2f}")





    # ── 4. 保存转移概率矩阵 ───────────────────────────────
    tracker.save_to_csv("transition_v2c.csv")





    # ── 5. 保存推理结果 ───────────────────────────────────
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n✅ 推理结果已保存至: {output_csv}")

    # ── 6. 统计分析 ───────────────────────────────────────
    print("\n" + "=" * 80)
    print("📊 测试统计")
    print("=" * 80)

    print("\n【决策分布】")
    for d, c in output_df['decision'].value_counts().items():
        print(f"  {d}: {c} ({c / len(output_df) * 100:.1f}%)")

    print("\n【完整决策标识分布（decision_full）】")
    for d, c in output_df['decision_full'].value_counts().items():
        print(f"  {d}: {c} ({c / len(output_df) * 100:.1f}%)")

    print("\n【terminal_subtype 分布】")
    decided_df = output_df[output_df['decision'].isin(['Accept', 'Reject'])]
    if len(decided_df) > 0:
        for st, c in decided_df['terminal_subtype'].value_counts().items():
            print(f"  {st}: {c} ({c / len(decided_df) * 100:.1f}%)")

    print("\n【初始模型识别分布】")
    for m, c in output_df['initial_model_type'].value_counts().items():
        print(f"  {m}: {c} ({c / len(output_df) * 100:.1f}%)")

    print("\n【各模型决策倾向】")
    for model in ['M1', 'M2', 'M3', 'M4']:
        sub = output_df[output_df['initial_model_type'] == model]
        if len(sub) == 0:
            continue
        n          = len(sub)
        accept_pct = (sub['decision'] == 'Accept').sum() / n * 100
        reject_pct = (sub['decision'] == 'Reject').sum() / n * 100
        nd_pct     = (sub['decision'] == 'No Decision').sum() / n * 100
        avg_r      = sub['total_reward'].mean()
        avg_steps  = sub['total_steps'].mean()
        print(f"  {model}(n={n}): Accept {accept_pct:.1f}% | Reject {reject_pct:.1f}% | "
              f"No Decision {nd_pct:.1f}% | Avg Reward {avg_r:.2f} | Avg Steps {avg_steps:.1f}")

    print("\n【性能指标】")
    print(f"  平均步数:   {output_df['total_steps'].mean():.2f}")
    print(f"  平均奖励:   {output_df['total_reward'].mean():.2f}")
    print(f"  奖励标准差: {output_df['total_reward'].std():.2f}")
    print(f"  最高奖励:   {output_df['total_reward'].max():.2f}")
    print(f"  最低奖励:   {output_df['total_reward'].min():.2f}")

    print("\n【动作使用频率】")
    all_actions = []
    for seq in output_df['actions_sequence']:
        all_actions.extend(str(seq).split(' -> '))
    for action, count in pd.Series(all_actions).value_counts().items():
        print(f"  {action}: {count} ({count / len(all_actions) * 100:.1f}%)")

    print("=" * 80 + "\n")
    return output_df


# ── 入口 ──────────────────────────────────────────────────
if __name__ == "__main__":
    INPUT_FILE   = "comprehensive_test_data.csv"
    OUTPUT_FILE  = "results_v2c.csv"
    MODEL_FILE   = "agent_v1.pth"
    ROWS_TO_TEST = 30000  # None 表示全部行

    run_csv_test(INPUT_FILE, OUTPUT_FILE, MODEL_FILE, num_rows=ROWS_TO_TEST)