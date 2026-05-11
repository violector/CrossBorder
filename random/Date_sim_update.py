import pandas as pd
import torch
import ast
import os
import numpy as np
from collections import defaultdict
from radom_new import DQNAgent, CrossBorderDQNMDP, StateEncoder, RiskClassifier


class TransitionTracker:
    """转移概率追踪器"""

    def __init__(self):
        self.counts = defaultdict(lambda: defaultdict(int))

    def record(self, s, a, ns):
        """记录状态转移"""
        self.counts[(s, a)][ns] += 1

    def get_probabilities(self):
        """计算转移概率"""
        prob_results = []
        for (s, a), next_states in self.counts.items():
            total = sum(next_states.values())
            for ns, count in next_states.items():
                prob_results.append({
                    'current_state': s,
                    'action': a,
                    'next_state': ns,
                    'count': count,
                    'probability': round(count / total, 4)
                })
        return pd.DataFrame(prob_results)

    def save_to_csv(self, filename):
        """保存转移概率矩阵"""
        df = self.get_probabilities()
        df.to_csv(filename, index=False)
        print(f"✅ 转移概率矩阵已保存至: {filename}")


def run_csv_test(input_csv, output_csv, model_path, num_rows=None):
    """
    在CSV数据集上测试训练好的Agent

    参数:
        input_csv: 输入数据文件路径
        output_csv: 输出结果文件路径
        model_path: 训练好的模型路径
        num_rows: 测试行数（None表示全部）
    """
    # --- 1. 初始化环境与 Agent ---
    env = CrossBorderDQNMDP()

    # 关键修改：state_dim = 15 (包含eval_count)
    state_dim = 16  # 6 categorical + 4 g + 4 m_metrics + 1 eval_count
    action_dim = 5
    agent = DQNAgent(state_dim, action_dim)
    encoder = StateEncoder()

    # 加载预训练权重
    if os.path.exists(model_path):
        agent.load(model_path)
        agent.model.eval()  # 设置为评估模式
        agent.epsilon = 0.0  # 关闭探索，使用纯贪婪策略
        print(f"✓ 已加载模型: {model_path}")
        print(f"✓ 设置为评估模式 (epsilon=0.0)")
    else:
        print(f"❌ Error: 找不到模型文件 {model_path}")
        return

    # --- 2. 读取数据 ---
    print("📂 读取数据集...")
    df = pd.read_csv(input_csv)

    # 数据验证
    print(f"✓ 数据集总行数: {len(df)}")
    print(f"✓ 数据列: {list(df.columns)}")

    # 检查必需列
    required_cols = ['d', 'b', 'r', 'c', 'k', 'm', 'm_metrics']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ 缺少必需列: {missing_cols}")
        return

    # 显示前几行样本
    print("\n📊 数据样本（前3行）:")
    print(df.head(3).to_string())
    print()

    # 数据分布统计
    print("📈 数据特征分布:")
    for col in ['d', 'b', 'r', 'c', 'm']:
        unique_vals = df[col].value_counts()
        print(f"  {col}: {dict(unique_vals)}")
    print()

    if num_rows is not None:
        df = df.head(num_rows)
        print(f"⚙️  限制测试行数: {num_rows}")

    results = []
    tracker = TransitionTracker()
    print(f"\n🚀 开始推理测试...\n")

    # --- 3. 逐行推理 ---
    for index, row in df.iterrows():
        # 解析 m_metrics（处理CSV中的字符串格式）
        try:
            if isinstance(row['m_metrics'], str):
                # 移除可能的空格，确保正确解析
                m_metrics_str = row['m_metrics'].strip()
                m_metrics = ast.literal_eval(m_metrics_str)
            elif isinstance(row['m_metrics'], list):
                m_metrics = row['m_metrics']
            else:
                m_metrics = [0.8, 0.8, 0.8, 0.1]

            # 确保是4个元素的列表
            if not isinstance(m_metrics, list):
                m_metrics = [0.8, 0.8, 0.8, 0.1]
            elif len(m_metrics) != 4:
                # 如果长度不对，填充或截断
                if len(m_metrics) < 4:
                    m_metrics = m_metrics + [0.1] * (4 - len(m_metrics))
                else:
                    m_metrics = m_metrics[:4]
        except Exception as e:
            print(f"⚠️  行 {index}: m_metrics 解析失败，使用默认值。错误: {e}")
            m_metrics = [0.8, 0.8, 0.8, 0.1]

        # 关键修复 1: 使用贝叶斯先验计算初始 g 值
        # 这是训练模型所期望的输入格式
        try:
            initial_g = RiskClassifier.compute_initial_g(
                row['d'], row['b'], row['r'], row['c'], row['m']
            )
        except Exception as e:
            print(f"⚠️  行 {index}: g值计算失败。错误: {e}")
            initial_g = [0.25, 0.25, 0.25, 0.25]

        # 构建初始状态字典（完整版本）
        current_state_dict = {
            'd': row['d'],
            'b': row['b'],
            'r': row['r'],
            'c': row['c'],
            'k': row['k'],
            'm': row['m'],
            'm_metrics': m_metrics,
            'g': initial_g,
            'eval_count': 0  # 新增：评估计数器
        }

        # 关键修复 2: 环境状态同步
        # 确保环境的内部状态与当前状态字典一致
        initial_model = env.get_model_type(current_state_dict)
        env.current_model_name = initial_model

        current_model_name = initial_model
        done = False
        step_count = 0
        actions_taken = []
        state_history = [initial_model]  # 记录状态轨迹
        g_history = [initial_g.copy()]  # 记录g值演化
        k_history = [row['k']]  # 记录k值演化
        total_reward = 0

        # --- 4. 决策循环 ---
        while not done and step_count < 20:
            # 编码状态向量
            state_vec = encoder.encode(current_state_dict)

            # 关键修复 3: 使用动作掩码的推理
            with torch.no_grad():
                state_t = torch.FloatTensor(state_vec).unsqueeze(0)
                q_values = agent.model(state_t)[0]

                # 获取当前状态的有效动作
                valid_actions = env.get_valid_actions(current_state_dict)

                # 掩码无效动作
                masked_q = q_values.clone()
                all_actions = list(range(action_dim))
                for a_idx in all_actions:
                    if a_idx not in valid_actions:
                        masked_q[a_idx] = -float('inf')

                # 选择Q值最大的有效动作
                action_idx = torch.argmax(masked_q).item()

            action_name = env.actions[action_idx]
            actions_taken.append(action_name)

            # 环境执行
            next_state_dict, reward, done = env.step(current_state_dict, action_idx)

            # 记录转移（基于模型类型 M1-M4）
            next_model_name = env.get_model_type(next_state_dict) if not done else env.current_model_name
            tracker.record(current_model_name, action_name, next_model_name)

            # 记录历史
            state_history.append(next_model_name)
            g_history.append(next_state_dict.get('g', [0.25] * 4))
            k_history.append(next_state_dict.get('k', 'final'))

            # 更新状态
            current_model_name = next_model_name
            current_state_dict = next_state_dict
            total_reward += reward
            step_count += 1

        # --- 5. 收集结果 ---
        res_entry = row.to_dict()

        # 基础信息
        res_entry['initial_identified_model'] = initial_model
        res_entry['initial_g'] = str([f'{x:.3f}' for x in initial_g])
        res_entry['initial_k'] = row['k']

        # 决策轨迹
        res_entry['actions_sequence'] = " -> ".join(actions_taken)
        res_entry['state_trajectory'] = " -> ".join(state_history)
        res_entry['total_steps'] = step_count

        # 最终状态
        res_entry['final_state'] = current_model_name
        res_entry['final_g'] = str([f'{x:.3f}' for x in g_history[-1]])
        res_entry['final_k'] = k_history[-1]

        # 奖励信息
        res_entry['total_reward'] = round(total_reward, 2)

        # 判断决策类型
        if 'Accept' in current_model_name:
            res_entry['decision'] = 'Accept'
            res_entry['terminal_subtype'] = current_model_name.split('_')[
                -1] if '_' in current_model_name else 'unknown'
        elif 'Reject' in current_model_name:
            res_entry['decision'] = 'Reject'
            res_entry['terminal_subtype'] = current_model_name.split('_')[
                -1] if '_' in current_model_name else 'unknown'
        else:
            res_entry['decision'] = 'No Decision'
            res_entry['terminal_subtype'] = 'N/A'

        results.append(res_entry)

        # 进度显示
        if (index + 1) % 1000 == 0:
            print(f"已处理 {index + 1} 条... 平均奖励: {np.mean([r['total_reward'] for r in results[-1000:]]):.2f}")




    # --- 6. 保存转移概率矩阵 ---
    tracker.save_to_csv("transition_v7.csv")





    # --- 7. 写入最终 CSV ---
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n✅ 测试完成！推理结果已保存至: {output_csv}")

    # --- 8. 统计分析 ---
    print("\n" + "=" * 80)
    print("📊 测试统计")
    print("=" * 80)

    # 决策统计
    decision_counts = output_df['decision'].value_counts()
    print("\n【决策分布】")
    for decision, count in decision_counts.items():
        pct = count / len(output_df) * 100
        print(f"  {decision}: {count} ({pct:.1f}%)")

    # 子类分布（仅针对有决策的）
    decided_df = output_df[output_df['decision'].isin(['Accept', 'Reject'])]
    if len(decided_df) > 0:
        subtype_counts = decided_df['terminal_subtype'].value_counts()
        print("\n【终端子类分布】")
        for subtype, count in subtype_counts.items():
            pct = count / len(decided_df) * 100
            print(f"  {subtype}: {count} ({pct:.1f}%)")

    # 模型初始识别分布
    model_counts = output_df['initial_identified_model'].value_counts()
    print("\n【初始模型识别分布】")
    for model, count in model_counts.items():
        pct = count / len(output_df) * 100
        print(f"  {model}: {count} ({pct:.1f}%)")

    # 平均步数和奖励
    print("\n【性能指标】")
    print(f"  平均步数: {output_df['total_steps'].mean():.2f}")
    print(f"  平均奖励: {output_df['total_reward'].mean():.2f}")
    print(f"  奖励标准差: {output_df['total_reward'].std():.2f}")

    # 按模型类型的决策倾向
    print("\n【各模型决策倾向】")
    for model in ['M1', 'M2', 'M3', 'M4']:
        model_df = output_df[output_df['initial_identified_model'] == model]
        if len(model_df) > 0:
            accept_rate = (model_df['decision'] == 'Accept').sum() / len(model_df) * 100
            reject_rate = (model_df['decision'] == 'Reject').sum() / len(model_df) * 100
            no_decision_rate = (model_df['decision'] == 'No Decision').sum() / len(model_df) * 100
            avg_reward = model_df['total_reward'].mean()
            print(f"  {model}: Accept {accept_rate:.1f}% | Reject {reject_rate:.1f}% | "
                  f"No Decision {no_decision_rate:.1f}% | Avg Reward {avg_reward:.2f}")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    # 配置参数
    INPUT_FILE = "E:\\pythonProject\\transformed_data.csv"
    OUTPUT_FILE = "results_v7.csv"
    MODEL_FILE = "agent_v4.pth"  # 训练好的模型
    ROWS_TO_TEST = None  # 测试全部，None表示所有行

    run_csv_test(INPUT_FILE, OUTPUT_FILE, MODEL_FILE, num_rows=ROWS_TO_TEST)