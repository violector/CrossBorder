import torch
import ast
import os
from random.radom_update import DQNAgent, CrossBorderDQNMDP, StateEncoder
import pandas as pd
from collections import defaultdict


class TransitionTracker:
    def __init__(self):
        # 存储格式: { (state, action): {next_state: count} }
        self.counts = defaultdict(lambda: defaultdict(int))

    def record(self, s, a, ns):
        self.counts[(s, a)][ns] += 1

    def get_probabilities(self):
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
        df = self.get_probabilities()
        df.to_csv(filename, index=False)
        print(f"转移概率矩阵已保存至: {filename}")


def run_csv_test(input_csv, output_csv, model_path, num_rows=None):
    # --- 1. 初始化环境与 Agent ---
    env = CrossBorderDQNMDP()
    state_dim = 14
    action_dim = 5
    agent = DQNAgent(state_dim, action_dim)
    encoder = StateEncoder()

    # 加载预训练权重
    if os.path.exists(model_path):
        agent.load(model_path)
        agent.model.eval()  # 切换到评估模式
    else:
        print(f"Error: 找不到模型文件 {model_path}")
        return

    # --- 2. 读取 CSV 数据 ---
    df = pd.read_csv(input_csv)

    # 如果指定了行数，则截取
    if num_rows is not None:
        df = df.head(num_rows)

    results = []

    print(f"开始测试，总计 {len(df)} 条数据...")
    tracker = TransitionTracker()

    # --- 3. 逐行推理 ---
    for index, row in df.iterrows():
        # 解析数据
        custom_input = {
            'd': row['d'],
            'b': row['b'],
            'r': row['r'],
            'c': row['c'],
            'k': row['k'],
            'm': row['m'],
            # 将字符串 "[0.1, 0.2...]" 转为 list
            'm_metrics': ast.literal_eval(row['m_metrics'])
        }

        # 记录初始识别出的模型类型
        initial_model = env.get_model_type(custom_input)
        current_model_name = env.get_model_type(custom_input)
        current_state_dict = custom_input
        done = False
        step_count = 0
        actions_taken = []
        total_reward = 0

        # 重置环境内部状态（如拒绝计数器）
        env.reset()
        # 手动修正初始状态为CSV中的输入，而非s0
        env.current_model_name = initial_model

        # 决策循环
        while not done and step_count < 20:  # 防止死循环
            state_vec = encoder.encode(current_state_dict)

            with torch.no_grad():
                q_values = agent.model(torch.FloatTensor(state_vec).unsqueeze(0))
                action_idx = torch.argmax(q_values[0]).item()

            action_name = env.actions[action_idx]
            actions_taken.append(action_name)

            # 环境执行
            next_state_dict, reward, done = env.step(current_state_dict, action_idx)
            next_model_name = env.current_model_name
            tracker.record(current_model_name, action_name, next_model_name)
            current_model_name = next_model_name
            total_reward += reward
            current_state_dict = next_state_dict
            step_count += 1
        tracker.save_to_csv("transition_probabilities_random_update.csv")
        # --- 4. 收集结果 ---
        res_entry = row.to_dict()  # 保留原始输入
        res_entry['initial_identified_model'] = initial_model
        res_entry['actions_sequence'] = " -> ".join(actions_taken)
        res_entry['total_steps'] = step_count
        res_entry['final_state'] = env.current_model_name
        res_entry['total_reward'] = round(total_reward, 2)
        results.append(res_entry)

    # --- 5. 写入 CSV ---
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"测试完成！结果已保存至: {output_csv}")


if __name__ == "__main__":
    # 配置参数
    INPUT_FILE = "transformed_data.csv"  # 你的原始数据文件
    OUTPUT_FILE = "results_random_update.csv"  # 结果输出文件
    MODEL_FILE = "best_agent_bayesian_eval.pth"  # 训练好的模型路径

    # 控制读取行数：None 为全部读取，数字(如 5) 为读取前5行
    ROWS_TO_TEST = 66000

    run_csv_test(INPUT_FILE, OUTPUT_FILE, MODEL_FILE, num_rows=ROWS_TO_TEST)