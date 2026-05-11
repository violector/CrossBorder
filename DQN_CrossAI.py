import os

import numpy as np
import random

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt

class RiskClassifier:
    """根据特征 (d,b,m,r,c) 模拟计算风险概率向量 g"""

    @staticmethod
    def compute_g(d, b, r, c, m):
        # 简化版统计分类器：根据特征累加权重
        # 权重：[Minimal, Limited, High, Unacceptable]
        score = np.array([1.0, 0.5, 0.1, 0.0])  # 基础分

        if b == 'categorisation': score += [0, 0, 0.5, 5.0]
        if b == 'remote_id': score += [0, 0.1, 2.0, 0.5]
        if r == 'high-risk': score += [0, 0.1, 3.0, 0]
        if c == 'non_compliant': score += [0, 0.5, 1.0, 0]
        if d in ['multimodal', 'video']: score += [0, 0.5, 1.0, 0]
        if m == 'optimal': score += [0.5, 1.0, 1.5, 0]

        # Softmax 归一化为概率
        exp_s = np.exp(score - np.max(score))
        return exp_s / exp_s.sum()

class StateEncoder:
    def __init__(self):
        # 定义分类变量的映射
        self.d_map = {'structured': 0, 'unstructured_text': 1, 'image': 2, 'video': 3, 'audio': 4, 'multimodal': 5}
        self.b_map = {'none': 0, 'verification': 1, 'remote_id': 2, 'categorisation': 3}
        self.r_map = {'false': 0, 'high-risk': 1}
        self.c_map = {'EU_only': 0, 'adequacy': 1, 'inadequate_SCC': 2, 'non_compliant': 3}
        self.k_map = {'start': 0, 'evidence': 1, 'proposed': 2, 'final': 3}
        self.m_map = {'optimal': 0, 'suboptimal': 1, 'insufficient': 2}

    def encode(self, state_dict):
        d = state_dict.get('d', 'structured')
        b = state_dict.get('b', 'none')
        r = state_dict.get('r', 'false')
        c = state_dict.get('c', 'EU_only')
        m = state_dict.get('m', 'optimal')
        k = state_dict.get('k', 'start')

        # 动态计算或获取 g
        g_vector = state_dict.get('g', RiskClassifier.compute_g(d, b, r, c, m))

        vec = [
            self.d_map[d], self.b_map[b], self.r_map[r],
            self.c_map[c], self.k_map[k], self.m_map[m]
        ]
        vec.extend(g_vector)
        vec.extend(state_dict.get('m_metrics', [0.8, 0.8, 0.8, 0.1]))
        return np.array(vec, dtype=np.float32)


# ----------------- 2. 基于精确转移概率的环境 -----------------
class CrossBorderDQNMDP:
    def __init__(self):
        self.encoder = StateEncoder()
        self.actions = ['a_next', 'a_eval', 'a_mitig', 'a_accept', 'a_reject']
        self.action_dim = len(self.actions)
        self.state_dim = 14

        # 奖励设置
        self.R_CORRECT = 70
        self.R_MISCLASS = -100
        self.R_XB = -40
        self.R_INFO = -5
        self.R_eval = -3
        self.R_eval_M = -1
        self.R_mitig  = -3
        self.R_mitig_M = -1

        # 模型初始特征模板
        self.templates = {
            'M1': {'d': 'structured', 'b': 'none', 'r': 'false', 'c': 'EU_only', 'k': 'start', 'm': 'optimal',
                   'g': [0.6, 0.3, 0.1, 0.0]},
            'M2': {'d': 'multimodal', 'b': 'remote_id', 'r': 'high-risk', 'c': 'adequacy', 'k': 'start', 'm': 'suboptimal',
                   'g': [0.05, 0.1, 0.7, 0.15]},
            'M3': {'d': 'multimodal', 'b': 'remote_id', 'r': 'high-risk', 'c': 'non_compliant', 'k': 'start',
                   'm': 'suboptimal', 'g': [0.1, 0.2, 0.6, 0.1]},
            'M4': {'d': 'multimodal', 'b': 'categorisation', 'r': 'high-risk', 'c': 'adequacy', 'k': 'start',
                   'm': 'insufficient', 'g': [0.0, 0.05, 0.3, 0.65]}
        }

    def reset(self):
        self.current_model_name = 's0'
        self.reject_count = 0  # 重置计数器
        return self._get_state_dict('s0')

    def _update_g_randomly(self, g_vector):
        """模拟 eval 后的 g 值微调"""
        noise = np.random.normal(0, 0.05, size=len(g_vector))
        new_g = np.array(g_vector) + noise
        new_g = np.clip(new_g, 0.0, 1.0)
        return (new_g / new_g.sum()).tolist()

    def _get_state_dict(self, name):

        if name == 'Review_Required':
            return {
                'd': 'structured', 'b': 'none', 'r': 'false', 'c': 'EU_only',
                'k': 'final', 'm': 'optimal', 'g': [0, 0, 0, 0], 'm_metrics': [0, 0, 0, 0]
            }

        # 处理终端状态
        if name in ['Reject', 'Accept_M1', 'Accept_M2', 'Accept_M3']:
            return {
                'd': 'structured', 'b': 'none', 'r': 'false', 'c': 'EU_only',
                'k': 'final', 'm': 'optimal', 'g': [0, 0, 0, 0], 'm_metrics': [0, 0, 0, 0]
            }

        if name == 's0':
            return {'d': 'structured', 'b': 'none', 'r': 'false', 'c': 'EU_only',
                    'k': 'start', 'm': 'optimal', 'g': [1.0, 0, 0, 0], 'm_metrics': [0, 0, 0, 0]}

            # 获取基础模板
        state = self.templates[name].copy()

        # 动态计算该模板的 g 值
        state['g'] = RiskClassifier.compute_g(
            state['d'], state['b'], state['r'], state['c'], state['m']
        )
        return state

    def get_model_type(self, s):
        """根据特征自动识别当前状态对应的逻辑分类"""
        if s.get('b') == 'categorisation': return 'M4'
        if s.get('c') in ['non_compliant', 'inadequate_SCC']: return 'M3'
        if s.get('d') in ['multimodal', 'video'] or s.get('r') == 'high-risk': return 'M2'
        return 'M1'

        # return self.templates[name].copy()

    def step(self, current_state_dict, action_idx):
        action = self.actions[action_idx]
        curr_name = self.get_model_type(current_state_dict) if self.current_model_name != 's0' else 's0'
        next_name = curr_name
        reward = self.R_INFO
        done = False

        res_state = current_state_dict.copy()
        temp_g = current_state_dict.get('g', [0.25] * 4)

        if action == 'a_reject' and curr_name != 's0':
            self.reject_count += 1
            if self.reject_count >= 2:
                print(">>> [System Notification] 数据需要重新审查 (Review Required) <<<")
                next_name = 'Review_Required'
                done = True
                res_state = self._get_state_dict(next_name)
                self.current_model_name = next_name
                return res_state, -10, done  # 给予一定负反馈
            else:
                # 第一次拒绝：修改 k 值为 evidence，并继续留在当前模型类型的决策中
                next_name = 's0'
                res_state = self._get_state_dict('s0')
                res_state['k'] = 'evidence'
                self.current_model_name = 's0'
                # 这里不修改 next_name，保持在当前 M1-M4 状态，但属性已变
                reward = self.R_INFO
                return res_state, reward, done

        # --- 精确转移概率逻辑 ---
        if curr_name == 's0':
            if action == 'a_next':
                next_name = np.random.choice(['M1', 'M2', 'M3'], p=[0.34, 0.33, 0.33])
                reward = self.R_INFO
            else:
                next_name = 's0'
                reward = -200
                done = False

        elif curr_name == 'M1':
            if action == 'a_eval':
                next_name = np.random.choice(['M1', 'M2'], p=[0.9, 0.1])
                if next_name == 'M1': temp_g = self._update_g_randomly(temp_g)
                reward = self.R_eval_M
            elif action == 'a_mitig':
                next_name = 'M1'  # 改进公平性，概率仍为1.0
                temp_g = [0.7, 0.25, 0.05, 0.0]
                reward = self.R_mitig_M
            elif action == 'a_accept':
                done = True
                next_name = 'Accept_M1'  # <--- 明确终端状态
                # 0.9 正确，0.1 误判
                reward = (0.9 * self.R_CORRECT) + (0.1 * self.R_MISCLASS)
            elif action == 'a_next':
                reward = self.R_XB


        elif curr_name == 'M2':
            if action == 'a_eval':
                next_name = np.random.choice(['M2', 'M3', 'M4'], p=[0.8, 0.1, 0.1])
                if next_name == 'M2': temp_g = self._update_g_randomly(temp_g)
                reward = self.R_eval_M
            elif action == 'a_mitig':
                next_name = np.random.choice(['M1', 'M2', 'M3'], p=[0.2, 0.6, 0.2])
                reward = self.R_mitig_M
            elif action == 'a_accept':
                done = True
                next_name = 'Accept_M2'  # <--- 明确终端状态
                reward = (0.7 * self.R_CORRECT) + (0.3 * self.R_MISCLASS)
            elif action == 'a_next':
                reward = self.R_XB

        elif curr_name == 'M3':
            if action == 'a_eval':
                next_name = np.random.choice(['M3', 'M2', 'M4'], p=[0.7, 0.15, 0.15])
                if next_name == 'M3': temp_g = self._update_g_randomly(temp_g)
                reward = self.R_eval
            elif action == 'a_mitig':
                next_name = np.random.choice(['M2', 'M3'], p=[0.5, 0.5])
                reward = self.R_mitig
            elif action == 'a_accept':
                done = True
                next_name = 'Accept_M3'  # <--- 明确终端状态
                # 涉及跨境惩罚
                reward = (0.6 * self.R_CORRECT) + (0.4 * self.R_MISCLASS) + self.R_XB
            elif action == 'a_next':
                reward = self.R_XB

        elif curr_name == 'M4':
            if action == 'a_eval':
                temp_g = self._update_g_randomly(temp_g)
                next_name = 'M4'
                reward = self.R_eval
            elif action == 'a_mitig':
                next_name = np.random.choice(['M2', 'M4'], p=[0.3, 0.7])
                reward = self.R_mitig
            elif action == 'a_accept':
                done = True
                next_name = 'Reject'
                reward = self.R_CORRECT
            elif action == 'a_next':
                reward = self.R_XB

        self.current_model_name = next_name

        if done or next_name == 's0':
            res_state = self._get_state_dict(next_name)
        else:
            res_state = self.templates[next_name].copy()
            res_state['g'] = temp_g

        return res_state, reward, done


# ----------------- 2. DQN 神经网络 -----------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# ----------------- 3. 增强环境与 DQN Agent -----------------
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.memory = deque(maxlen=5000)
        # --- 缓冲区设置 ---
        # self.sim_memory = deque(maxlen=5000)  # 原有的 Sim Buffer
        # self.real_memory = deque(maxlen=5000)  # 新增的 Real Buffer



        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        # self.model = QNetwork(state_dim, action_dim)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        # 改成Action和target网格
        self.batch_size = 64
        self.target_update_iter = 100  # 每隔100次训练更新一次目标网络
        self.learn_step_counter = 0
        self.model = QNetwork(state_dim, action_dim)
        self.target_model = QNetwork(state_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()  # 目标网络只用于预测
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    # A和T的动作
    def act(self, state_vec):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)

        state_t = torch.FloatTensor(state_vec).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_t)
        return torch.argmax(q_values[0]).item()

    def store_transition(self, s, a, r, ns, d):
        self.memory.append((s, a, r, ns, d))

    def save(self, filename):
        """保存模型参数"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'target_model_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)
        print(f"Model saved to: {filename}")

    def load(self, filename):
        """加载模型参数"""
        if os.path.exists(filename):
            checkpoint = torch.load(filename)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.target_model.load_state_dict(checkpoint['target_model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            print(f"已从 {filename} 成功加载模型")
        else:
            print(f"未找到文件: {filename}")

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        # 1. 向量化采样
        minibatch = random.sample(self.memory, self.batch_size)

        # 将数据转换为 Tensor，利用 PyTorch 的并行计算
        states = torch.FloatTensor(np.array([x[0] for x in minibatch]))
        actions = torch.LongTensor(np.array([x[1] for x in minibatch])).view(-1, 1)
        rewards = torch.FloatTensor(np.array([x[2] for x in minibatch])).view(-1, 1)
        next_states = torch.FloatTensor(np.array([x[3] for x in minibatch]))
        dones = torch.FloatTensor(np.array([x[4] for x in minibatch])).view(-1, 1)

        # 2. 计算当前 Q 值 (行为网络)
        curr_q = self.model(states).gather(1, actions)

        # 3. 计算目标 Q 值 (目标网络)
        # 使用 target_model 获取下一状态的最大 Q 值，且不计算梯度
        with torch.no_grad():
            max_next_q = self.target_model(next_states).max(1)[0].view(-1, 1)
            # 如果是终结状态，则没有未来奖励 (1 - dones)
            target_q = rewards + (self.gamma * max_next_q * (1 - dones))

        # 4. 计算损失并优化
        loss = nn.MSELoss()(curr_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 5. 定期更新目标网络
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_iter == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        # 6. 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train_dqn(episodes=500, batch_size=64):
    env = CrossBorderDQNMDP()
    agent = DQNAgent(env.state_dim, env.action_dim)
    encoder = StateEncoder()
    rewards_history = []

    best_mean_reward = -float('inf')
    save_path = "best_dqn_model4.pth"

    print("开始训练...")
    for e in range(episodes):
        state_dict = env.reset()
        state_vec = encoder.encode(state_dict)
        total_reward = 0
        done = False
        step_count = 0

        while not done and step_count < 30:
            action_idx = agent.act(state_vec)
            next_state_dict, reward, done = env.step(state_dict, action_idx)
            next_state_vec = encoder.encode(next_state_dict)

            agent.store_transition(state_vec, action_idx, reward, next_state_vec, done)
            state_dict = next_state_dict
            state_vec = next_state_vec
            total_reward += reward
            step_count += 1
            agent.replay()

        rewards_history.append(total_reward)

        if (e + 1) >= 50:
            current_mean_reward = np.mean(rewards_history[-50:])  # 计算最近50轮的平均奖励

            if current_mean_reward > best_mean_reward:
                best_mean_reward = current_mean_reward
                # 保存最佳模型
                agent.save(save_path)
                print(f"🌟 发现更好的模型！Episode: {e + 1}, Mean Reward: {best_mean_reward:.2f}, 已保存。")

        if (e + 1) % 50 == 0:
            print(f"Episode: {e + 1}/{episodes}, Mean Reward: {np.mean(rewards_history[-50:]):.2f}, Score: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    print(f"\n训练结束。历史最高平均奖励: {best_mean_reward:.2f}")
    return agent, encoder, env, rewards_history


def run_instance_test(agent, encoder, env, custom_input):
    """
    输入: 自定义的状态字典
    输出: Agent 的决策序列和最终结果
    """
    print("\n" + "=" * 30)
    print("🚀 开始实例决策测试")
    print(f"输入数据: {custom_input}")
    print("=" * 30)

    current_modeltype = env.get_model_type(custom_input)
    print(f"First model:{current_modeltype}")

    current_state_dict = custom_input
    step_count = 0
    done = False
    decision_path = []

    # 将 Agent 设置为评估模式（不进行随机探索）
    agent.model.eval()

    while not done and step_count < 15:
        step_count += 1
        # 1. 编码状态
        state_vec = encoder.encode(current_state_dict)

        # 2. Agent 预测决策 (Q-values)
        with torch.no_grad():
            q_values = agent.model(torch.FloatTensor(state_vec).unsqueeze(0))
            action_idx = torch.argmax(q_values[0]).item()

        action_name = env.actions[action_idx]

        # 3. 环境执行动作
        next_state_dict, reward, done = env.step(current_state_dict, action_idx)

        # 记录路径
        decision_path.append({
            "step": step_count,
            "action": action_name,
            "reward": reward,
            "next_model": env.current_model_name
        })

        print(f"Step {step_count}: Action [{action_name}] -> Reward: {reward}, Next_model: {env.current_model_name}")

        current_state_dict = next_state_dict

    print("=" * 30)
    print(f"🏁 Final State: {env.current_model_name}")
    print(f"最终状态数据预览: {current_state_dict}")
    print("=" * 30)
    return decision_path


def plot_learning_curve(rewards):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Episode Reward', color='skyblue', alpha=0.6)
    # 计算滑动平均以平滑曲线
    if len(rewards) > 10:
        smooth_rewards = [np.mean(rewards[max(0, i - 10):i + 1]) for i in range(len(rewards))]
        plt.plot(smooth_rewards, label='MA (10)', color='red')

    plt.title("Agent Training Performance over Time")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # --- 阶段 1: 训练 ---
    trained_agent, encoder, env, history = train_dqn(episodes=900)

    plot_learning_curve(history)

    # 测试案例 A: 低风险/合规路径
    test_data_low = {
        'd': 'structured', 'b': 'none', 'r': 'high-risk', 'c': 'EU_only',
        'k': 'start', 'm': 'suboptimal', 'm_metrics': [0.95, 0.95, 0.9, 0.05]
    }

    # 测试案例 B: 高风险/跨境合规问题
    test_data_high = {
        'd': 'multimodal', 'b': 'remote_id', 'r': 'high-risk', 'c': 'non_compliant', 'k': 'start',
     'm': 'insufficient'
    }

    # 执行测试
    path_a = run_instance_test(trained_agent, encoder, env, test_data_low)
    path_b = run_instance_test(trained_agent, encoder, env, test_data_high)



    # def replay_buffer(self):
    #     # 确保 Buffer 里有足够数据
    #     if len(self.sim_memory) < self.batch_size:
    #         return
    #
    #     # 计算采样数量
    #     n_real = int(self.batch_size * 0.3)
    #     n_sim = self.batch_size - n_real
    #
    #     # 如果 Real Buffer 还没填满 30%，则用 Sim 补齐
    #     if len(self.real_memory) < n_real:
    #         minibatch = random.sample(self.sim_memory, self.batch_size)
    #     else:
    #         real_batch = random.sample(self.real_memory, n_real)
    #         sim_batch = random.sample(self.sim_memory, n_sim)
    #         minibatch = real_batch + sim_batch
    #         random.shuffle(minibatch)  # 打乱混合后的顺序
    #
    #     # 转换为 Tensor (保持原有逻辑)
    #     states = torch.FloatTensor(np.array([x[0] for x in minibatch]))
    #     actions = torch.LongTensor(np.array([x[1] for x in minibatch])).view(-1, 1)
    #     rewards = torch.FloatTensor(np.array([x[2] for x in minibatch])).view(-1, 1)
    #     next_states = torch.FloatTensor(np.array([x[3] for x in minibatch]))
    #     dones = torch.FloatTensor(np.array([x[4] for x in minibatch])).view(-1, 1)
    #
    #     # Q-Learning 更新
    #     curr_q = self.model(states).gather(1, actions)
    #     with torch.no_grad():
    #         max_next_q = self.target_model(next_states).max(1)[0].view(-1, 1)
    #         target_q = rewards + (self.gamma * max_next_q * (1 - dones))
    #
    #     loss = nn.MSELoss()(curr_q, target_q)
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #
    #     self.learn_step_counter += 1
    #     if self.learn_step_counter % self.target_update_iter == 0:
    #         self.target_model.load_state_dict(self.model.state_dict())
    #
    #     if self.epsilon > self.epsilon_min:
    #         self.epsilon *= self.epsilon_decay

# def train_dqn_buffer(episodes=500):
#     env = CrossBorderDQNMDP()
#     agent = DQNAgent(env.state_dim, env.action_dim)
#     encoder = StateEncoder()
#
#     # 如果有之前的模型，可以加载
#     # agent.load("my_agent_v1.pth")
#
#     # 预加载 CSV 真实数据（如果有）
#     # agent.load_real_data_from_csv("real_data.csv", encoder)
#
#     for e in range(episodes):
#         state_dict = env.reset()
#         state_vec = encoder.encode(state_dict)
#         done = False
#
#         while not done:
#             action_idx = agent.act(state_vec)
#             next_state_dict, reward, done = env.step(state_dict, action_idx)
#             next_state_vec = encoder.encode(next_state_dict)
#
#             # --- 只将“刚刚经历的经验”存入 Sim Buffer ---
#             agent.store_sim_transition(state_vec, action_idx, reward, next_state_vec, done)
#
#             state_dict = next_state_dict
#             state_vec = next_state_vec
#
#             # Replay 内部会自动按 3:7 混合采样
#             agent.replay_buffer()
#
#         # 每隔 100 个 episode 保存一次
#         if (e + 1) % 100 == 0:
#             agent.save(f"agent_ep{e + 1}.pth")

    # --- 保存和加载模型 ---
    # def save(self, filename="dqn_agent.pth"):
    #     torch.save({
    #         'model_state_dict': self.model.state_dict(),
    #         'optimizer_state_dict': self.optimizer.state_dict(),
    #         'epsilon': self.epsilon
    #     }, filename)
    #     print(f" Model saved to {filename}")
    #
    # def load(self, filename="dqn_agent.pth"):
    #     if os.path.exists(filename):
    #         checkpoint = torch.load(filename)
    #         self.model.load_state_dict(checkpoint['model_state_dict'])
    #         self.target_model.load_state_dict(self.model.state_dict())
    #         self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #         self.epsilon = checkpoint['epsilon']
    #         print(f"Model loaded from {filename}")
    #     else:
    #         print(" No checkpoint found, starting from scratch.")
    #
    # def load_real_data_from_csv(self, csv_path, encoder):
    #     df = pd.read_csv(csv_path)
    #     for _, row in df.iterrows():
    #         # 1. 构造当前状态向量
    #         state_dict = {
    #             'd': row['d'], 'b': row['b'], 'r': row['r'],
    #             'c': row['c'], 'k': row['k'], 'm': row['m'],
    #             'g': [row['g1'], row['g2'], row['g3'], row['g4']],
    #             'm_metrics': [row['m1'], row['m2'], row['m3'], row['m4']]
    #         }
    #         state_vec = encoder.encode(state_dict)
    #
    #         # 2. 构造下一状态向量
    #         next_state_dict = {
    #             'd': row['next_d'], 'b': row['next_b'], 'r': row['next_r'],
    #             'c': row['next_c'], 'k': row['next_k'], 'm': row['next_m'],
    #             'g': [row['next_g1'], row['next_g2'], row['next_g3'], row['next_g4']],
    #             'm_metrics': [row['m1'], row['m2'], row['m3'], row['m4']]  # 假设指标变化不大
    #         }
    #         next_state_vec = encoder.encode(next_state_dict)
    #
    #         # 3. 存入 Real Buffer
    #         self.real_memory.append((
    #             state_vec,
    #             int(row['action']),
    #             float(row['reward']),
    #             next_state_vec,
    #             int(row['done'])
    #         ))
    #     print(f"成功从 CSV 加载了 {len(self.real_memory)} 条真实经验数据。")