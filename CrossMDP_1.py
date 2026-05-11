import numpy as np
import random
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
        if m == 'low': score += [0.5, 1.0, 1.5, 0]

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
        self.m_map = {'low': 0, 'medium': 1, 'high': 2}

    def encode(self, state_dict):
        d = state_dict.get('d', 'structured')
        b = state_dict.get('b', 'none')
        r = state_dict.get('r', 'false')
        c = state_dict.get('c', 'EU_only')
        m = state_dict.get('m', 'low')
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
        self.R_CORRECT = 50
        self.R_MISCLASS = -100
        self.R_XB = -40
        self.R_INFO = -5

        # 模型初始特征模板
        self.templates = {
            'M1': {'d': 'structured', 'b': 'none', 'r': 'false', 'c': 'EU_only', 'k': 'start', 'm': 'high',
                   'g': [0.6, 0.3, 0.1, 0.0]},
            'M2': {'d': 'multimodal', 'b': 'remote_id', 'r': 'high-risk', 'c': 'adequacy', 'k': 'start', 'm': 'medium',
                   'g': [0.05, 0.1, 0.7, 0.15]},
            'M3': {'d': 'multimodal', 'b': 'remote_id', 'r': 'high-risk', 'c': 'non_compliant', 'k': 'start',
                   'm': 'medium', 'g': [0.1, 0.2, 0.6, 0.1]},
            'M4': {'d': 'multimodal', 'b': 'categorisation', 'r': 'high-risk', 'c': 'adequacy', 'k': 'start',
                   'm': 'medium', 'g': [0.0, 0.05, 0.3, 0.65]}
        }

    def reset(self):
        self.current_model_name = 's0'
        return self._get_state_dict('s0')

    def _get_state_dict(self, name):

        # 处理终端状态
        if name in ['Reject', 'Accept_M1', 'Accept_M2', 'Accept_M3']:
            return {
                'd': 'structured', 'b': 'none', 'r': 'false', 'c': 'EU_only',
                'k': 'final', 'm': 'low', 'g': [0, 0, 0, 0], 'm_metrics': [0, 0, 0, 0]
            }

        if name == 's0':
            return {'d': 'structured', 'b': 'none', 'r': 'false', 'c': 'EU_only',
                    'k': 'start', 'm': 'low', 'g': [1.0, 0, 0, 0], 'm_metrics': [0, 0, 0, 0]}

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
        reward = 0
        done = False

        # --- 精确转移概率逻辑 ---
        if curr_name == 's0':
            if action == 'a_next':
                next_name = np.random.choice(['M1', 'M2', 'M3'], p=[0.34, 0.33, 0.33])
                reward = self.R_INFO
            else:
                reward = -100  # 非法动作惩罚

        elif curr_name == 'M1':
            if action == 'a_eval':
                next_name = np.random.choice(['M1', 'M2'], p=[0.9, 0.1])
                reward = self.R_INFO
            elif action == 'a_mitig':
                next_name = 'M1'  # 改进公平性，概率仍为1.0
                reward = self.R_INFO
            elif action == 'a_accept':
                done = True
                # 0.9 正确，0.1 误判
                reward = (0.9 * self.R_CORRECT) + (0.1 * self.R_MISCLASS)
            elif action == 'a_reject':
                next_name = 's0'

        elif curr_name == 'M2':
            if action == 'a_eval':
                next_name = np.random.choice(['M2', 'M3', 'M4'], p=[0.8, 0.1, 0.1])
                reward = self.R_INFO
            elif action == 'a_mitig':
                next_name = np.random.choice(['M1', 'M2', 'M3'], p=[0.2, 0.6, 0.2])
                reward = self.R_INFO
            elif action == 'a_accept':
                done = True
                reward = (0.7 * self.R_CORRECT) + (0.3 * self.R_MISCLASS)
            elif action == 'a_reject':
                next_name = 's0'

        elif curr_name == 'M3':
            if action == 'a_eval':
                next_name = np.random.choice(['M3', 'M2', 'M4'], p=[0.7, 0.15, 0.15])
                reward = self.R_INFO
            elif action == 'a_mitig':
                next_name = np.random.choice(['M2', 'M3'], p=[0.5, 0.5])
                reward = self.R_INFO
            elif action == 'a_accept':
                done = True
                # 涉及跨境惩罚
                reward = (0.6 * self.R_CORRECT) + (0.4 * self.R_MISCLASS) + self.R_XB
            elif action == 'a_reject':
                next_name = 's0'

        elif curr_name == 'M4':
            if action == 'a_eval':
                next_name = 'M4'
                reward = self.R_INFO
            elif action == 'a_mitig':
                next_name = np.random.choice(['M2', 'M4'], p=[0.3, 0.7])
                reward = self.R_INFO
            elif action == 'a_accept':
                done = True
                next_name = 'Reject'
                reward = self.R_MISCLASS
            elif action == 'a_reject':
                next_name = 's0'

        self.current_model_name = next_name
        return self._get_state_dict(next_name), reward, done


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

    #普通
    # def act(self, state_vec):
    #     if np.random.rand() <= self.epsilon:
    #         return random.randrange(self.action_dim)
    #     with torch.no_grad():
    #         state_t = torch.FloatTensor(state_vec).unsqueeze(0)
    #         return torch.argmax(self.model(state_t)[0]).item()
    #
    # def replay(self, batch_size):
    #     if len(self.memory) < batch_size: return
    #     minibatch = random.sample(self.memory, batch_size)
    #     for s, a, r, ns, d in minibatch:
    #         target = r
    #         if not d:
    #             with torch.no_grad():
    #                 target = r + self.gamma * torch.max(self.model(torch.FloatTensor(ns).unsqueeze(0))[0]).item()
    #
    #         curr_q = self.model(torch.FloatTensor(s).unsqueeze(0))
    #         target_q = curr_q.clone()
    #         target_q[0][a] = target
    #
    #         loss = nn.MSELoss()(curr_q, target_q)
    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()
    #     if self.epsilon > self.epsilon_min: self.epsilon *= self.epsilon_decay

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

# ----------------- 4. 训练与可视化函数 -----------------
# def train_dqn(episodes=200):
#     env = CrossBorderDQNMDP()  # 使用上一轮定义的精确转移概率环境
#     agent = DQNAgent(env.state_dim, env.action_dim)
#     rewards_history = []
#
#     print("开始训练 DQN...")
#     for e in range(episodes):
#         state_dict = env.reset()
#         total_reward = 0
#         for time in range(20):
#             state_vec = env.encoder.encode(state_dict)
#             action = agent.act(state_vec)
#             next_state_dict, reward, done = env.step(state_dict, action)
#             next_state_vec = env.encoder.encode(next_state_dict)
#
#             agent.memory.append((state_vec, action, reward, next_state_vec, done))
#             state_dict = next_state_dict
#             total_reward += reward
#             if done: break
#
#         agent.replay(32)
#         rewards_history.append(total_reward)
#         if e % 20 == 0: print(f"Episode: {e}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")
#
#     return agent, rewards_history, env
#
#
# def visualize(rewards_history):
#     # 1. 绘制奖励曲线
#     plt.figure(figsize=(10, 5))
#     plt.plot(rewards_history)
#     plt.title("DQN Training Rewards")
#     plt.xlabel("Episode")
#     plt.ylabel("Accumulated Reward")
#     plt.grid(True)
#     plt.show()
#
#
# # 执行训练与测试
# trained_agent, history, mdp_env = train_dqn(600)
# visualize(history)


# ----------------- 4. 训练过程 -----------------

def train_dqn(episodes=500, batch_size=64):
    env = CrossBorderDQNMDP()
    agent = DQNAgent(env.state_dim, env.action_dim)
    encoder = StateEncoder()

    rewards_history = []

    print("开始训练...")
    for e in range(episodes):
        state_dict = env.reset()
        state_vec = encoder.encode(state_dict)
        total_reward = 0
        done = False
        step_count = 0

        while not done and step_count < 30:  # 增加步数上限防止死循环
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

        if (e + 1) % 50 == 0:
            print(f"Episode: {e + 1}/{episodes}, Score: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    def run_instance_test(agent, encoder, env, custom_input):
        """
        输入: 自定义的状态字典
        输出: Agent 的决策序列和最终结果
        """
        print("\n" + "=" * 30)
        print("🚀 开始实例决策测试")
        print(f"输入数据: {custom_input}")
        print("=" * 30)

        current_state_dict = custom_input
        step_count = 0
        done = False
        decision_path = []

        # 将 Agent 设置为评估模式（不进行随机探索）
        agent.model.eval()

        while not done and step_count < 10:
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

            print(f"Step {step_count}: 执行 [{action_name}] -> 奖励: {reward}, 下一状态模型: {env.current_model_name}")

            current_state_dict = next_state_dict

        print("=" * 30)
        print(f"🏁 最终终端状态: {env.current_model_name}")
        print(f"最终状态数据预览: {current_state_dict}")
        print("=" * 30)
        return decision_path
    # 1. 准备输入
    test_data = {
        'd': 'structured', 'b': 'none', 'r': 'false', 'c': 'EU_only',
        'k': 'start', 'm': 'high', 'm_metrics': [0.95, 0.95, 0.9, 0.05]
    }
    test1 = {'d': 'multimodal', 'b': 'remote_id', 'r': 'high-risk', 'c': 'non_compliant',
                      'k': 'start', 'm': 'medium', 'm_metrics': [0.7, 0.7, 0.6, 0.3]}

    path = run_instance_test(agent, encoder, env, test_data)
    path1 = run_instance_test(agent, encoder, env, test1)
    # for e in range(episodes):
    #     state_dict = env.reset()
    #     state_vec = encoder.encode(state_dict)
    #     total_reward = 0
    #     done = False
    #
    #     while not done:
    #         # 1. 选择动作
    #         action_idx = agent.act(state_vec)
    #
    #         # 2. 与环境交互
    #         next_state_dict, reward, done = env.step(state_dict, action_idx)
    #         next_state_vec = encoder.encode(next_state_dict)
    #
    #         # 3. 存储经验
    #         agent.memory.append((state_vec, action_idx, reward, next_state_vec, done))
    #
    #         # 4. 更新状态
    #         state_dict = next_state_dict
    #         state_vec = next_state_vec
    #         total_reward += reward
    #
    #         # 5. 经验回放 (训练网络)
    #         agent.replay(batch_size)
    #
    #     rewards_history.append(total_reward)
    #
    #     if (e + 1) % 50 == 0:
    #         print(f"Episode: {e + 1}/{episodes}, Score: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

    return agent, rewards_history, path


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

trained_agent, history, path = train_dqn(episodes=800)
plot_learning_curve(history)






# ----------------- 5. 实际使用测试代码 -----------------
# def thagent(agent, custom_state_dict):
#         """
#         输入一个特定的状态字典，查看模型的决策
#         """
#         encoder = StateEncoder()
#         # 1. 编码
#         state_vec = encoder.encode(custom_state_dict)
#
#         # 2. 预测 Q 值
#         agent.model.eval()  # 切换到评估模式
#         with torch.no_grad():
#             state_t = torch.FloatTensor(state_vec).unsqueeze(0)
#             q_values = agent.model(state_t)
#             action_idx = torch.argmax(q_values[0]).item()
#
#         # 3. 解析结果
#         actions = ['a_next', 'a_eval', 'a_mitig', 'a_accept', 'a_reject']
#         chosen_action = actions[action_idx]
#
#         print("-" * 30)
#         print(f"输入状态特征: d={custom_state_dict.get('d')}, c={custom_state_dict.get('c')}, m={custom_state_dict.get('m')}")
#         print(f"模型预测动作: 【{chosen_action}】")
#         # 打印每个动作的置信度（Q值）
#         for i, a_name in enumerate(actions):
#             print(f"  - {a_name}: {q_values[0][i].item():.4f}")
# def thagent(agent, episodes=5):

    # env = CrossBorderDQNMDP()
    # encoder = StateEncoder()
    # print("\n--- 实际测试运行 ---")
    #
    # for i in range(episodes):
    #     state_dict = env.reset()
    #     print(f"\n测试轮次 {i + 1}:")
    #     done = False
    #     steps = 0
    #
    #     while not done and steps < 10:
    #         state_vec = encoder.encode(state_dict)
    #         # 测试时关闭随机探索 (epsilon=0)
    #         with torch.no_grad():
    #             action_idx = torch.argmax(agent.model(torch.FloatTensor(state_vec).unsqueeze(0))[0]).item()
    #
    #         action_name = env.actions[action_idx]
    #         curr_model = env.current_model_name
    #
    #         next_state_dict, reward, done = env.step(state_dict, action_idx)
    #
    #         print(f"  [状态: {curr_model}] -> 动作: {action_name} -> 奖励: {reward}")
    #
    #         state_dict = next_state_dict
    #         steps += 1
    #
    #     if done:
    #         print(f"  结果: 流程结束。")


# ----------------- 6. 执行主程序 -----------------

# if __name__ == "__main__":
#     # 1. 运行训练
#     trained_agent, history = train_dqn(episodes=300, batch_size=32)
#
#     # 2. 绘制训练曲线
#     plt.figure(figsize=(10, 5))
#     plt.plot(history)
#     plt.title("DQN Training Rewards")
#     plt.xlabel("Episode")
#     plt.ylabel("Total Reward")
#     plt.grid(True)
#     plt.show()
#
#     easy_case = {
#         'd': 'structured', 'b': 'none', 'r': 'false', 'c': 'EU_only',
#         'k': 'start', 'm': 'high', 'm_metrics': [0.95, 0.95, 0.9, 0.05]
#     }
#
#     risk_case = {
#         'd': 'multimodal', 'b': 'categorisation', 'r': 'high-risk', 'c': 'non_compliant',
#         'k': 'start', 'm': 'low', 'm_metrics': [0.4, 0.5, 0.3, 0.8]
#     }
#
#     # 3. 运行测试
#     thagent(trained_agent, easy_case)
#     thagent(trained_agent, risk_case)

# def train_and_run_tests():
#     # 初始化环境、编码器和智能体
#     env = CrossBorderDQNMDP()
#     agent = DQNAgent(env.state_dim, env.action_dim)
#     encoder = StateEncoder()
#
#     # --- A. 训练阶段 ---
#     episodes = 1000
#     batch_size = 32
#     total_reward = 0
#     print("正在基于 MDP 逻辑训练 Agent...")
#     for e in range(episodes):
#         state_dict = env.reset()
#         state_vec = encoder.encode(state_dict)
#         done = False
#         while not done:
#             action_idx = agent.act(state_vec)
#             next_state_dict, reward, done = env.step(state_dict, action_idx)
#             next_state_vec = encoder.encode(next_state_dict)
#             agent.memory.append((state_vec, action_idx, reward, next_state_vec, done))
#             total_reward += reward
#             state_vec = next_state_vec
#             state_dict = next_state_dict
#             agent.replay(batch_size)
#         if (e + 1) % 100 == 0:
#             print(f"Episode: {e + 1}/{episodes}, Score: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
#
#     # --- B. 实际使用测试（输入具体特征字典） ---
#     print("\n" + "=" * 50)
#     print("【特征驱动决策测试】")
#
#     # 定义你要求的具体输入
#     test_cases = [
#         {
#             "name": "M1 (Structured/Low-risk)",
#             "data": {'d': 'structured', 'b': 'none', 'r': 'false', 'c': 'EU_only',
#                      'k': 'start', 'm': 'high', 'm_metrics': [0.95, 0.95, 0.9, 0.05]}
#         },
#         {
#             "name": "M3 (Non-compliant/Cross-border)",
#             "data": {'d': 'multimodal', 'b': 'remote_id', 'r': 'high-risk', 'c': 'non_compliant',
#                      'k': 'start', 'm': 'medium', 'm_metrics': [0.7, 0.7, 0.6, 0.3]}
#         },
#         {
#             "name": "M4 (Prohibited/Emotion-recognition)",
#             "data": {'d': 'multimodal', 'b': 'categorisation', 'r': 'high-risk', 'c': 'adequacy',
#                      'k': 'start', 'm': 'low', 'm_metrics': [0.5, 0.5, 0.4, 0.8]}
#         }
#     ]
#
#     for case in test_cases:
#         print(f"\n>>> 测试场景: {case['name']}")
#         current_state = case['data']
#         env.current_model_name = 'Custom_Input'  # 标记为自定义输入
#
#         path = []
#         done = False
#         step_limit = 10
#
#         while not done and step_limit > 0:
#             curr_model_type = env.get_model_type(current_state)
#             state_vec = encoder.encode(current_state)
#
#             # 纯利用模式预测最优动作
#             agent.model.eval()
#             with torch.no_grad():
#                 q_values = agent.model(torch.FloatTensor(state_vec).unsqueeze(0))[0]
#                 action_idx = torch.argmax(q_values).item()
#
#             action_name = env.actions[action_idx]
#
#             # 获取当前逻辑下的模型分类（用于输出展示）
#             # 注意：环境 step 内部会基于 current_state 的特征来应用逻辑
#             next_state, reward, done = env.step(current_state, action_idx)
#             next_model_type = env.get_model_type(next_state) if not done else env.current_model_name
#             # 记录并打印轨迹
#             path.append(action_name)
#
#             print(
#                 f"  选择模型 [{curr_model_type}] 执行 {action_name.ljust(8)} -> 获得奖励: {reward:<4} -> 进入模型: {next_model_type}")
#
#             current_state = next_state
#             step_limit -= 1
#
#         print(f"  决策序列: {' -> '.join(path)}")
#         # 根据最后动作或环境状态判断最终终端
#         if "Reject" in env.current_model_name or (path[-1] == "a_accept" and "M4" in case['name']):
#             print(f"  最终归宿: 🛑 Reject [{curr_model_type}] (拒绝/误判拦截)")
#         elif path[-1] == "a_accept":
#             print(f"  最终归宿: ✅ Accept [{curr_model_type}] (接受部署)")
#         else:
#             print("  最终归宿: ⚠️ 流程中断/持续评估")
#
#
# # ----------------- 执行 -----------------
# if __name__ == "__main__":
#     train_and_run_tests()