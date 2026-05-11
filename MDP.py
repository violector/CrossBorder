import numpy as np
import random

import matplotlib.pyplot as plt


# ----------------- 1. 定义状态、动作和参数 -----------------
class CrossBorderAIMDP:
    def __init__(self):
        # 1.1 状态空间 (S)
        self.states = {
            's0': 0, 'M1': 1, 'M2': 2, 'M3': 3, 'M4': 4,
            'Accept_M1': 5, 'Accept_M2': 6, 'Accept_M3': 7, 'Reject': 8
        }
        self.num_states = len(self.states)
        self.state_names = {v: k for k, v in self.states.items()}
        self.terminal_states = [5, 6, 7, 8]

        # 1.2 动作空间 (A)
        self.actions = {
            'a_next': 0, 'a_eval': 1, 'a_mitig': 2, 'a_accept': 3, 'a_reject': 4
        }
        self.num_actions = len(self.actions)
        self.action_names = {v: k for k, v in self.actions.items()}

        # 1.3 奖励参数 (R)
        # 根据文本定义：Rcorrect > 0, Rmisclass << 0, Rxb < 0, Rinfo < 0
        self.R_CORRECT = 50  # 正确分类的奖励
        self.R_MISCLASS = -100  # 错误分类的惩罚 (严重)
        self.R_XB = -40  # 跨境非合规惩罚
        self.R_INFO = -5  # 信息成本/延迟惩罚

        # 1.4 初始化转移概率 P(s'|s, a) 和奖励 R(s, a, s')
        self.transitions = self._define_transitions()
        self.rewards = self._define_rewards()

    # ----------------- 2. 定义转移概率 P(s'|s, a) -----------------
    def _define_transitions(self):
        T = {s: {a: [] for a in range(self.num_actions)} for s in range(self.num_states)}

        def s_id(name):
            return self.states[name]

        s0 = s_id('s0')
        T[s0][self.actions['a_next']] = [(s_id('M1'), 0.34), (s_id('M2'), 0.33), (s_id('M3'), 0.33)]

        M1 = s_id('M1')
        T[M1][self.actions['a_eval']] = [(M1, 0.9), (s_id('M2'), 0.1)]
        T[M1][self.actions['a_mitig']] = [(M1, 1.0)]
        T[M1][self.actions['a_accept']] = [(s_id('Accept_M1'), 0.9),
                                           (s_id('Reject'), 0.1)]
        T[M1][self.actions['a_reject']] = [(s0, 1.0)]

        M2 = s_id('M2')
        T[M2][self.actions['a_eval']] = [(M2, 0.8), (s_id('M3'), 0.1), (s_id('M4'), 0.1)]
        T[M2][self.actions['a_mitig']] = [(s_id('M1'), 0.2), (M2, 0.6), (s_id('M3'), 0.2)]
        T[M2][self.actions['a_accept']] = [(s_id('Accept_M2'), 0.7), (s_id('Reject'), 0.3)]
        T[M2][self.actions['a_reject']] = [(s0, 1.0)]

        M3 = s_id('M3')
        T[M3][self.actions['a_eval']] = [(M3, 0.7), (s_id('M2'), 0.15), (s_id('M4'), 0.15)]
        T[M3][self.actions['a_mitig']] = [(s_id('M2'), 0.5), (M3, 0.5)]
        T[M3][self.actions['a_accept']] = [(s_id('Accept_M3'), 0.6), (s_id('Reject'), 0.4)]
        # Note: M3 implies cross-border friction, so acceptance has higher misclassification risk.
        T[M3][self.actions['a_reject']] = [(s0, 1.0)]

        M4 = s_id('M4')
        T[M4][self.actions['a_eval']] = [(M4, 1.0)]
        T[M4][self.actions['a_mitig']] = [(s_id('M2'), 0.3), (M4, 0.7)]
        T[M4][self.actions['a_accept']] = [(s_id('Reject'), 1.0)]  # Prohibited models cannot be accepted
        T[M4][self.actions['a_reject']] = [(s0, 1.0)]

        for s in self.terminal_states:
            for a in range(self.num_actions):
                T[s][a] = [(s, 1.0)]  # Remain in terminal state

        return T

    # ----------------- 3. 定义奖励 R(s, a, s') -----------------
    def _define_rewards(self):
        R = {s: {a: 0.0 for a in range(self.num_actions)} for s in range(self.num_states)}

        def s_id(name):
            return self.states[name]

        for s in range(self.num_states):
            for a in [self.actions['a_next'], self.actions['a_eval'], self.actions['a_mitig']]:
                if s not in self.terminal_states:
                    R[s][a] = self.R_INFO

        M1 = s_id('M1')
        R[M1][self.actions['a_accept']] = (0.9 * self.R_CORRECT) + (
                    0.1 * self.R_MISCLASS)  # Correct (0.9), Misclass (0.1)

        M2 = s_id('M2')
        R[M2][self.actions['a_accept']] = (0.7 * self.R_CORRECT) + (0.3 * self.R_MISCLASS)

        M3 = s_id('M3')
        R[M3][self.actions['a_accept']] = (0.6 * self.R_CORRECT) + (0.4 * self.R_MISCLASS) + self.R_XB

        M4 = s_id('M4')
        R[M4][self.actions[
            'a_accept']] = self.R_MISCLASS  # Attempting to accept a prohibited model is a guaranteed severe penalty.


        return R

    # ----------------- 4. MDP 核心函数：执行动作 -----------------
    def step(self, state, action):
        if state in self.terminal_states:
            return state, 0.0, True

        # 1. 获取奖励
        reward = self.rewards[state][action]

        # 2. 根据转移概率选择下一个状态
        transitions = self.transitions[state][action]
        if not transitions:
            # Should not happen in this defined MDP for non-terminal states
            return state, reward, False

        next_states, probabilities = zip(*transitions)
        next_state = random.choices(next_states, weights=probabilities, k=1)[0]

        # 3. 判断是否是终端状态
        done = next_state in self.terminal_states

        return next_state, reward, done


# ----------------- 5. Q-Learning 智能体 -----------------
class QLearningAgent:
    def __init__(self, mdp, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.9995, min_epsilon=0.01):
        self.mdp = mdp
        self.alpha = alpha  # 学习率 (Learning Rate)
        self.gamma = gamma  # 折扣因子 (Discount Factor)
        self.epsilon = epsilon  # 探索率 (Exploration Rate)
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Q表: 初始化为零
        self.q_table = np.zeros((mdp.num_states, mdp.num_actions))

    def choose_action(self, state):
        if random.random() < self.epsilon:
            # 探索: 随机选择一个动作
            return random.choice(list(self.mdp.actions.values()))
        else:
            # 利用: 选择Q值最大的动作
            return np.argmax(self.q_table[state, :])

    def learn(self, state, action, reward, next_state, done):
        # 1. 计算 Q(s', a') 的最大值
        if done:
            max_future_q = 0
        else:
            max_future_q = np.max(self.q_table[next_state, :])

        # 2. Q-Learning 更新公式
        old_q = self.q_table[state, action]
        new_q = old_q + self.alpha * (reward + self.gamma * max_future_q - old_q)
        self.q_table[state, action] = new_q

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


# ----------------- 6. 训练函数 -----------------
def train_agent(mdp, agent, episodes):
    rewards_history = []

    for episode in range(episodes):
        current_state = mdp.states['s0']
        done = False
        episode_reward = 0

        while not done:
            action = agent.choose_action(current_state)
            next_state, reward, done = mdp.step(current_state, action)

            agent.learn(current_state, action, reward, next_state, done)

            current_state = next_state
            episode_reward += reward

        agent.decay_epsilon()
        rewards_history.append(episode_reward)

    return rewards_history


# ----------------- 7. 执行与可视化 -----------------
# 1. 初始化 MDP 和 Agent
mdp = CrossBorderAIMDP()
agent = QLearningAgent(mdp, alpha=0.1, gamma=0.9, epsilon_decay=0.99995)

# 2. 训练智能体
print("开始训练 Q-Learning 智能体...")
rewards_history = train_agent(mdp, agent, episodes=100000)
print("训练完成。")


# 3. 策略可视化
def visualize_policy(agent, mdp):
    print("\n## 🎯 Q-Learning 最优策略 (π*(s))")
    print("-" * 50)

    policy = {}
    for state_id in range(mdp.num_states):
        state_name = mdp.state_names[state_id]
        if state_id in mdp.terminal_states:
            policy[state_name] = "TERMINAL"
            continue

        best_action_id = np.argmax(agent.q_table[state_id, :])
        best_action_name = mdp.action_names[best_action_id]

        # 过滤掉无法执行的动作 (例如 s0 只能 a_next)
        valid_actions = [mdp.action_names[a] for a, transitions in mdp.transitions[state_id].items() if transitions]

        # 如果当前最佳动作无效，选择有效动作中的最佳
        if best_action_name not in valid_actions:
            # 找到有效动作中 Q 值最高的
            valid_q_values = {a: agent.q_table[state_id, mdp.actions[a]] for a in valid_actions}
            best_action_name = max(valid_q_values, key=valid_q_values.get)

        policy[state_name] = best_action_name

    # 打印策略表
    print("状态 | 最佳动作")
    print("--- | ---")
    for s, a in policy.items():
        print(f"{s:<3} | {a}")

    # 绘制奖励历史图
    plt.figure(figsize=(10, 6))
    plt.plot(np.convolve(rewards_history, np.ones(500) / 500, mode='valid'), label='500-Episode Moving Average Reward')
    plt.title('Q-Learning 训练过程中的累计奖励 (平滑)')
    plt.xlabel('训练周期 (Episode)')
    plt.ylabel('平均累计奖励')
    plt.grid(True)
    plt.legend()
    plt.show()


visualize_policy(agent, mdp)


def predict_next_action(agent, state_name):
    """
    根据智能体的 Q 表格，预测给定状态下应采取的最佳动作。

    Args:
        agent (QLearningAgent): 训练好的 Q-Learning 智能体实例。
        state_name (str): 待预测的状态名称 (例如 'M1', 'M3', 's0')。

    Returns:
        str: 最佳动作的名称。
    """
    try:
        # 1. 将状态名称转换为状态 ID
        state_id = agent.mdp.states[state_name]
    except KeyError:
        return f"错误：状态名称 '{state_name}' 无效。请使用 {list(agent.mdp.states.keys())} 中的一个。"

    if state_id in agent.mdp.terminal_states:
        return f"状态 {state_name} 是终端状态，决策过程已结束。"

    # 2. 从 Q-table 中获取该状态下所有动作的 Q 值
    q_values = agent.q_table[state_id, :]

    # 3. 找到 Q 值最大的动作 ID
    best_action_id = np.argmax(q_values)
    best_action_name = agent.mdp.action_names[best_action_id]

    # 4. 确保选中的是有效动作 (防止 Q-table 随机初始化导致选到 s0 的 a_eval/a_mitig)
    # 检查最佳动作是否在当前状态的有效转移中
    valid_actions_ids = [a for a, transitions in agent.mdp.transitions[state_id].items() if transitions]

    if best_action_id not in valid_actions_ids:
        # 如果最佳动作无效，则从有效动作中选择 Q 值最高的
        valid_q_values = {a: q_values[a] for a in valid_actions_ids}
        best_action_id = max(valid_q_values, key=valid_q_values.get)
        best_action_name = agent.mdp.action_names[best_action_id]

    print(f"--- Q 值概览 ({state_name}) ---")
    action_q_map = {agent.mdp.action_names[i]: q_values[i] for i in range(agent.mdp.num_actions)}
    print(action_q_map)
    print("----------------------------")

    return best_action_name
























































# class CrossBorderMDP:
#     def __init__(self):
#         self.actions = ['a_next', 'a_eval', 'a_mitig', 'a_accept', 'a_reject']
#
#         self.current_state_name = 's0'
#         self.current_features = None  # (d, b, m, r, c, j, k, g)
#
#         # 奖励函数，数值暂时不确定
#         self.R_correct = 2
#         self.R_misclass = -1
#         self.R_xb = -1
#         self.R_info = 1
#
#         # 模拟环境的“真值” (Ground Truth)，用于计算奖励
#         # 格式: {模型名: (真实风险等级, 是否跨境合规)}
#         # 0:Minimal, 1:Limited, 2:High, 3:Unacceptable
#         self.ground_truth = {
#             'M1': {'tier': 0, 'xb_compliant': True},
#             'M2': {'tier': 2, 'xb_compliant': True},
#             'M3': {'tier': 2, 'xb_compliant': False},  # 初始是不合规的
#             'M4': {'tier': 3, 'xb_compliant': True}
#         }
#
#     def reset(self):
#         self.current_state_name = 's0'
#         self.current_features = None
#         return self.current_state_name
#
#     def _get_risk_tier_from_g(self, g_vector):
#         """根据网关风险评分g推断当前的预测等级 (用于Agent的决策)"""
#         # g = (min, lim, high, ban)
#         return np.argmax(g_vector)
#
#     def step(self, action):
#         """
#         执行动作，返回: (next_state, reward, done, info)
#         """
#         state = self.current_state_name
#         reward = 0
#         done = False
#         info = {}
#
#         # --- 1. S0: Start State ---
#         if state == 's0':
#             if action == 'a_next':
#                 # 概率分布: M1(0.34), M2(0.33), M3(0.33)
#                 next_model = np.random.choice(['M1', 'M2', 'M3'], p=[0.34, 0.33, 0.33])
#                 self.current_state_name = next_model
#                 # 在此设置初始特征向量 (简化处理，仅设置g)
#                 if next_model == 'M1':
#                     self.current_features = {'g': [0.6, 0.3, 0.1, 0.0], 'c': 'adequate'}
#                 elif next_model == 'M2':
#                     self.current_features = {'g': [0.05, 0.1, 0.7, 0.15], 'c': 'adequate'}
#                 elif next_model == 'M3':
#                     self.current_features = {'g': [0.1, 0.2, 0.6, 0.1], 'c': 'inadequate'}
#             else:
#                 # 在s0只能做a_next，否则惩罚
#                 reward = -10
#
#         elif action == 'a_accept':
#             done = True
#             model_id = state
#             true_tier = self.ground_truth[model_id]['tier']
#             pred_tier = self._get_risk_tier_from_g(self.current_features['g'])
#
#             # 特殊情况：M4 永远不能被接受
#             if model_id == 'M4':
#                 reward = self.R_misclass  # 严重惩罚
#                 self.current_state_name = 'Reject'
#             else:
#                 if true_tier == pred_tier:
#                     reward += self.R_correct
#                 else:
#                     reward += self.R_misclass
#
#                 # 检查跨境合规性 (针对 M3)
#                 if not self.ground_truth[model_id]['xb_compliant']:
#                     reward += self.R_xb
#
#                 self.current_state_name = f"Accept_{model_id}"
#
#         elif action == 'a_reject':
#             # 拒绝当前模型，返回 s0
#             self.current_state_name = 's0'
#             reward = 0  # 或者给予微小的惩罚以鼓励通过模型而不是拒绝所有
#
#         # --- 3. Evaluate Action ---
#         elif action == 'a_eval':
#             reward = self.R_info
#
#             if state == 'M1':
#                 # M1 eval: 0.9 stay M1, 0.1 become M2
#                 if np.random.rand() < 0.1:
#                     self.current_state_name = 'M2'
#                     self.current_features = {'g': [0.05, 0.1, 0.7, 0.15], 'c': 'adequate'}
#                 # else: stay M1, update g (simulated)
#
#             elif state == 'M2':
#                 # M2 eval: 0.8 stay, 0.1 -> M3, 0.1 -> M4
#                 rand = np.random.rand()
#                 if rand < 0.1:
#                     self.current_state_name = 'M3'
#                     self.current_features = {'g': [0.1, 0.2, 0.6, 0.1], 'c': 'inadequate'}
#                 elif rand < 0.2:
#                     self.current_state_name = 'M4'
#                     self.current_features = {'g': [0.0, 0.05, 0.3, 0.65], 'c': 'adequate'}
#
#             elif state == 'M3':
#                 # M3 eval: 0.7 stay, 0.15 -> M2, 0.15 -> M4
#                 rand = np.random.rand()
#                 if rand < 0.15:
#                     self.current_state_name = 'M2'
#                     self.ground_truth['M3']['xb_compliant'] = True  # 发现其实是合规的
#                 elif rand < 0.30:
#                     self.current_state_name = 'M4'
#
#             elif state == 'M4':
#                 # M4 eval: confirm prohibited
#                 self.current_features['g'] = [0.0, 0.0, 0.0, 1.0]
#
#         # --- 4. Mitigate Action ---
#         elif action == 'a_mitig':
#             reward = self.R_info
#
#             if state == 'M1':
#                 # 改善公平性，g vector 变得更好
#                 self.current_features['g'] = [0.7, 0.25, 0.05, 0.0]
#
#             elif state == 'M2':
#                 # 0.2 -> M1, 0.6 stay M2, 0.2 -> M3
#                 rand = np.random.rand()
#                 if rand < 0.2:
#                     self.current_state_name = 'M1'
#                     self.current_features['g'] = [0.6, 0.3, 0.1, 0.0]
#                 elif rand > 0.8:
#                     self.current_state_name = 'M3'
#
#             elif state == 'M3':
#                 # 0.5 -> M2 (Fix cross-border), 0.5 stay
#                 if np.random.rand() < 0.5:
#                     self.current_state_name = 'M2'
#                     self.current_features['c'] = 'adequate'  # 修复了数据问题
#                     # Update ground truth for simulation consistency
#                     self.ground_truth['M3']['xb_compliant'] = True
#
#             elif state == 'M4':
#                 # 0.3 -> M2 (Remove emotion rec), 0.7 stay
#                 if np.random.rand() < 0.3:
#                     self.current_state_name = 'M2'
#                     self.current_features['g'] = [0.05, 0.1, 0.7, 0.15]
#
#         return self.current_state_name, reward, done, {}