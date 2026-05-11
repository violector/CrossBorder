import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os


# ----------------- 1. 风险分类与状态编码 -----------------
class RiskClassifier:
    @staticmethod
    def compute_g(d, b, r, c, m):
        # 基础分权重：[Minimal, Limited, High, Unacceptable]
        score = np.array([1.0, 0.5, 0.1, 0.0])
        if b == 'categorisation': score += [0, 0, 0.5, 5.0]
        if b == 'remote_id': score += [0, 0.1, 2.0, 0.5]
        if r == 'high-risk': score += [0, 0.1, 3.0, 0]
        if c == 'non_compliant': score += [0, 0.5, 1.0, 0]
        if d in ['multimodal', 'video']: score += [0, 0.5, 1.0, 0]
        if m == 'optimal': score += [0.5, 1.0, 1.5, 0]

        exp_s = np.exp(score - np.max(score))
        return (exp_s / exp_s.sum()).tolist()


class StateEncoder:
    def __init__(self):
        self.d_map = {'structured': 0, 'unstructured_text': 1, 'image': 2, 'video': 3, 'audio': 4, 'multimodal': 5}
        self.b_map = {'none': 0, 'verification': 1, 'remote_id': 2, 'categorisation': 3}
        self.r_map = {'false': 0, 'high-risk': 1}
        self.c_map = {'EU_only': 0, 'adequacy': 1, 'inadequate_SCC': 2, 'non_compliant': 3}
        self.k_map = {'start': 0, 'evidence': 1, 'proposed': 2, 'final': 3}
        self.m_map = {'optimal': 0, 'suboptimal': 1, 'insufficient': 2}

    def encode(self, state_dict):
        d, b, r, c = state_dict['d'], state_dict['b'], state_dict['r'], state_dict['c']
        m, k = state_dict['m'], state_dict['k']

        g_vector = state_dict.get('g', RiskClassifier.compute_g(d, b, r, c, m))
        vec = [self.d_map[d], self.b_map[b], self.r_map[r], self.c_map[c], self.k_map[k], self.m_map[m]]
        vec.extend(g_vector)
        vec.extend(state_dict.get('m_metrics', [0.8, 0.8, 0.8, 0.1]))
        return np.array(vec, dtype=np.float32)


# ----------------- 2. 基于参数化概率的环境 -----------------
class CrossBorderDQNMDP:
    def __init__(self):
        self.encoder = StateEncoder()
        self.actions = ['a_next', 'a_eval', 'a_mitig', 'a_accept', 'a_reject']
        self.action_dim = len(self.actions)
        self.state_dim = 14

        # 奖励设置
        self.R_CORRECT, self.R_MISCLASS = 70, -100
        self.R_XB, self.R_INFO = -40, -5
        self.R_eval, self.R_eval_M = -3, -1
        self.R_mitig, self.R_mitig_M = -3, -1

        # 转移概率参数表
        self.P_INIT = (['M1', 'M2', 'M3', 'M4'], [0.25, 0.25, 0.25, 0.25])
        self.P_EVAL = {
            'M1': (['M1', 'M2'], [0.9, 0.1]),
            'M2': (['M2', 'M3'], [0.9, 0.1]),
            'M3': (['M3', 'M2', 'M4'], [0.7, 0.25, 0.05]),
            'M4': (['M4'], [1.0])
        }
        self.P_MITIG = {
            'M1': (['M1'], [1.0]),
            'M2': (['M1', 'M3', 'M2'], [0.2, 0.2, 0.6]),
            'M3': (['M2', 'M3'], [0.5, 0.5]),
            'M4': (['M4', 'M2'], [0.8, 0.2])
        }
        self.P_DECISION_SUCCESS = {'M1': 0.9, 'M2': 0.7, 'M3': 0.6, 'M4': 0.1}

        # 状态模板
        self.templates = {
            'M1': {'d': 'structured', 'b': 'none', 'r': 'false', 'c': 'EU_only', 'k': 'proposed', 'm': 'optimal'},
            'M2': {'d': 'multimodal', 'b': 'remote_id', 'r': 'high-risk', 'c': 'adequacy', 'k': 'proposed',
                   'm': 'suboptimal'},
            'M3': {'d': 'multimodal', 'b': 'remote_id', 'r': 'high-risk', 'c': 'non_compliant', 'k': 'proposed',
                   'm': 'suboptimal'},
            'M4': {'d': 'multimodal', 'b': 'categorisation', 'r': 'high-risk', 'c': 'adequacy', 'k': 'proposed',
                   'm': 'insufficient'}
        }

    def reset(self):
        self.current_model_name = 's0'
        return self._get_state_dict('s0')

    def _get_state_dict(self, name):
        # 兼容 Accept_M1, Accept_M2... 以及 Reject 和 s0
        if name == 's0' or name == 'Reject' or name.startswith('Accept_'):
            return {
                'd': 'structured', 'b': 'none', 'r': 'false', 'c': 'EU_only',
                'k': 'final', 'm': 'optimal',
                'g': [0.0] * 4,  # 终止状态 g 向量清零或设为特定值
                'm_metrics': [0.0] * 4
            }

        # 正常中间状态 M1-M4
        state = self.templates[name].copy()
        state['g'] = RiskClassifier.compute_g(state['d'], state['b'], state['r'], state['c'], state['m'])
        return state

    def get_model_type(self, s):
        if s.get('b') == 'categorisation': return 'M4'
        if s.get('c') in ['non_compliant', 'inadequate_SCC']: return 'M3'
        if s.get('d') in ['multimodal', 'video'] or s.get('r') == 'high-risk': return 'M2'
        return 'M1'

    def step(self, current_state_dict, action_idx):
        action = self.actions[action_idx]
        curr_name = self.get_model_type(current_state_dict) if self.current_model_name != 's0' else 's0'
        next_name, reward, done = curr_name, self.R_INFO, False
        temp_g = current_state_dict.get('g', [0.25] * 4)

        if curr_name == 's0':
            if action == 'a_next':
                next_name = np.random.choice(self.P_INIT[0], p=self.P_INIT[1])
            else:
                return self._get_state_dict('s0'), -200, True

        elif action == 'a_eval':
            reward = self.R_eval if curr_name in ['M3', 'M4'] else self.R_eval_M
            next_name = np.random.choice(self.P_EVAL[curr_name][0], p=self.P_EVAL[curr_name][1])
            temp_g = self._update_g_randomly(temp_g)

        elif action == 'a_mitig':
            reward = self.R_mitig if curr_name in ['M3', 'M4'] else self.R_mitig_M
            next_name = np.random.choice(self.P_MITIG[curr_name][0], p=self.P_MITIG[curr_name][1])


        elif action in ['a_accept', 'a_reject']:

            done = True

            # 随机决定当前属于哪个子类场景 (a-d)

            sub_type = np.random.choice(self.sub_types)

            p_success = self.P_DECISION_SUCCESS[curr_name]

            if action == 'a_accept':

                is_correct = np.random.random() < p_success

                reward = self.R_CORRECT if is_correct else self.R_MISCLASS

                next_name = f"Accept_{curr_name}_{sub_type}"

            else:

                is_correct_rej = np.random.random() > p_success

                reward = self.R_CORRECT if is_correct_rej else (self.R_MISCLASS / 2)

                next_name = f"Reject_{curr_name}_{sub_type}"

        self.current_model_name = next_name
        res_state = self._get_state_dict(next_name) if (done or next_name == 's0') else {**self.templates[next_name],
                                                                                         'g': temp_g}
        return res_state, reward, done

    def _update_g_randomly(self, g):
        new_g = np.array(g) + np.random.normal(0, 0.05, 4)
        new_g = np.clip(new_g, 0, 1)
        return (new_g / new_g.sum()).tolist()


# ----------------- 3. DQN 核心 -----------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(),
                                 nn.Linear(64, action_dim))

    def forward(self, x): return self.net(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim, self.action_dim = state_dim, action_dim
        self.memory = deque(maxlen=10000)
        self.gamma, self.epsilon = 0.99, 1.0
        self.epsilon_min, self.epsilon_decay = 0.01, 0.996
        self.batch_size, self.target_update_iter = 64, 100
        self.learn_step_counter = 0

        self.model = QNetwork(state_dim, action_dim)
        self.target_model = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def act(self, state):
        if np.random.rand() <= self.epsilon: return random.randrange(self.action_dim)
        with torch.no_grad(): return torch.argmax(self.model(torch.FloatTensor(state).unsqueeze(0))[0]).item()

    def replay(self):
        if len(self.memory) < self.batch_size: return
        batch = random.sample(self.memory, self.batch_size)
        s, a, r, ns, d = zip(*batch)

        s, a, r, ns, d = torch.FloatTensor(np.array(s)), torch.LongTensor(a).view(-1, 1), \
                         torch.FloatTensor(r).view(-1, 1), torch.FloatTensor(np.array(ns)), torch.FloatTensor(d).view(
            -1, 1)

        curr_q = self.model(s).gather(1, a)
        max_next_q = self.target_model(ns).max(1)[0].detach().view(-1, 1)
        target_q = r + (self.gamma * max_next_q * (1 - d))

        loss = nn.MSELoss()(curr_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_iter == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        if self.epsilon > self.epsilon_min: self.epsilon *= self.epsilon_decay


# ----------------- 4. 训练函数 -----------------
def train():
    env = CrossBorderDQNMDP()
    agent = DQNAgent(env.state_dim, env.action_dim)
    save_path = "best_agent_moreModel_1.pth"
    rewards_history = []
    best_mean = -float('inf')

    print("Training Dynamic-MDP Agent...")
    for e in range(1500):  # 增加轮数以应对随机性
        state_dict = env.reset()
        state_vec = env.encoder.encode(state_dict)
        total_r, done, steps = 0, False, 0

        while not done and steps < 40:
            action = agent.act(state_vec)
            next_dict, reward, done = env.step(state_dict, action)
            next_vec = env.encoder.encode(next_dict)

            agent.memory.append((state_vec, action, reward, next_vec, done))
            state_vec, state_dict = next_vec, next_dict
            total_r += reward
            steps += 1
            agent.replay()

        rewards_history.append(total_r)

        # 保存表现最优的模型
        if (e + 1) >= 50:
            current_mean = np.mean(rewards_history[-50:])
            if current_mean > best_mean:
                best_mean = current_mean
                agent.save(save_path)
                print(f"⭐ New Best! Ep {e + 1}: Mean Reward {best_mean:.2f}")

        if (e + 1) % 50 == 0:
            print(f"Ep {e + 1}: Last 50 Avg = {np.mean(rewards_history[-50:]):.1f}, Eps = {agent.epsilon:.2f}")

    return agent


if __name__ == "__main__":
    trained_agent = train()