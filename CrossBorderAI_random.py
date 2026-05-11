import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os


# ----------------- 1. 风险分类与状态编码 -----------------
class RiskClassifier:
    """静态风险评估器：提供基础风险感知"""

    @staticmethod
    def compute_g(d, b, r, c, m):
        score = np.array([1.0, 0.5, 0.1, 0.0])  # 基础分
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

        # 确保 g_vector 是4个元素
        if not isinstance(g_vector, list):
            g_vector = [0.25, 0.25, 0.25, 0.25]
        elif len(g_vector) != 4:
            g_vector = g_vector[:4] + [0.0] * (4 - len(g_vector))

        vec = [
            self.d_map[d],
            self.b_map[b],
            self.r_map[r],
            self.c_map[c],
            self.k_map[k],
            self.m_map[m]
        ]
        vec.extend(g_vector)

        # 确保 m_metrics 是4个元素
        m_metrics = state_dict.get('m_metrics', [0.8, 0.8, 0.8, 0.1])
        if len(m_metrics) != 4:
            m_metrics = m_metrics[:4] + [0.0] * (4 - len(m_metrics))
        vec.extend(m_metrics)

        return np.array(vec, dtype=np.float32)


# ----------------- 2. 固定结构 + 随机概率的 MDP 环境 -----------------
class CrossBorderDQNMDP:
    def __init__(self):
        self.encoder = StateEncoder()
        self.actions = ['a_next', 'a_eval', 'a_mitig', 'a_accept', 'a_reject']
        self.action_dim = len(self.actions)
        self.state_dim = 14

        # 奖励设置
        self.R_CORRECT, self.R_MISCLASS = 50, -80
        self.R_XB, self.R_INFO = -30, -3
        self.R_eval, self.R_mitig = -2, -3

        # 子类场景 (a-d)
        self.sub_types = ['a', 'b', 'c', 'd']

        # ========== 核心：定义转移结构（哪些状态可以转移到哪些状态）==========
        # 每次执行动作时，会在这些状态中随机生成概率分布

        # a_next 动作：从 S0 可以到达的状态
        self.INIT_STRUCTURE = ['M1', 'M2', 'M3', 'M4']

        # a_eval 动作：每个状态可以转移到的状态列表
        self.EVAL_STRUCTURE = {
            'M1': ['M1', 'M2'],
            'M2': ['M2', 'M3', 'M4'],
            'M3': ['M3', 'M2', 'M4'],
            'M4': ['M4']
        }

        # a_mitig 动作：每个状态可以转移到的状态列表
        self.MITIG_STRUCTURE = {
            'M1': ['M1'],
            'M2': ['M1', 'M3', 'M2'],
            'M3': ['M2', 'M3'],
            'M4': ['M4', 'M2']
        }

        # a_accept / a_reject 决策成功的基础概率范围（会加入随机扰动）
        self.DECISION_BASE_PROBS = {
            'M1': {'accept': 0.9, 'reject': 0.1},
            'M2': {'accept': 0.7, 'reject': 0.3},
            'M3': {'accept': 0.6, 'reject': 0.4},
            'M4': {'accept': 0.1, 'reject': 0.9}
        }

        self.templates = {
            'M1': {'d': 'structured', 'b': 'none', 'r': 'false', 'c': 'EU_only', 'k': 'proposed', 'm': 'optimal'},
            'M2': {'d': 'multimodal', 'b': 'remote_id', 'r': 'high-risk', 'c': 'adequacy', 'k': 'proposed',
                   'm': 'suboptimal'},
            'M3': {'d': 'multimodal', 'b': 'remote_id', 'r': 'high-risk', 'c': 'non_compliant', 'k': 'proposed',
                   'm': 'suboptimal'},
            'M4': {'d': 'multimodal', 'b': 'categorisation', 'r': 'high-risk', 'c': 'adequacy', 'k': 'proposed',
                   'm': 'insufficient'}
        }

    def _get_random_probs(self, n):
        """生成随机概率分布（归一化）"""
        p = np.random.rand(n)
        return (p / p.sum()).tolist()

    def _get_noisy_prob(self, base_prob, noise_scale=0.1):
        """为基础概率添加噪声，保持在[0,1]范围内"""
        noisy = base_prob + np.random.normal(0, noise_scale)
        return np.clip(noisy, 0.05, 0.95)

    def reset(self):
        self.current_model_name = 's0'
        return self._get_state_dict('s0')

    def _get_state_dict(self, name):
        if name == 's0' or 'Accept' in name or 'Reject' in name:
            return {'d': 'structured', 'b': 'none', 'r': 'false', 'c': 'EU_only',
                    'k': 'final', 'm': 'optimal', 'g': [0.25, 0.25, 0.25, 0.25], 'm_metrics': [0.0, 0.0, 0.0, 0.0]}
        state = self.templates[name].copy()
        state['g'] = RiskClassifier.compute_g(state['d'], state['b'], state['r'], state['c'], state['m'])
        if 'm_metrics' not in state:
            state['m_metrics'] = [0.8, 0.8, 0.8, 0.1]
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

        # 1. 从初始状态开始（a_next）
        if curr_name == 's0':
            if action == 'a_next':
                # 结构固定：只能到 M1/M2/M3/M4
                # 概率随机：每次生成新的概率分布
                random_probs = self._get_random_probs(len(self.INIT_STRUCTURE))
                next_name = np.random.choice(self.INIT_STRUCTURE, p=random_probs)
                print(f"[a_next] S0 -> {next_name} (probs: {[f'{p:.3f}' for p in random_probs]})")
            else:
                # 在 s0 只能执行 a_next
                return self._get_state_dict('s0'), -150, True

        # 2. 评估动作（a_eval）- 固定结构，随机概率
        elif action == 'a_eval':
            reward = self.R_eval
            # 获取当前状态可以转移到的状态列表
            possible_states = self.EVAL_STRUCTURE[curr_name]
            # 每次随机生成新的转移概率
            random_probs = self._get_random_probs(len(possible_states))
            next_name = np.random.choice(possible_states, p=random_probs)
            print(
                f"[a_eval] {curr_name} -> {next_name} (probs: {dict(zip(possible_states, [f'{p:.3f}' for p in random_probs]))})")
            # 评估后更新风险向量
            temp_g = self._update_g_slightly(temp_g)

        # 3. 缓解动作（a_mitig）- 固定结构，随机概率
        elif action == 'a_mitig':
            reward = self.R_mitig
            # 获取当前状态可以转移到的状态列表
            possible_states = self.MITIG_STRUCTURE[curr_name]
            # 每次随机生成新的转移概率
            random_probs = self._get_random_probs(len(possible_states))
            next_name = np.random.choice(possible_states, p=random_probs)
            print(
                f"[a_mitig] {curr_name} -> {next_name} (probs: {dict(zip(possible_states, [f'{p:.3f}' for p in random_probs]))})")

        # 4. 最终决策（a_accept / a_reject）- 基础概率 + 随机噪声
        elif action in ['a_accept', 'a_reject']:
            done = True
            sub_type = np.random.choice(self.sub_types)

            if action == 'a_accept':
                # 使用基础概率 + 随机扰动
                base_p = self.DECISION_BASE_PROBS[curr_name]['accept']
                p_correct = self._get_noisy_prob(base_p, noise_scale=0.15)
                is_correct = np.random.random() < p_correct
                reward = self.R_CORRECT if is_correct else self.R_MISCLASS
                next_name = f"Accept_{curr_name}_{sub_type}"
                print(
                    f"[a_accept] {curr_name} -> {next_name} (success_prob: {p_correct:.3f}, result: {'✓' if is_correct else '✗'})")
            else:  # a_reject
                base_p = self.DECISION_BASE_PROBS[curr_name]['reject']
                p_correct = self._get_noisy_prob(base_p, noise_scale=0.15)
                is_correct = np.random.random() < p_correct
                reward = self.R_CORRECT if is_correct else (self.R_MISCLASS / 2)
                next_name = f"Reject_{curr_name}_{sub_type}"
                print(
                    f"[a_reject] {curr_name} -> {next_name} (success_prob: {p_correct:.3f}, result: {'✓' if is_correct else '✗'})")

        self.current_model_name = next_name

        # 构建返回状态
        if done or next_name == 's0':
            res_state = self._get_state_dict(next_name)
        else:
            res_state = {
                **self.templates[next_name],
                'g': temp_g,
                'm_metrics': current_state_dict.get('m_metrics', [0.8, 0.8, 0.8, 0.1])
            }

        return res_state, reward, done

    def _update_g_slightly(self, g):
        """评估后轻微更新风险向量"""
        new_g = np.array(g) + np.random.normal(0, 0.03, 4)
        new_g = np.clip(new_g, 1e-5, 1.0)
        return (new_g / new_g.sum()).tolist()


# ----------------- 3. DQN Agent -----------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim, self.action_dim = state_dim, action_dim
        self.memory = deque(maxlen=15000)
        self.gamma, self.epsilon = 0.99, 1.0
        self.epsilon_min, self.epsilon_decay = 0.05, 0.996
        self.batch_size, self.target_update_iter = 64, 100
        self.learn_step_counter = 0

        self.model = QNetwork(state_dim, action_dim)
        self.target_model = QNetwork(state_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            return torch.argmax(self.model(state_t)[0]).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        s, a, r, ns, d = zip(*batch)

        try:
            s = torch.FloatTensor(np.array(s))
            ns = torch.FloatTensor(np.array(ns))
        except ValueError as e:
            print(f"State shape error: {e}")
            return

        a = torch.LongTensor(a).view(-1, 1)
        r = torch.FloatTensor(r).view(-1, 1)
        d = torch.FloatTensor(d).view(-1, 1)

        curr_q = self.model(s).gather(1, a)
        max_next_q = self.target_model(ns).max(1)[0].detach().view(-1, 1)
        target_q = r + (self.gamma * max_next_q * (1 - d))

        loss = nn.MSELoss()(curr_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_iter == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        torch.save({'model': self.model.state_dict(), 'epsilon': self.epsilon}, filename)

    def load(self, filename):
        if os.path.exists(filename):
            ckpt = torch.load(filename)
            self.model.load_state_dict(ckpt['model'])
            self.target_model.load_state_dict(ckpt['model'])
            self.epsilon = ckpt['epsilon']
            print(f"✓ Loaded model from {filename}")


# ----------------- 4. 训练入口 -----------------
def train(verbose=True):
    env = CrossBorderDQNMDP()
    agent = DQNAgent(env.state_dim, env.action_dim)
    save_path = "best_agent_random_probs.pth"
    rewards_history = []
    best_mean = -float('inf')

    print("🚀 Training DQN Agent (Fixed Structure + Random Probabilities)...")
    print(f"State dim: {env.state_dim}, Action dim: {env.action_dim}")
    print("\n转移结构：")
    print("  ✓ S0 --[a_next]--> {M1, M2, M3, M4}")
    print("  ✓ M1 --[a_eval]--> {M1, M2}")
    print("  ✓ M2 --[a_eval]--> {M2, M3, M4}")
    print("  ✓ M3 --[a_eval]--> {M3, M2, M4}")
    print("  ✓ M4 --[a_eval]--> {M4}")
    print("  ✓ M1-M4 --[a_mitig]--> (见代码定义)")
    print("  ✓ M1-M4 --[a_accept/reject]--> 终止")
    print("\n⚡ 每次转移的概率都是随机生成的！\n")

    for e in range(2000):
        state_dict = env.reset()
        state_vec = env.encoder.encode(state_dict)

        assert state_vec.shape[0] == env.state_dim, f"State shape mismatch"

        total_r, done, steps = 0, False, 0

        if verbose and (e + 1) <= 3:
            print(f"\n{'=' * 60}\nEpisode {e + 1} - 展示前3轮的详细转移过程\n{'=' * 60}")

        while not done and steps < 50:
            action = agent.act(state_vec)
            next_dict, reward, done = env.step(state_dict, action)
            next_vec = env.encoder.encode(next_dict)

            agent.memory.append((state_vec, action, reward, next_vec, done))
            state_vec, state_dict = next_vec, next_dict
            total_r += reward
            steps += 1
            agent.replay()

        rewards_history.append(total_r)

        if verbose and (e + 1) <= 3:
            print(f"\nEpisode {e + 1} 结束: 总奖励 = {total_r:.1f}, 步数 = {steps}\n")

        # 保存最优模型
        if (e + 1) >= 100:
            current_mean = np.mean(rewards_history[-100:])
            if current_mean > best_mean:
                best_mean = current_mean
                agent.save(save_path)
                print(f"⭐ New Best! Ep {e + 1}: Mean Reward {best_mean:.2f}, Eps {agent.epsilon:.3f}")

        if (e + 1) % 100 == 0:
            recent_mean = np.mean(rewards_history[-100:])
            print(f"Ep {e + 1}: Last 100 Avg = {recent_mean:.1f}, Eps = {agent.epsilon:.3f}")

    print(f"\n✓ Training complete! Best average reward: {best_mean:.2f}")
    print(f"Model saved to: {save_path}")
    return agent


if __name__ == "__main__":
    trained_agent = train(verbose=True)