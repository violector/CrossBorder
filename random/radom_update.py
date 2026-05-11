import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os


# ----------------- 1. 风险分类器（基于贝叶斯更新）-----------------
class RiskClassifier:
    """
    风险分类器：基于状态特征(d,b,r,c,m)计算g值
    g = (g_min, g_lim, g_high, g_ban) 代表属于各风险等级的概率
    """

    @staticmethod
    def compute_initial_g(d, b, r, c, m):
        """
        基于状态特征计算初始g值（先验概率）
        这是一个监督分类器，根据特征推断风险等级
        """
        score = np.array([1.0, 0.5, 0.1, 0.0])  # 基础分：倾向于minimal risk

        # Biometric category 影响
        if b == 'categorisation':
            score += [0, 0, 0.5, 5.0]  # 极高风险（禁止类）
        if b == 'remote_id':
            score += [0, 0.1, 2.5, 0.5]  # 高风险
        if b == 'verification':
            score += [0, 0.8, 0.2, 0]  # 有限风险

        # Risk flags 影响
        if r == 'high-risk':
            score += [0, 0.15, 3.0, 0.3]  # 显著增加high-risk概率

        # Cross-border 影响
        if c == 'non_compliant':
            score += [0, 0.2, 2.0, 0.5]  # 不合规跨境传输→高风险
        elif c == 'inadequate_SCC':
            score += [0, 0.5, 1.0, 0.1]  # SCC不充分→有限/高风险
        elif c == 'adequacy':
            score += [0, 0.3, 0.2, 0]  # 充分性决定→有限风险

        # Dataset type 影响
        if d in ['multimodal', 'video']:
            score += [0, 0.3, 1.2, 0.1]  # 复杂数据类型→风险提升
        elif d in ['audio', 'image']:
            score += [0, 0.5, 0.3, 0]

        # Model performance 影响
        if m == 'insufficient':
            score += [0, 0.1, 0.5, 1.5]  # 性能不足→可能禁止
        elif m == 'suboptimal':
            score += [0, 0.5, 0.5, 0]
        elif m == 'optimal':
            score += [0.8, 0.5, 0, 0]  # 性能优秀→降低风险

        # Softmax归一化
        exp_s = np.exp(score - np.max(score))
        return (exp_s / exp_s.sum()).tolist()

    @staticmethod
    def bayesian_update_g(prior_g, evidence_likelihood, noise_scale=0.05):
        """
        贝叶斯更新：根据新证据调整g值

        prior_g: 先验概率分布 [g_min, g_lim, g_high, g_ban]
        evidence_likelihood: 新证据在各等级下的似然度
        noise_scale: 观测噪声（模拟评估的不确定性）

        后验 ∝ 先验 × 似然
        """
        prior = np.array(prior_g)
        likelihood = np.array(evidence_likelihood)

        # 添加观测噪声
        likelihood += np.random.normal(0, noise_scale, 4)
        likelihood = np.clip(likelihood, 0.01, 10.0)

        # 贝叶斯更新
        posterior = prior * likelihood

        # 归一化
        posterior = np.clip(posterior, 1e-6, 1.0)
        return (posterior / posterior.sum()).tolist()


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
        g_vector = state_dict.get('g', [0.25, 0.25, 0.25, 0.25])

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

        m_metrics = state_dict.get('m_metrics', [0.8, 0.8, 0.8, 0.1])
        if len(m_metrics) != 4:
            m_metrics = m_metrics[:4] + [0.0] * (4 - len(m_metrics))
        vec.extend(m_metrics)

        # 评估次数作为状态特征
        eval_count = state_dict.get('eval_count', 0)
        vec.append(min(eval_count / 8.0, 1.0))

        return np.array(vec, dtype=np.float32)


# ----------------- 2. MDP 环境（所有eval都更新g和k）-----------------
class CrossBorderDQNMDP:
    def __init__(self):
        self.encoder = StateEncoder()
        self.actions = ['a_next', 'a_eval', 'a_mitig', 'a_accept', 'a_reject']
        self.action_dim = len(self.actions)
        self.state_dim = 15

        # 奖励设置
        self.R_CORRECT, self.R_MISCLASS = 50, -80
        self.R_XB, self.R_INFO = -30, -1
        self.R_eval, self.R_mitig = -5, -5
        self.R_EVAL_EXCEEDED = -30

        self.sub_types = ['a', 'b', 'c', 'd']
        self.MAX_EVAL_PER_MODEL = 8

        # 子类风险等级（a最低，d最高）
        # 用于生成符合模型风险等级的子类分布
        self.SUBTYPE_RISK_LEVELS = {'a': 0, 'b': 1, 'c': 2, 'd': 3}

        # k值进展路径
        self.K_PROGRESSION = ['start', 'evidence', 'proposed', 'final']

        # 转移结构
        self.INIT_STRUCTURE = ['M1', 'M2', 'M3', 'M4']

        self.EVAL_STRUCTURE = {
            'M1': ['M1', 'M2'],
            'M2': ['M2', 'M3', 'M4'],
            'M3': ['M3', 'M2', 'M4'],
            'M4': ['M4', 'M3', 'M2']
        }

        self.MITIG_STRUCTURE = {
            'M1': ['M1'],
            'M2': ['M1', 'M3', 'M2'],
            'M3': ['M2', 'M3'],
            'M4': ['M4', 'M2']
        }

        # 决策成功率（基础值，会被g和k调整）
        self.DECISION_BASE_PROBS = {
            'M1': {'accept': 0.9, 'reject': 0.1},
            'M2': {'accept': 0.7, 'reject': 0.3},
            'M3': {'accept': 0.6, 'reject': 0.4},
            'M4': {'accept': 0.1, 'reject': 0.9}
        }

        # 模型真实风险等级（ground truth，用于生成观测证据）
        self.TRUE_TIER = {
            'M1': [0.85, 0.12, 0.03, 0.0],  # minimal risk
            'M2': [0.10, 0.60, 0.28, 0.02],  # limited risk
            'M3': [0.05, 0.15, 0.75, 0.05],  # high risk
            'M4': [0.02, 0.08, 0.25, 0.65]  # banned
        }

        self.templates = {
            'M1': {'d': 'structured', 'b': 'none', 'r': 'false', 'c': 'EU_only', 'k': 'start', 'm': 'optimal'},
            'M2': {'d': 'multimodal', 'b': 'remote_id', 'r': 'high-risk', 'c': 'adequacy', 'k': 'start',
                   'm': 'suboptimal'},
            'M3': {'d': 'multimodal', 'b': 'remote_id', 'r': 'high-risk', 'c': 'non_compliant', 'k': 'start',
                   'm': 'suboptimal'},
            'M4': {'d': 'multimodal', 'b': 'categorisation', 'r': 'high-risk', 'c': 'adequacy', 'k': 'start',
                   'm': 'insufficient'}
        }

    def _get_random_probs(self, n):
        p = np.random.rand(n)
        return (p / p.sum()).tolist()

    def _get_noisy_prob(self, base_prob, noise_scale=0.1):
        noisy = base_prob + np.random.normal(0, noise_scale)
        return np.clip(noisy, 0.05, 0.95)

    def reset(self):
        self.current_model_name = 's0'
        return self._get_state_dict('s0')

    def _get_state_dict(self, name):
        if name == 's0' or 'Accept' in name or 'Reject' in name:
            return {
                'd': 'structured', 'b': 'none', 'r': 'false', 'c': 'EU_only',
                'k': 'final', 'm': 'optimal',
                'g': [0.25, 0.25, 0.25, 0.25],
                'm_metrics': [0.0, 0.0, 0.0, 0.0],
                'eval_count': 0
            }

        state = self.templates[name].copy()
        # 计算初始g值（基于特征的先验）
        state['g'] = RiskClassifier.compute_initial_g(
            state['d'], state['b'], state['r'], state['c'], state['m']
        )
        if 'm_metrics' not in state:
            state['m_metrics'] = [0.8, 0.8, 0.8, 0.1]
        if 'eval_count' not in state:
            state['eval_count'] = 0
        return state

    def get_model_type(self, s):
        if s.get('b') == 'categorisation': return 'M4'
        if s.get('c') in ['non_compliant', 'inadequate_SCC']: return 'M3'
        if s.get('d') in ['multimodal', 'video'] or s.get('r') == 'high-risk': return 'M2'
        return 'M1'

    def _generate_evidence_likelihood(self, model_type, eval_count):
        """
        生成新证据的似然度（模拟观测结果）

        基于真实等级生成观测，但带有噪声
        评估次数越多，观测越准确
        """
        true_dist = np.array(self.TRUE_TIER[model_type])

        # 噪声随评估次数减少（越评估越准确）
        noise_scale = max(0.1, 0.5 - eval_count * 0.05)

        # 生成似然度（从真实分布采样，加噪声）
        likelihood = true_dist + np.random.normal(0, noise_scale, 4)
        likelihood = np.clip(likelihood, 0.01, 2.0)

        return likelihood.tolist()

    def _update_k_value(self, current_k, eval_count):
        """
        根据评估次数推进k值（分类流程阶段）

        start -> evidence -> proposed -> final
        """
        k_idx = self.K_PROGRESSION.index(current_k)

        if eval_count >= 6:
            new_k_idx = 3  # final
        elif eval_count >= 3:
            new_k_idx = 2  # proposed
        elif eval_count >= 1:
            new_k_idx = 1  # evidence
        else:
            new_k_idx = 0  # start

        new_k_idx = max(k_idx, new_k_idx)
        return self.K_PROGRESSION[new_k_idx]

    def _select_subtype_by_risk(self, model_type, decision):
        """
        根据模型类型和决策，使用正态分布选择子类

        a: 风险最小
        b: 风险较小
        c: 风险较高
        d: 风险最高

        策略：
        - M1 + accept → 倾向选择 a/b（低风险子类）
        - M4 + reject → 倾向选择 c/d（高风险子类）
        - M2/M3 → 倾向 b/c（中等风险）
        """
        # 定义每个模型类型的风险中心（0-3，对应a-d）
        model_risk_center = {
            'M1': 0.5,  # 倾向a (minimal risk)
            'M2': 1.5,  # 倾向b (limited risk)
            'M3': 2.3,  # 倾向c (high risk)
            'M4': 3.0  # 倾向d (banned)
        }

        # 根据决策调整中心点
        center = model_risk_center[model_type]

        if decision == 'accept':
            # accept决策：应该是低风险模型，向低风险子类偏移
            center -= 0.3
        else:  # reject
            # reject决策：应该是高风险模型，向高风险子类偏移
            center += 0.3

        # 使用正态分布采样（标准差0.8，允许一定随机性）
        noise = np.random.normal(0, 0.8)
        sampled_value = center + noise

        # 映射到子类索引 [0, 1, 2, 3] → ['a', 'b', 'c', 'd']
        sampled_value = np.clip(sampled_value, 0, 3)
        subtype_idx = int(round(sampled_value))

        return self.sub_types[subtype_idx]

    def get_valid_actions(self, state_dict):
        curr_name = self.get_model_type(state_dict) if self.current_model_name != 's0' else 's0'

        if curr_name == 's0':
            # s0状态：只能执行a_next
            return [0]  # a_next的索引
        else:
            # 其他状态：可以执行a_eval, a_mitig, a_accept, a_reject
            # 但不能执行a_next
            return [1, 2, 3, 4]  # a_eval, a_mitig, a_accept, a_reject的索引

    def step(self, current_state_dict, action_idx):
        action = self.actions[action_idx]
        curr_name = self.get_model_type(current_state_dict) if self.current_model_name != 's0' else 's0'
        next_name, reward, done = curr_name, self.R_INFO, False

        # 获取当前状态
        temp_g = current_state_dict.get('g', [0.25] * 4)
        temp_k = current_state_dict.get('k', 'start')
        temp_eval_count = current_state_dict.get('eval_count', 0)

        # ========== 检查动作有效性 ==========
        valid_actions = self.get_valid_actions(current_state_dict)
        if action_idx not in valid_actions:
            # 无效动作：给予严厉惩罚并保持状态不变
            print(f"❌ [Invalid Action!] {action} 不能在 {curr_name} 状态下执行")
            return current_state_dict, -100, False

        # 1. 从初始状态开始 (只有s0能到这里，因为上面已经检查了)
        if curr_name == 's0':
            # s0状态只能执行a_next（已在valid_actions中确保）
            random_probs = self._get_random_probs(len(self.INIT_STRUCTURE))
            next_name = np.random.choice(self.INIT_STRUCTURE, p=random_probs)
            temp_eval_count = 0
            temp_k = 'start'
            print(f"🚀 [Start] S0 -> {next_name}")

        # ========== 2. 评估动作 - 所有状态都更新g和k ==========
        elif action == 'a_eval':
            # 检查评估次数限制
            if temp_eval_count >= self.MAX_EVAL_PER_MODEL:
                print(f"⚠️  [Eval limit!] {curr_name} 已评估{temp_eval_count}次，必须决策！")
                reward = self.R_EVAL_EXCEEDED
                next_name = curr_name
            else:
                reward = self.R_eval
                temp_eval_count += 1

                # 状态转移
                possible_states = self.EVAL_STRUCTURE[curr_name]
                random_probs = self._get_random_probs(len(possible_states))
                next_name = np.random.choice(possible_states, p=random_probs)

                # 如果转移到新模型，重置计数和状态
                if next_name != curr_name:
                    print(f"🔄 [Transition] {curr_name} -> {next_name}, 重置评估")
                    temp_eval_count = 1
                    temp_k = 'start'
                    # 新模型的初始g值
                    temp_g = RiskClassifier.compute_initial_g(
                        self.templates[next_name]['d'],
                        self.templates[next_name]['b'],
                        self.templates[next_name]['r'],
                        self.templates[next_name]['c'],
                        self.templates[next_name]['m']
                    )

                # ========== 核心：贝叶斯更新g值 ==========
                # 生成新证据的似然度
                evidence = self._generate_evidence_likelihood(next_name, temp_eval_count)

                # 贝叶斯更新
                temp_g = RiskClassifier.bayesian_update_g(
                    prior_g=temp_g,
                    evidence_likelihood=evidence,
                    noise_scale=max(0.03, 0.1 - temp_eval_count * 0.01)
                )

                # 更新k值
                old_k = temp_k
                temp_k = self._update_k_value(temp_k, temp_eval_count)

                # 可视化反馈
                eval_status = "🟡" if temp_eval_count < 4 else "🟠" if temp_eval_count < 7 else "🔴"
                k_changed = " (k↑)" if temp_k != old_k else ""

                # 找出最高概率的等级
                tier_names = ['minimal', 'limited', 'high', 'banned']
                dominant_tier = tier_names[np.argmax(temp_g)]
                confidence = max(temp_g)

                print(f"{eval_status} [Eval {temp_eval_count}/8] {next_name} | k={temp_k}{k_changed} | "
                      f"g={[f'{x:.2f}' for x in temp_g]} → {dominant_tier}({confidence:.2f})")

                if temp_eval_count == self.MAX_EVAL_PER_MODEL:
                    print(f"🚨 [Max reached!] 下一步必须决策")
                    reward += -5

        # 3. 缓解动作
        elif action == 'a_mitig':
            reward = self.R_mitig
            possible_states = self.MITIG_STRUCTURE[curr_name]
            random_probs = self._get_random_probs(len(possible_states))
            next_name = np.random.choice(possible_states, p=random_probs)

            if next_name != curr_name:
                print(f"🔧 [Mitigation] {curr_name} -> {next_name}")
                temp_eval_count = 0
                temp_k = 'start'
                temp_g = RiskClassifier.compute_initial_g(
                    self.templates[next_name]['d'],
                    self.templates[next_name]['b'],
                    self.templates[next_name]['r'],
                    self.templates[next_name]['c'],
                    self.templates[next_name]['m']
                )

        # 4. 最终决策（受g值和k值影响）
        elif action in ['a_accept', 'a_reject']:
            done = True

            # 决策类型
            decision_type = 'accept' if action == 'a_accept' else 'reject'

            # 使用正态分布选择子类（a风险最小，d风险最大）
            sub_type = self._select_subtype_by_risk(curr_name, decision_type)

            # 基础成功率
            base_p = self.DECISION_BASE_PROBS[curr_name][decision_type]

            # k值奖励（流程越完整，决策越准）
            k_bonus = {'start': 0, 'evidence': 0.05, 'proposed': 0.1, 'final': 0.15}[temp_k]

            # 评估次数调整
            if 3 <= temp_eval_count <= 6:
                eval_bonus = 0.1
            elif temp_eval_count < 3:
                eval_bonus = -0.1
            else:
                eval_bonus = -0.05

            # g值对决策的影响：如果g值与决策一致，提升准确率
            g_array = np.array(temp_g)
            if action == 'a_accept':
                # accept适合minimal/limited风险
                g_alignment = g_array[0] + g_array[1]  # minimal + limited
            else:  # reject
                # reject适合high/banned风险
                g_alignment = g_array[2] + g_array[3]  # high + banned

            g_bonus = (g_alignment - 0.5) * 0.2  # 最多±0.1调整

            # 最终准确率
            adjusted_p = base_p + k_bonus + eval_bonus + g_bonus
            p_correct = self._get_noisy_prob(adjusted_p, noise_scale=0.1)

            is_correct = np.random.random() < p_correct
            reward = self.R_CORRECT if is_correct else self.R_MISCLASS

            # 终端状态格式：Accept_M1_a 或 Reject_M3_c
            next_name = f"{decision_type.capitalize()}_{curr_name}_{sub_type}"

            tier_names = ['minimal', 'limited', 'high', 'banned']
            dominant = tier_names[np.argmax(temp_g)]
            print(f"{'✅' if is_correct else '❌'} [Decision] {decision_type.upper()} | {curr_name} → {next_name} | "
                  f"k={temp_k}, evals={temp_eval_count}, g→{dominant} | p={p_correct:.3f}")

        self.current_model_name = next_name

        if done or next_name == 's0':
            res_state = self._get_state_dict(next_name)
        else:
            res_state = {
                **self.templates[next_name],
                'g': temp_g,
                'k': temp_k,
                'm_metrics': current_state_dict.get('m_metrics', [0.8, 0.8, 0.8, 0.1]),
                'eval_count': temp_eval_count
            }

        return res_state, reward, done


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

        # 动作名称映射
        self.action_names = ['a_next', 'a_eval', 'a_mitig', 'a_accept', 'a_reject']

    def act(self, state, valid_actions=None):
        """
        选择动作，支持动作掩码

        state: 状态向量
        valid_actions: 可用动作列表（索引），None表示所有动作可用
        """
        if valid_actions is None:
            valid_actions = list(range(self.action_dim))

        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.model(state_t)[0]

                # 将不可用动作的Q值设为负无穷
                masked_q = q_values.clone()
                invalid_actions = [i for i in range(self.action_dim) if i not in valid_actions]
                masked_q[invalid_actions] = -float('inf')

                return torch.argmax(masked_q).item()

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
def train(verbose_episodes=3):
    env = CrossBorderDQNMDP()
    agent = DQNAgent(env.state_dim, env.action_dim)
    save_path = "agent_v2.pth"
    rewards_history = []
    best_mean = -float('inf')


    for e in range(200):
        state_dict = env.reset()
        state_vec = env.encoder.encode(state_dict)

        assert state_vec.shape[0] == env.state_dim

        total_r, done, steps = 0, False, 0

        if (e + 1) <= verbose_episodes:
            print(f"\n{'=' * 80}\n📊 Episode {e + 1} - 贝叶斯评估过程\n{'=' * 80}")

        while not done and steps < 50:
            # 获取当前状态的有效动作
            valid_actions = env.get_valid_actions(state_dict)

            # Agent根据有效动作选择
            action = agent.act(state_vec, valid_actions=valid_actions)

            next_dict, reward, done = env.step(state_dict, action)
            next_vec = env.encoder.encode(next_dict)

            agent.memory.append((state_vec, action, reward, next_vec, done))
            state_vec, state_dict = next_vec, next_dict
            total_r += reward
            steps += 1
            agent.replay()

        rewards_history.append(total_r)

        if (e + 1) <= verbose_episodes:
            print(f"\n✅ Episode {e + 1} 结束: 总奖励={total_r:.1f}, 步数={steps}\n")

        if (e + 1) >= 100:
            current_mean = np.mean(rewards_history[-100:])
            if current_mean > best_mean:
                best_mean = current_mean
                agent.save(save_path)
                print(f"⭐ New Best! Ep {e + 1}: Mean Reward {best_mean:.2f}, Eps {agent.epsilon:.3f}")

        if (e + 1) % 100 == 0:
            recent_mean = np.mean(rewards_history[-100:])
            print(f"Ep {e + 1}: Last 100 Avg = {recent_mean:.1f}, Eps = {agent.epsilon:.3f}")

    print(f"\n✓ Training complete! Best: {best_mean:.2f}")
    print(f"Model saved: {save_path}")
    return agent


if __name__ == "__main__":
    trained_agent = train(verbose_episodes=3)