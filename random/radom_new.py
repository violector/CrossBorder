import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os


# ----------------- 1. 风险分类器（基于贝叶斯更新）-----------------
class RiskClassifier:
    @staticmethod
    def compute_initial_g(d, b, r, c, m):
        score = np.array([1.0, 0.5, 0.1, 0.0])

        if b == 'categorisation':
            score += [0, 0, 0.5, 5.0]
        if b == 'remote_id':
            score += [0, 0.1, 2.5, 0.5]
        if b == 'verification':
            score += [0, 0.8, 0.2, 0]

        if r == 'high-risk':
            score += [0, 0.15, 3.0, 0.3]

        if c == 'non_compliant':
            score += [0, 0.2, 2.0, 0.5]
        elif c == 'inadequate_SCC':
            score += [0, 0.5, 1.0, 0.1]
        elif c == 'adequacy':
            score += [0, 0.3, 0.2, 0]

        if d in ['multimodal', 'video']:
            score += [0, 0.3, 1.2, 0.1]
        elif d in ['audio', 'image']:
            score += [0, 0.5, 0.3, 0]

        if m == 'insufficient':
            score += [0, 0.1, 0.5, 1.5]
        elif m == 'suboptimal':
            score += [0, 0.5, 0.5, 0]
        elif m == 'optimal':
            score += [0.8, 0.5, 0, 0]

        exp_s = np.exp(score - np.max(score))
        return (exp_s / exp_s.sum()).tolist()

    @staticmethod
    def bayesian_update_g(prior_g, evidence_likelihood, noise_scale=0.05):
        prior = np.array(prior_g)
        likelihood = np.array(evidence_likelihood)
        likelihood += np.random.normal(0, noise_scale, 4)
        likelihood = np.clip(likelihood, 0.01, 10.0)
        posterior = prior * likelihood
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
        # ★ 新增：model_type 映射
        self.model_type_map = {'M1': 0, 'M2': 1, 'M3': 2, 'M4': 3, 's0': -1}

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

        eval_count = state_dict.get('eval_count', 0)
        vec.append(min(eval_count / 8.0, 1.0))

        #新增：将 model_type 编码进状态向量（归一化到 [0,1]）
        model_type = state_dict.get('model_type', 's0')
        mt_idx = max(self.model_type_map.get(model_type, -1), 0)  # s0 映射为 0
        vec.append(mt_idx / 3.0)  # 归一化：M1=0, M2=0.33, M3=0.67, M4=1.0

        return np.array(vec, dtype=np.float32)


# ----------------- 2. MDP 环境 -----------------
class CrossBorderDQNMDP:
    def __init__(self):
        self.encoder = StateEncoder()
        self.actions = ['a_next', 'a_eval', 'a_mitig', 'a_accept', 'a_reject']
        self.action_dim = len(self.actions)
        self.state_dim = 16


        self.R_CORRECT = 50
        self.R_MISCLASS = -80
        self.R_FALSE_REJECT = -40  # ★ 额外惩罚：错误拒绝（本应accept却reject）
        self.R_FALSE_ACCEPT = -60  # ★ 额外惩罚：错误接受（本应reject却accept，更危险）
        self.R_XB, self.R_INFO = -30, -1
        self.R_eval, self.R_mitig = -5, -5
        self.R_EVAL_EXCEEDED = -30

        self.sub_types = ['a', 'b', 'c', 'd']
        self.MAX_EVAL_PER_MODEL = 8
        self.K_PROGRESSION = ['start', 'evidence', 'proposed', 'final']
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

        self.DECISION_BASE_PROBS = {
            'M1': {'accept': 0.9, 'reject': 0.1},
            'M2': {'accept': 0.7, 'reject': 0.3},
            'M3': {'accept': 0.6, 'reject': 0.4},
            'M4': {'accept': 0.1, 'reject': 0.9}
        }

        self.TRUE_TIER = {
            'M1': [0.85, 0.12, 0.03, 0.0],
            'M2': [0.10, 0.60, 0.28, 0.02],
            'M3': [0.05, 0.15, 0.75, 0.05],
            'M4': [0.02, 0.08, 0.25, 0.65]
        }
        self.ACCEPT_CORRECT_TIERS = {0, 1}  # minimal, limited
        self.REJECT_CORRECT_TIERS = {2, 3}  # high, banned

        self.templates = {
            'M1': {'d': 'structured',  'b': 'none',           'r': 'false',     'c': 'EU_only',      'k': 'start', 'm': 'optimal'},
            'M2': {'d': 'multimodal',  'b': 'remote_id',      'r': 'high-risk', 'c': 'adequacy',     'k': 'start', 'm': 'suboptimal'},
            'M3': {'d': 'multimodal',  'b': 'remote_id',      'r': 'high-risk', 'c': 'non_compliant','k': 'start', 'm': 'suboptimal'},
            'M4': {'d': 'multimodal',  'b': 'categorisation', 'r': 'high-risk', 'c': 'adequacy',     'k': 'start', 'm': 'insufficient'}
        }

        # 子类正态分布参数：每个模型类型对应的均值（0~3 对应 a~d）
        # M1 风险最低 → 均值偏向 a(0)；M4 风险最高 → 均值偏向 d(3)
        self.SUBTYPE_NORMAL_PARAMS = {
            'M1': {'mean': 0.5, 'std': 0.6},
            'M2': {'mean': 1.5, 'std': 0.7},
            'M3': {'mean': 2.3, 'std': 0.6},
            'M4': {'mean': 3.0, 'std': 0.5},
        }
        #  扰动强度（越大越随机）
        self.DIRICHLET_ALPHA = 2.0

        self.episode_true_tiers = {}

    def _get_random_probs(self, n):
        p = np.random.rand(n)
        return (p / p.sum()).tolist()

    def _get_noisy_prob(self, base_prob, noise_scale=0.1):
        noisy = base_prob + np.random.normal(0, noise_scale)
        return np.clip(noisy, 0.05, 0.95)

    # 核心修改：正态分布 + Dirichlet 扰动的子类选择
    def _select_subtype_by_risk(self, model_type, decision):
        """
        子类选择：正态分布概率权重 + Dirichlet 扰动

        步骤：
        1. 根据模型类型获取正态分布参数（均值、标准差）
        2. 根据 accept/reject 决策微调均值
        3. 计算 a/b/c/d 四个离散点在正态分布下的概率密度（作为权重）
        4. 叠加 Dirichlet 扰动引入随机性
        5. 按最终权重采样子类
        """
        params = self.SUBTYPE_NORMAL_PARAMS[model_type]
        mu = params['mean']
        sigma = params['std']

        # 根据决策微调均值
        if decision == 'accept':
            mu -= 0.3   # accept → 偏向低风险子类
        else:
            mu += 0.3   # reject → 偏向高风险子类

        mu = np.clip(mu, 0.0, 3.0)

        #  计算四个子类（0,1,2,3）的正态概率密度权重
        positions = np.array([0.0, 1.0, 2.0, 3.0])  # a=0, b=1, c=2, d=3
        normal_weights = np.exp(-0.5 * ((positions - mu) / sigma) ** 2)
        normal_weights /= normal_weights.sum()  # 归一化

        # Dirichlet 扰动：alpha 越大，扰动越小（越接近正态权重）
        # alpha 向量 = normal_weights * DIRICHLET_ALPHA（保留正态形状）
        alpha = normal_weights * self.DIRICHLET_ALPHA
        alpha = np.clip(alpha, 1e-3, None)  # 避免 alpha=0
        perturbed_weights = np.random.dirichlet(alpha)

        # 按扰动后的权重采样
        subtype_idx = np.random.choice(4, p=perturbed_weights)
        return self.sub_types[subtype_idx]

    def reset(self):
        self.current_model_name = 's0'
        self.episode_true_tiers = {}
        return self._get_state_dict('s0')


    def _get_state_dict(self, name):
        if name == 's0' or 'Accept' in name or 'Reject' in name:
            return {
                'd': 'structured', 'b': 'none', 'r': 'false', 'c': 'EU_only',
                'k': 'final', 'm': 'optimal',
                'g': [0.25, 0.25, 0.25, 0.25],
                'm_metrics': [0.0, 0.0, 0.0, 0.0],
                'eval_count': 0,
                'model_type': name  # ★ 直接记录
            }

        state = self.templates[name].copy()
        state['g'] = RiskClassifier.compute_initial_g(
            state['d'], state['b'], state['r'], state['c'], state['m']
        )
        state['m_metrics'] = [0.8, 0.8, 0.8, 0.1]
        state['eval_count'] = 0
        state['model_type'] = name  # ★ 直接记录模型类型
        return state

    def get_model_type(self, state_dict):
        if 'model_type' in state_dict:
            mt = state_dict['model_type']
            # 终止状态或初始状态
            if mt in ['s0'] or 'Accept' in str(mt) or 'Reject' in str(mt):
                return 's0'
            return mt

        # Fallback：仅作为兼容保留，正常路径不会走到这里
        if state_dict.get('b') == 'categorisation':
            return 'M4'
        if state_dict.get('c') in ['non_compliant', 'inadequate_SCC']:
            return 'M3'
        if state_dict.get('d') in ['multimodal', 'video'] or state_dict.get('r') == 'high-risk':
            return 'M2'
        return 'M1'

    def get_valid_actions(self, state_dict):
        curr_name = self.get_model_type(state_dict)
        if curr_name == 's0':
            return [0]   # 只能 a_next
        else:
            return [1, 2, 3, 4]  # a_eval, a_mitig, a_accept, a_reject

    def _generate_evidence_likelihood(self, model_type, eval_count):
        true_dist = np.array(self.TRUE_TIER[model_type])
        noise_scale = max(0.1, 0.5 - eval_count * 0.05)
        likelihood = true_dist + np.random.normal(0, noise_scale, 4)
        return np.clip(likelihood, 0.01, 2.0).tolist()

    def _update_k_value(self, current_k, eval_count):
        k_idx = self.K_PROGRESSION.index(current_k)
        if eval_count >= 6:
            new_k_idx = 3
        elif eval_count >= 3:
            new_k_idx = 2
        elif eval_count >= 1:
            new_k_idx = 1
        else:
            new_k_idx = 0
        return self.K_PROGRESSION[max(k_idx, new_k_idx)]

        # ★ 新增：采样隐藏的 Ground Truth 类别
        # =========================================================

    def _sample_ground_truth_tier(self, model_type):
        """
        从模型的真实分布中采样一个隐藏的 Ground Truth 风险等级。

        TRUE_TIER 定义了每个模型类型的真实概率分布，
        Agent 永远无法直接观测到采样结果，只能通过 a_eval 间接推断。

        返回：0=minimal, 1=limited, 2=high, 3=banned
        """
        true_dist = self.TRUE_TIER[model_type]
        return np.random.choice(4, p=true_dist)

    def _get_or_sample_true_tier(self, model_type):
        if model_type not in self.episode_true_tiers:
            self.episode_true_tiers[model_type] = self._sample_ground_truth_tier(model_type)
        return self.episode_true_tiers[model_type]



        # =========================================================
        # ★ 新增：计算 g 与 true_tier 的一致性奖励（替代原 g_bonus）
        # =========================================================

    def _compute_g_calibration_bonus(self, g_array, true_tier, decision):
        """
        计算 g 的校准质量奖励（奖励评估准确，而非奖励极端分布）

        逻辑：
        - g 对 true_tier 的概率越高 → 说明贝叶斯推断越准确 → bonus 越大
        - g 的熵越高（越不确定）→ 强行决策的惩罚越大
        - 与决策方向无关，只看 g 是否准确反映了真实风险

        参数：
            g_array:   当前 g 值（4维概率向量）
            true_tier: 隐藏的真实风险等级（0~3）
            decision:  'accept' 或 'reject'

        返回：float，范围约 [-0.15, +0.15]
        """
        # 1. 校准奖励：g 对真实等级的置信度
        calibration = g_array[true_tier]  # g 给真实等级分配的概率，越高越准

        # 2. 熵惩罚：g 越均匀（不确定）→ 决策越草率 → 惩罚
        entropy = -np.sum(g_array * np.log(g_array + 1e-8))
        max_entropy = np.log(4)  # 均匀分布的最大熵
        entropy_penalty = (entropy / max_entropy) * 0.1  # 最多 -0.1

        # 3. 合并：校准奖励 - 熵惩罚
        bonus = (calibration - 0.5) * 0.2 - entropy_penalty

        return float(np.clip(bonus, -0.15, 0.15))

    def step(self, current_state_dict, action_idx):
        action = self.actions[action_idx]
        curr_name = self.get_model_type(current_state_dict)  #现在直接从字典读取
        next_name, reward, done = curr_name, self.R_INFO, False

        temp_g = current_state_dict.get('g', [0.25] * 4)
        temp_k = current_state_dict.get('k', 'start')
        temp_eval_count = current_state_dict.get('eval_count', 0)

        valid_actions = self.get_valid_actions(current_state_dict)
        if action_idx not in valid_actions:
            print(f"❌ [Invalid Action!] {action} 不能在 {curr_name} 状态下执行")
            return current_state_dict, -100, False

        if curr_name == 's0':
            # random_probs = self._get_random_probs(len(self.INIT_STRUCTURE))
            # next_name = np.random.choice(self.INIT_STRUCTURE, p=random_probs)
            # temp_eval_count = 0
            # temp_k = 'start'
            # print(f"🚀 [Start] S0 -> {next_name}")
            init_weights = np.array([0.35, 0.35, 0.15, 0.15])  # M1, M2 更频繁
            next_name = np.random.choice(self.INIT_STRUCTURE, p=init_weights)
            temp_eval_count = 0
            temp_k = 'start'
            print(f"🚀 [Start] S0 -> {next_name}")

        elif action == 'a_eval':
            if temp_eval_count >= self.MAX_EVAL_PER_MODEL:
                print(f"⚠️  [Eval limit!] {curr_name} 已评估{temp_eval_count}次，必须决策！")
                reward = self.R_EVAL_EXCEEDED
                next_name = curr_name
            else:
                reward = self.R_eval
                temp_eval_count += 1

                possible_states = self.EVAL_STRUCTURE[curr_name]
                random_probs = self._get_random_probs(len(possible_states))
                next_name = np.random.choice(possible_states, p=random_probs)

                if next_name != curr_name:
                    print(f"🔄 [Transition] {curr_name} -> {next_name}, 重置评估")
                    temp_eval_count = 1
                    temp_k = 'start'
                    temp_g = RiskClassifier.compute_initial_g(
                        self.templates[next_name]['d'],
                        self.templates[next_name]['b'],
                        self.templates[next_name]['r'],
                        self.templates[next_name]['c'],
                        self.templates[next_name]['m']
                    )

                evidence = self._generate_evidence_likelihood(next_name, temp_eval_count)
                temp_g = RiskClassifier.bayesian_update_g(
                    prior_g=temp_g,
                    evidence_likelihood=evidence,
                    noise_scale=max(0.03, 0.1 - temp_eval_count * 0.01)
                )
                old_k = temp_k
                temp_k = self._update_k_value(temp_k, temp_eval_count)

                eval_status = "🟡" if temp_eval_count < 4 else "🟠" if temp_eval_count < 7 else "🔴"
                k_changed = " (k↑)" if temp_k != old_k else ""
                tier_names = ['minimal', 'limited', 'high', 'banned']
                dominant_tier = tier_names[np.argmax(temp_g)]
                confidence = max(temp_g)

                print(f"{eval_status} [Eval {temp_eval_count}/8] {next_name} | k={temp_k}{k_changed} | "
                      f"g={[f'{x:.2f}' for x in temp_g]} → {dominant_tier}({confidence:.2f})")

                if temp_eval_count == self.MAX_EVAL_PER_MODEL:
                    print(f"🚨 [Max reached!] 下一步必须决策")
                    reward += -5

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
                # =========================================================
                # ★ 核心修改：决策判定引入 Ground Truth
                # =========================================================
        elif action in ['a_accept', 'a_reject']:
            done = True
            decision_type = 'accept' if action == 'a_accept' else 'reject'

            # ★ Step 1：采样隐藏的 Ground Truth 风险等级
            true_tier = self._get_or_sample_true_tier(curr_name)
            tier_names = ['minimal', 'limited', 'high', 'banned']

            # ★ Step 2：判定决策是否正确（基于 Ground Truth，与 g 无关）
            if decision_type == 'accept':
                direction_correct = (true_tier in self.ACCEPT_CORRECT_TIERS)
            else:
                direction_correct = (true_tier in self.REJECT_CORRECT_TIERS)

            # ★ Step 3：g 的作用 → 校准奖励（奖励推断准确，惩罚不确定性）
            g_array = np.array(temp_g)
            g_calib_bonus = self._compute_g_calibration_bonus(g_array, true_tier, decision_type)

            # ★ Step 4：k 值奖励（流程越完整，基础成功率越高）
            k_bonus = {'start': 0, 'evidence': 0.05, 'proposed': 0.1, 'final': 0.15}[temp_k]

            # ★ Step 5：评估次数调整
            eval_bonus = 0.1 if 3 <= temp_eval_count <= 6 else (-0.1 if temp_eval_count < 3 else -0.05)

            # ★ Step 6：基础成功率（来自模型类型的先验，反映真实难度）
            base_p = self.DECISION_BASE_PROBS[curr_name][decision_type]

            # ★ Step 7：最终 p_correct（只影响"运气"成分，方向正确性由 true_tier 决定）
            adjusted_p = base_p + k_bonus + eval_bonus + g_calib_bonus
            p_correct = self._get_noisy_prob(adjusted_p, noise_scale=0.1)

            # ★ Step 8：双重判定
            #   - 方向正确（true_tier 匹配）：有 p_correct 概率成功（模拟执行风险）
            #   - 方向错误（true_tier 不匹配）：有小概率侥幸成功（模拟监管宽松）
            if direction_correct:
                is_correct = np.random.random() < p_correct
                reward = self.R_CORRECT if is_correct else self.R_MISCLASS
            else:
                is_correct = np.random.random() < 0.05
                if is_correct:
                    reward = self.R_CORRECT
                else:
                    # ★ 区分错误类型，给予不同惩罚
                    if decision_type == 'reject' and true_tier in self.ACCEPT_CORRECT_TIERS:
                        reward = self.R_FALSE_REJECT  # 本应accept却reject：-40
                    else:
                        reward = self.R_FALSE_ACCEPT  # 本应reject却accept：-60（更严重）

            # reward = self.R_CORRECT if is_correct else self.R_MISCLASS

            sub_type = self._select_subtype_by_risk(curr_name, decision_type)
            next_name = f"{decision_type.capitalize()}_{curr_name}_{sub_type}"

            dominant = tier_names[np.argmax(temp_g)]
            true_label = tier_names[true_tier]
            g_match = "✓" if np.argmax(temp_g) == true_tier else "✗"

            print(f"{'✅' if is_correct else '❌'} [Decision] {decision_type.upper()} | "
                    f"{curr_name} → {next_name} | "
                    f"true={true_label}, g→{dominant}[{g_match}] | "
                    f"dir={'✓' if direction_correct else '✗'}, "
                    f"k={temp_k}, evals={temp_eval_count}, p={p_correct:.3f}")

        self.current_model_name = next_name

        if done or next_name == 's0':
            res_state = self._get_state_dict(next_name)
        else:
            res_state = {
                **self.templates[next_name],
                'g': temp_g,
                'k': temp_k,
                'm_metrics': current_state_dict.get('m_metrics', [0.8, 0.8, 0.8, 0.1]),
                'eval_count': temp_eval_count,
                'model_type': next_name
            }

        return res_state, reward, done


# ----------------- 3. DQN Agent（不变）-----------------
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
        self.priorities = deque(maxlen=15000)  # ★ 优先级队列
        self.priority_alpha = 0.6  # ★ 优先级指数
        self.priority_beta = 0.4  # ★ 重要性采样权重
        self.gamma, self.epsilon = 0.99, 1.0
        self.epsilon_min, self.epsilon_decay = 0.05, 0.996
        self.batch_size, self.target_update_iter = 64, 100
        self.learn_step_counter = 0

        self.model = QNetwork(state_dim, action_dim)
        self.target_model = QNetwork(state_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.action_names = ['a_next', 'a_eval', 'a_mitig', 'a_accept', 'a_reject']

    def act(self, state, valid_actions=None):
        if valid_actions is None:
            valid_actions = list(range(self.action_dim))
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state_t)[0]
            masked_q = q_values.clone()
            invalid_actions = [i for i in range(self.action_dim) if i not in valid_actions]
            masked_q[invalid_actions] = -float('inf')
            return torch.argmax(masked_q).item()

    def remember(self, s, a, r, ns, d):
        """★ 新增：存储时赋予初始优先级（用奖励绝对值作为优先级代理）"""
        self.memory.append((s, a, r, ns, d))
        # 奖励越极端（无论正负）→ 优先级越高
        priority = abs(r) + 1.0  # +1 防止优先级为0
        self.priorities.append(priority)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        # ★ 按优先级采样
        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.priority_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), self.batch_size,
                                   replace=False, p=probs)
        batch = [list(self.memory)[i] for i in indices]

        # 重要性采样权重（修正偏差）
        weights = (len(self.memory) * probs[indices]) ** (-self.priority_beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights).view(-1, 1)

        s, a, r, ns, d = zip(*batch)
        s  = torch.FloatTensor(np.array(s))
        ns = torch.FloatTensor(np.array(ns))
        a  = torch.LongTensor(a).view(-1, 1)
        r  = torch.FloatTensor(r).view(-1, 1)
        d  = torch.FloatTensor(d).view(-1, 1)

        curr_q      = self.model(s).gather(1, a)
        max_next_q  = self.target_model(ns).max(1)[0].detach().view(-1, 1)
        target_q    = r + (self.gamma * max_next_q * (1 - d))

        # ★ 加权损失
        loss = (weights * nn.MSELoss(reduction='none')(curr_q, target_q)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # ★ 更新优先级（用 TD-error 更新）
        td_errors = (curr_q - target_q).detach().abs().squeeze().tolist()
        if isinstance(td_errors, float):
            td_errors = [td_errors]
        mem_list = list(self.memory)
        for i, idx in enumerate(indices):
            if idx < len(self.priorities):
                self.priorities[idx] = td_errors[i] + 1.0

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_iter == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        # batch = random.sample(self.memory, self.batch_size)
        # s, a, r, ns, d = zip(*batch)
        # try:
        #     s = torch.FloatTensor(np.array(s))
        #     ns = torch.FloatTensor(np.array(ns))
        # except ValueError as e:
        #     print(f"State shape error: {e}")
        #     return
        # a = torch.LongTensor(a).view(-1, 1)
        # r = torch.FloatTensor(r).view(-1, 1)
        # d = torch.FloatTensor(d).view(-1, 1)
        # curr_q = self.model(s).gather(1, a)
        # max_next_q = self.target_model(ns).max(1)[0].detach().view(-1, 1)
        # target_q = r + (self.gamma * max_next_q * (1 - d))
        # loss = nn.MSELoss()(curr_q, target_q)
        # self.optimizer.zero_grad()
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        # self.optimizer.step()
        # self.learn_step_counter += 1
        # if self.learn_step_counter % self.target_update_iter == 0:
        #     self.target_model.load_state_dict(self.model.state_dict())
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay

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
    save_path = "agent_v4.pth"
    rewards_history = []
    best_mean = -float('inf')

    for e in range(2000):
        state_dict = env.reset()
        state_vec = env.encoder.encode(state_dict)

        assert state_vec.shape[0] == env.state_dim, \
            f"State dim mismatch: got {state_vec.shape[0]}, expected {env.state_dim}"

        total_r, done, steps = 0, False, 0

        if (e + 1) <= verbose_episodes:
            print(f"\n{'=' * 80}\n📊 Episode {e + 1} - 贝叶斯评估过程\n{'=' * 80}")

        while not done and steps < 50:
            valid_actions = env.get_valid_actions(state_dict)
            action = agent.act(state_vec, valid_actions=valid_actions)
            next_dict, reward, done = env.step(state_dict, action)
            next_vec = env.encoder.encode(next_dict)
            # agent.memory.append((state_vec, action, reward, next_vec, done))
            agent.remember(state_vec, action, reward, next_vec, done)
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
    return agent


if __name__ == "__main__":
    trained_agent = train(verbose_episodes=3)