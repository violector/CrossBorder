import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os


# ----------------- 1. 风险分类器 -----------------
class RiskClassifier:
    @staticmethod
    def compute_initial_g(d, b, r, c, m):
        score = np.array([1.0, 0.5, 0.1, 0.0])
        if b == 'categorisation': score += [0, 0, 0.5, 5.0]
        if b == 'remote_id':      score += [0, 0.1, 2.5, 0.5]
        if b == 'verification':   score += [0, 0.8, 0.2, 0]
        if r == 'high-risk':      score += [0, 0.15, 3.0, 0.3]
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
        posterior = np.clip(prior * likelihood, 1e-6, 1.0)
        return (posterior / posterior.sum()).tolist()


# ----------------- 2. 状态编码器 -----------------
class StateEncoder:
    def __init__(self):
        self.d_map = {'structured': 0, 'unstructured_text': 1, 'image': 2,
                      'video': 3, 'audio': 4, 'multimodal': 5}
        self.b_map = {'none': 0, 'verification': 1, 'remote_id': 2, 'categorisation': 3}
        self.r_map = {'false': 0, 'high-risk': 1}
        self.c_map = {'EU_only': 0, 'adequacy': 1, 'inadequate_SCC': 2, 'non_compliant': 3}
        self.k_map = {'start': 0, 'evidence': 1, 'proposed': 2, 'final': 3}
        self.m_map = {'optimal': 0, 'suboptimal': 1, 'insufficient': 2}
        self.mt_map = {'M1': 0, 'M2': 1, 'M3': 2, 'M4': 3}

    def encode(self, state_dict):
        d, b, r, c = state_dict['d'], state_dict['b'], state_dict['r'], state_dict['c']
        m, k = state_dict['m'], state_dict['k']

        g_vector = state_dict.get('g', [0.25] * 4)
        if not isinstance(g_vector, list) or len(g_vector) != 4:
            g_vector = [0.25] * 4

        vec = [self.d_map[d], self.b_map[b], self.r_map[r],
               self.c_map[c], self.k_map[k], self.m_map[m]]
        vec.extend(g_vector)  # +4 = 10

        m_metrics = state_dict.get('m_metrics', [0.8, 0.8, 0.8, 0.1])
        if len(m_metrics) != 4:
            m_metrics = (m_metrics + [0.0] * 4)[:4]
        vec.extend(m_metrics)  # +4 = 14

        vec.append(min(state_dict.get('eval_count', 0) / 8.0, 1.0))   # 15
        mt = state_dict.get('model_type', 'M1')
        vec.append(self.mt_map.get(mt, 0) / 3.0)                       # 16
        vec.append(min(state_dict.get('candidates_remaining', 0) / 4.0, 1.0))  # 17
        vec.append(min(state_dict.get('mitig_count', 0) / 3.0, 1.0))   # 18

        return np.array(vec, dtype=np.float32)  # dim = 18


# ----------------- 3. MDP 环境 -----------------
class CrossBorderDQNMDP:
    def __init__(self):
        self.encoder = StateEncoder()
        self.actions = ['a_next', 'a_eval', 'a_mitig', 'a_accept', 'a_reject']
        self.action_dim = len(self.actions)
        self.state_dim = 18

        # 奖励参数
        self.R_CORRECT      =  50
        self.R_MISCLASS     = -80
        self.R_FALSE_REJECT = -40   # 本应 accept 却 reject
        self.R_FALSE_ACCEPT = -60   # 本应 reject 却 accept（更危险）
        self.R_INFO         =  -1
        self.R_eval         =  -3
        self.R_mitig_base   =  -3
        self.R_MITIG_SUCCESS =  15  # 缓解后风险等级下降
        self.R_MITIG_FAIL   =  -5   # 缓解后风险等级上升
        self.R_EVAL_EXCEEDED = -40
        self.R_RECKLESS     = -10   # 高风险模型评估不足直接决策
        # ── 新增 ──────────────────────────────────────────────────────────────
        # 跨境不合规惩罚：c=non_compliant 或 inadequate_SCC，
        # 且整个处理流程中未执行任何 a_mitig，直接做出 accept/reject 决策。
        # 语义：完全忽视 GDPR Chapter V 跨境传输义务而做出决策。
        # 取值 -25：比鲁莽惩罚（-10）更重，但轻于误分类（-60/-40），
        # 因为仅针对"程序违规"而非"实质判断错误"。
        self.R_XB           = -25
        # 跨境不合规的 c 值集合（EU_only 和 adequacy 视为合规，不触发）
        self.NON_COMPLIANT_C = {'non_compliant', 'inadequate_SCC'}
        # ─────────────────────────────────────────────────────────────────────

        self.MAX_MITIG_PER_MODEL = 3
        self.MAX_EVAL_PER_MODEL  = 5
        self.K_PROGRESSION = ['start', 'evidence', 'proposed', 'final']
        self.TIER_ORDER = {'M1': 0, 'M2': 1, 'M3': 2, 'M4': 3}

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
        self.MITIG_FIELD_CHANGES = {
            ('M2', 'M1'): {'d': 'structured', 'b': 'none',
                           'c': 'EU_only',    'm': 'optimal'},
            ('M2', 'M3'): {'c': 'non_compliant'},
            ('M3', 'M2'): {'c': 'adequacy'},
            ('M4', 'M2'): {'b': 'remote_id', 'c': 'adequacy', 'm': 'suboptimal'},
        }

        self.templates = {
            'M1': {'d': 'structured',  'b': 'none',           'r': 'false',
                   'c': 'EU_only',     'm': 'optimal'},
            'M2': {'d': 'multimodal',  'b': 'remote_id',      'r': 'high-risk',
                   'c': 'adequacy',    'm': 'suboptimal'},
            'M3': {'d': 'multimodal',  'b': 'remote_id',      'r': 'high-risk',
                   'c': 'non_compliant','m': 'suboptimal'},
            'M4': {'d': 'multimodal',  'b': 'categorisation', 'r': 'high-risk',
                   'c': 'adequacy',    'm': 'insufficient'},
        }

        self.TRUE_TIER = {
            'M1': [0.85, 0.12, 0.03, 0.00],
            'M2': [0.10, 0.60, 0.28, 0.02],
            'M3': [0.05, 0.15, 0.75, 0.05],
            'M4': [0.02, 0.08, 0.25, 0.65],
        }
        self.ACCEPT_CORRECT_TIERS = {0, 1}
        self.REJECT_CORRECT_TIERS = {2, 3}

        self.DECISION_BASE_PROBS = {
            'M1': {'accept': 0.9, 'reject': 0.1},
            'M2': {'accept': 0.7, 'reject': 0.3},
            'M3': {'accept': 0.6, 'reject': 0.4},
            'M4': {'accept': 0.2, 'reject': 0.8},
        }

        self.sub_types = ['a', 'b', 'c', 'd']
        self.SUBTYPE_NORMAL_PARAMS = {
            'M1': {'mean': 0.5, 'std': 0.6},
            'M2': {'mean': 1.5, 'std': 0.7},
            'M3': {'mean': 2.3, 'std': 0.6},
            'M4': {'mean': 3.0, 'std': 0.5},
        }
        self.DIRICHLET_ALPHA = 2.0

        self.candidate_pool = []
        self.current_candidate_idx = -1
        self.current_model_state = None
        self.episode_true_tiers = {}
        self.phase = 'select'

    # ── 工具方法 ──────────────────────────────────────────
    def _uniform_probs(self, n):
        return np.ones(n, dtype=np.float32) / n

    def _get_noisy_prob(self, p, noise=0.1):
        return float(np.clip(p + np.random.normal(0, noise), 0.05, 0.95))

    def _get_or_sample_true_tier(self, model_type):
        if model_type not in self.episode_true_tiers:
            self.episode_true_tiers[model_type] = int(
                np.random.choice(4, p=self.TRUE_TIER[model_type]))
        return self.episode_true_tiers[model_type]

    def _compute_g_calibration_bonus(self, g_array, true_tier):
        calibration    = float(g_array[true_tier])
        entropy        = float(-np.sum(g_array * np.log(g_array + 1e-8)))
        entropy_penalty = (entropy / np.log(4)) * 0.1
        return float(np.clip((calibration - 0.5) * 0.2 - entropy_penalty, -0.15, 0.15))

    def _update_k_value(self, current_k, eval_count):
        k_idx = self.K_PROGRESSION.index(current_k)
        if eval_count >= 6:   new_k_idx = 3
        elif eval_count >= 3: new_k_idx = 2
        elif eval_count >= 1: new_k_idx = 1
        else:                 new_k_idx = 0
        return self.K_PROGRESSION[max(k_idx, new_k_idx)]

    def _generate_evidence_likelihood(self, model_type, eval_count):
        true_dist   = np.array(self.TRUE_TIER[model_type])
        noise_scale = max(0.1, 0.5 - eval_count * 0.05)
        return np.clip(
            true_dist + np.random.normal(0, noise_scale, 4), 0.01, 2.0
        ).tolist()

    def _select_subtype(self, model_type, decision):
        p  = self.SUBTYPE_NORMAL_PARAMS[model_type]
        mu = float(np.clip(
            p['mean'] + (-0.3 if decision == 'accept' else 0.3), 0.0, 3.0))
        pos = np.array([0., 1., 2., 3.])
        w   = np.exp(-0.5 * ((pos - mu) / p['std']) ** 2)
        w  /= w.sum()
        alpha = np.clip(w * self.DIRICHLET_ALPHA, 1e-3, None)
        pw    = np.random.dirichlet(alpha)
        return self.sub_types[int(np.random.choice(4, p=pw))]

    # ── 新增辅助方法 ──────────────────────────────────────
    def _compute_xb_penalty(self, s):
        """
        计算跨境不合规惩罚 R_XB。

        触发条件（同时满足）：
          1. c 字段为 non_compliant 或 inadequate_SCC
             （表示存在第三国数据传输且无充分性决定或 SCC 不充分）
          2. mitig_count == 0
             （整个候选模型处理流程中从未执行 a_mitig，
               即完全没有尝试替换数据源或补充合规措施）

        不触发情况：
          - c == 'EU_only' 或 'adequacy'：数据来源合规，无需额外处理
          - mitig_count > 0：Agent 已尝试过缓解，即便最终未降级，
            也体现了对跨境义务的重视，不施加程序性惩罚

        返回：float，0（不触发）或 self.R_XB（触发）
        """
        if s.get('c') in self.NON_COMPLIANT_C and s.get('mitig_count', 0) == 0:
            return self.R_XB
        return 0

    # ── 状态构建 ──────────────────────────────────────────
    def _make_model_state(self, model_type, candidates_remaining=0, mitig_count=0):
        base = self.templates[model_type].copy()
        base['k']          = 'start'
        base['g']          = RiskClassifier.compute_initial_g(
            base['d'], base['b'], base['r'], base['c'], base['m'])
        base['m_metrics']  = [0.8, 0.8, 0.8, 0.1]
        base['eval_count'] = 0
        base['mitig_count'] = mitig_count
        base['model_type'] = model_type
        base['candidates_remaining'] = candidates_remaining
        return base

    def _s0_state(self):
        remaining = max(
            len(self.candidate_pool) - self.current_candidate_idx - 1, 0)
        return {
            'd': 'structured', 'b': 'none', 'r': 'false', 'c': 'EU_only',
            'k': 'start', 'm': 'optimal',
            'g': [0.25] * 4, 'm_metrics': [0.0] * 4,
            'eval_count': 0, 'mitig_count': 0,
            'model_type': 'M1', 'candidates_remaining': remaining,
        }

    def _terminal_state(self):
        return {
            'd': 'structured', 'b': 'none', 'r': 'false', 'c': 'EU_only',
            'k': 'final', 'm': 'optimal',
            'g': [0.25] * 4, 'm_metrics': [0.0] * 4,
            'eval_count': 0, 'mitig_count': 0,
            'model_type': 'M1', 'candidates_remaining': 0,
        }

    # ── 动作合法性 ────────────────────────────────────────
    def get_valid_actions(self, state_dict):
        if self.phase == 'select':
            return [0]
        valid  = [3, 4]
        g_conf = max(state_dict.get('g', [0.25] * 4))
        ec     = state_dict.get('eval_count', 0)
        if ec < self.MAX_EVAL_PER_MODEL and g_conf < 0.85:
            valid.append(1)
        if state_dict.get('mitig_count', 0) < self.MAX_MITIG_PER_MODEL:
            valid.append(2)
        return sorted(valid)

    # ── Episode 管理 ──────────────────────────────────────
    def reset(self):
        pool_size = np.random.randint(2, 5)
        weights   = np.array([0.35, 0.35, 0.15, 0.15])
        self.candidate_pool = list(
            np.random.choice(['M1', 'M2', 'M3', 'M4'], size=pool_size, p=weights))
        self.current_candidate_idx = -1
        self.current_model_state   = None
        self.episode_true_tiers    = {}
        self.phase = 'select'
        print(f"🎬 [Episode Start] 候选池: {self.candidate_pool}")
        return self._s0_state()

    # ── 核心步进 ──────────────────────────────────────────
    def step(self, current_state_dict, action_idx):
        action       = self.actions[action_idx]
        reward, done = self.R_INFO, False

        valid = self.get_valid_actions(current_state_dict)
        if action_idx not in valid:
            print(f"❌ [Invalid] {action} 在当前阶段({self.phase})不可用，"
                  f"合法动作: {[self.actions[i] for i in valid]}")
            return current_state_dict, -50, False

        # ══════════════════════════════════════════════════
        # a_next
        # ══════════════════════════════════════════════════
        if action == 'a_next':
            self.current_candidate_idx += 1
            model_type = self.candidate_pool[self.current_candidate_idx]
            remaining  = len(self.candidate_pool) - self.current_candidate_idx - 1
            self.current_model_state = self._make_model_state(
                model_type, candidates_remaining=remaining)
            self.phase = 'evaluate'
            print(f"🚀 [SelectNext] 候选 #{self.current_candidate_idx + 1}"
                  f"/{len(self.candidate_pool)}: {model_type} | 剩余候选: {remaining}")
            return self.current_model_state, self.R_INFO, False

        # ══════════════════════════════════════════════════
        # a_eval
        # ══════════════════════════════════════════════════
        elif action == 'a_eval':
            s = self.current_model_state
            if s['eval_count'] >= self.MAX_EVAL_PER_MODEL:
                print(f"⚠️  [Eval limit] {s['model_type']} 已评估 {s['eval_count']} 次，必须决策！")
                return self.current_model_state, self.R_EVAL_EXCEEDED, False

            s['eval_count'] += 1
            curr_type      = s['model_type']
            possible_types = self.EVAL_STRUCTURE[curr_type]
            next_type      = str(np.random.choice(
                possible_types, p=self._uniform_probs(len(possible_types))))

            if next_type != curr_type:
                print(f"🔄 [Reclassify via Eval] {curr_type} → {next_type}")
                remaining   = s['candidates_remaining']
                mitig_count = s['mitig_count']
                prev_eval   = s['eval_count']
                prev_k      = s['k']
                s = self._make_model_state(
                    next_type, candidates_remaining=remaining, mitig_count=mitig_count)
                s['eval_count'] = prev_eval
                s['k']          = self._update_k_value(prev_k, prev_eval)
                self.current_model_state = s

            evidence = self._generate_evidence_likelihood(s['model_type'], s['eval_count'])
            s['g']   = RiskClassifier.bayesian_update_g(
                prior_g=s['g'], evidence_likelihood=evidence,
                noise_scale=max(0.03, 0.1 - s['eval_count'] * 0.01))

            old_k  = s['k']
            s['k'] = self._update_k_value(s['k'], s['eval_count'])

            g_array  = np.array(s['g'])
            g_conf   = float(np.max(g_array))
            progressive_penalty = -float(s['eval_count'])
            confidence_penalty  = -4.0 * max(0, g_conf - 0.65)
            early_bonus         = 1.5 if s['eval_count'] <= 2 else 0.0
            reward              = progressive_penalty + confidence_penalty + early_bonus

            tier_names = ['minimal', 'limited', 'high', 'banned']
            dominant   = tier_names[int(np.argmax(g_array))]
            kc         = " (k↑)" if s['k'] != old_k else ""
            es = "🟡" if s['eval_count'] < 3 else "🟠" if s['eval_count'] < 5 else "🔴"
            print(f"{es} [Eval {s['eval_count']}/{self.MAX_EVAL_PER_MODEL}] "
                  f"{s['model_type']} | k={s['k']}{kc} | "
                  f"g={[f'{x:.2f}' for x in s['g']]} → {dominant}({g_conf:.2f}) | "
                  f"reward={reward:.2f} "
                  f"[prog={progressive_penalty:.1f}, conf={confidence_penalty:.2f}, early={early_bonus}]")
            if s['eval_count'] == self.MAX_EVAL_PER_MODEL:
                print("🚨 [Max evals] 下一步必须决策！")
                reward -= 3
            return self.current_model_state, reward, False

        # ══════════════════════════════════════════════════
        # a_mitig
        # ══════════════════════════════════════════════════
        elif action == 'a_mitig':
            s = self.current_model_state
            s['mitig_count'] += 1
            curr_type      = s['model_type']
            possible_types = self.MITIG_STRUCTURE[curr_type]
            next_type      = str(np.random.choice(
                possible_types, p=self._uniform_probs(len(possible_types))))
            curr_tier      = self.TIER_ORDER[curr_type]
            next_tier      = self.TIER_ORDER[next_type]
            field_changes  = self.MITIG_FIELD_CHANGES.get((curr_type, next_type), {})

            if next_type != curr_type:
                remaining   = s['candidates_remaining']
                mitig_count = s['mitig_count']
                prev_eval   = s['eval_count']
                prev_k      = s['k']
                prev_g      = s['g'][:]
                new_features = {
                    'd': s['d'], 'b': s['b'], 'r': s['r'],
                    'c': s['c'], 'm': s['m']
                }
                new_features.update(field_changes)
                new_g_prior = RiskClassifier.compute_initial_g(
                    new_features['d'], new_features['b'],
                    new_features['r'], new_features['c'], new_features['m'])
                blended_g = [(prev_g[i] + new_g_prior[i]) / 2.0 for i in range(4)]
                g_sum     = sum(blended_g)
                blended_g = [x / g_sum for x in blended_g]
                s = {
                    **new_features,
                    'k':                    prev_k,
                    'g':                    blended_g,
                    'm_metrics':            [0.8, 0.8, 0.8, 0.1],
                    'eval_count':           prev_eval,
                    'mitig_count':          mitig_count,
                    'model_type':           next_type,
                    'candidates_remaining': remaining,
                }
                self.current_model_state = s
                changes_desc = ', '.join(f"{k}: →{v}" for k, v in field_changes.items())
                print(f"🔧 [Mitig #{s['mitig_count']}] 字段修改: [{changes_desc}]")
            else:
                print(f"🔧 [Mitig #{s['mitig_count']}] 缓解无效果，特征不变")

            is_sideshift = (curr_type == 'M2' and next_type == 'M3')
            if next_tier < curr_tier:
                reward = self.R_mitig_base + self.R_MITIG_SUCCESS
                print(f"🔧✅ [Mitig #{s['mitig_count']}] "
                      f"{curr_type}(tier{curr_tier}) → {next_type}(tier{next_tier}) "
                      f"| 风险下降 | reward={reward}")
            elif next_tier == curr_tier:
                reward = self.R_mitig_base
                print(f"🔧⚪ [Mitig #{s['mitig_count']}] "
                      f"{curr_type} → {next_type}（无效果）| reward={reward}")
            elif is_sideshift:
                reward = self.R_mitig_base
                print(f"🔧🔀 [Mitig #{s['mitig_count']}] "
                      f"M2 → M3（风险侧移）| reward={reward}")
            else:
                reward = self.R_mitig_base + self.R_MITIG_FAIL
                print(f"🔧❌ [Mitig #{s['mitig_count']}] "
                      f"{curr_type}(tier{curr_tier}) → {next_type}(tier{next_tier}) "
                      f"| 风险上升 | reward={reward}")

            if s['mitig_count'] >= self.MAX_MITIG_PER_MODEL:
                print(f"⚠️  [Mitig limit] 已缓解 {s['mitig_count']} 次，后续不可再缓解")
            return self.current_model_state, reward, False

        # ══════════════════════════════════════════════════
        # a_accept
        # ══════════════════════════════════════════════════
        elif action == 'a_accept':
            s          = self.current_model_state
            model_type = s['model_type']
            true_tier  = self._get_or_sample_true_tier(model_type)
            tier_names = ['minimal', 'limited', 'high', 'banned']

            direction_correct = (true_tier in self.ACCEPT_CORRECT_TIERS)
            g_array   = np.array(s['g'])
            g_bonus   = self._compute_g_calibration_bonus(g_array, true_tier)
            k_bonus   = {'start': 0, 'evidence': 0.05,
                         'proposed': 0.1, 'final': 0.15}[s['k']]
            eval_bonus = (0.1 if 3 <= s['eval_count'] <= 6
                          else -0.1 if s['eval_count'] < 3 else -0.05)
            base_p    = self.DECISION_BASE_PROBS[model_type]['accept']
            p_correct = self._get_noisy_prob(base_p + k_bonus + eval_bonus + g_bonus)

            reckless = (self.R_RECKLESS
                        if model_type in ['M3', 'M4'] and s['eval_count'] < 2 else 0)

            if direction_correct:
                is_correct = np.random.random() < p_correct
                reward     = (self.R_CORRECT if is_correct else self.R_MISCLASS) + reckless
            else:
                is_correct = np.random.random() < 0.05
                reward     = (self.R_CORRECT if is_correct else self.R_FALSE_ACCEPT) + reckless

            # ── 跨境不合规惩罚 ────────────────────────────────────────────────
            xb_penalty = self._compute_xb_penalty(s)
            if xb_penalty < 0:
                reward += xb_penalty
                print(f"⚠️  [XB Penalty] c={s['c']}, mitig_count={s['mitig_count']} "
                      f"→ 跨境不合规未缓解直接决策，追加惩罚 {xb_penalty}")
            # ─────────────────────────────────────────────────────────────────

            sub      = self._select_subtype(model_type, 'accept')
            dominant = tier_names[int(np.argmax(g_array))]
            g_match  = "✓" if int(np.argmax(g_array)) == true_tier else "✗"
            print(f"{'✅' if is_correct else '❌'} [Accept] {model_type} → "
                  f"Accept_{model_type}_{sub} | "
                  f"true={tier_names[true_tier]}, g→{dominant}[{g_match}] | "
                  f"dir={'✓' if direction_correct else '✗'}, "
                  f"k={s['k']}, evals={s['eval_count']}, p={p_correct:.3f}"
                  + (f" ⚡reckless={reckless}" if reckless else "")
                  + (f" 🌐xb={xb_penalty}"     if xb_penalty < 0 else ""))
            return self._terminal_state(), reward, True

        # ══════════════════════════════════════════════════
        # a_reject
        # ══════════════════════════════════════════════════
        elif action == 'a_reject':
            s          = self.current_model_state
            model_type = s['model_type']
            true_tier  = self._get_or_sample_true_tier(model_type)
            tier_names = ['minimal', 'limited', 'high', 'banned']

            direction_correct = (true_tier in self.REJECT_CORRECT_TIERS)
            g_array   = np.array(s['g'])
            g_bonus   = self._compute_g_calibration_bonus(g_array, true_tier)
            k_bonus   = {'start': 0, 'evidence': 0.05,
                         'proposed': 0.1, 'final': 0.15}[s['k']]
            eval_bonus = (0.1 if 3 <= s['eval_count'] <= 6
                          else -0.1 if s['eval_count'] < 3 else -0.05)
            base_p    = self.DECISION_BASE_PROBS[model_type]['reject']
            p_correct = self._get_noisy_prob(base_p + k_bonus + eval_bonus + g_bonus)

            reckless = (self.R_RECKLESS
                        if model_type in ['M3', 'M4'] and s['eval_count'] < 2 else 0)

            if direction_correct:
                is_correct = np.random.random() < p_correct
                reward     = (self.R_CORRECT if is_correct else self.R_MISCLASS) + reckless
            else:
                is_correct = np.random.random() < 0.05
                reward     = (self.R_CORRECT if is_correct else self.R_FALSE_REJECT) + reckless

            # ── 跨境不合规惩罚 ────────────────────────────────────────────────
            xb_penalty = self._compute_xb_penalty(s)
            if xb_penalty < 0:
                reward += xb_penalty
                print(f"⚠️  [XB Penalty] c={s['c']}, mitig_count={s['mitig_count']} "
                      f"→ 跨境不合规未缓解直接决策，追加惩罚 {xb_penalty}")
            # ─────────────────────────────────────────────────────────────────

            remaining = len(self.candidate_pool) - self.current_candidate_idx - 1
            dominant  = tier_names[int(np.argmax(g_array))]
            g_match   = "✓" if int(np.argmax(g_array)) == true_tier else "✗"

            if remaining > 0:
                self.phase = 'select'
                self.current_model_state = None
                next_state = self._s0_state()
                done = False
                print(f"{'✅' if is_correct else '❌'} [Reject] {model_type} 丢弃 | "
                      f"true={tier_names[true_tier]}, g→{dominant}[{g_match}] | "
                      f"dir={'✓' if direction_correct else '✗'}, 剩余候选={remaining}"
                      + (f" ⚡reckless={reckless}" if reckless else "")
                      + (f" 🌐xb={xb_penalty}"     if xb_penalty < 0 else ""))
            else:
                next_state = self._terminal_state()
                done = True
                print(f"{'✅' if is_correct else '❌'} [Reject+Exhausted] {model_type} | "
                      f"所有候选已审查完毕"
                      + (f" ⚡reckless={reckless}" if reckless else "")
                      + (f" 🌐xb={xb_penalty}"     if xb_penalty < 0 else ""))
            return next_state, reward, done

        return current_state_dict, self.R_INFO, False


# ----------------- 4. Q 网络 -----------------
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),       nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# ----------------- 5. DQN Agent（优先经验回放）-----------------
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim, self.action_dim = state_dim, action_dim
        self.memory     = deque(maxlen=20000)
        self.priorities = deque(maxlen=20000)

        self.gamma              = 0.99
        self.epsilon            = 1.0
        self.epsilon_min        = 0.05
        self.epsilon_decay      = 0.997
        self.batch_size         = 64
        self.target_update_iter = 100
        self.learn_step_counter = 0
        self.priority_alpha     = 0.6
        self.priority_beta      = 0.4

        self.model        = QNetwork(state_dim, action_dim)
        self.target_model = QNetwork(state_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer    = optim.Adam(self.model.parameters(), lr=0.001)
        self.action_names = ['a_next', 'a_eval', 'a_mitig', 'a_accept', 'a_reject']

    def act(self, state, valid_actions=None):
        if valid_actions is None:
            valid_actions = list(range(self.action_dim))
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        with torch.no_grad():
            q = self.model(torch.FloatTensor(state).unsqueeze(0))[0].clone()
            q[[i for i in range(self.action_dim) if i not in valid_actions]] = -float('inf')
            return int(torch.argmax(q).item())

    def remember(self, s, a, r, ns, d, action_name=None):
        self.memory.append((s, a, r, ns, d))
        priority = abs(r) + 1.0
        self.priorities.append(float(priority))
        if action_name == 'a_mitig' and r > 0:
            for _ in range(2):
                self.memory.append((s, a, r, ns, d))
                self.priorities.append(float(priority * 1.5))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        priorities = np.array(self.priorities, dtype=np.float32)
        probs      = priorities ** self.priority_alpha
        probs     /= probs.sum()
        indices    = np.random.choice(len(self.memory), self.batch_size,
                                      replace=False, p=probs)
        batch   = [list(self.memory)[i] for i in indices]
        weights = (len(self.memory) * probs[indices]) ** (-self.priority_beta)
        weights = torch.FloatTensor(weights / weights.max()).view(-1, 1)

        s, a, r, ns, d = zip(*batch)
        try:
            s  = torch.FloatTensor(np.array(s))
            ns = torch.FloatTensor(np.array(ns))
        except ValueError as e:
            print(f"Shape error: {e}")
            return

        a      = torch.LongTensor(a).view(-1, 1)
        r      = torch.FloatTensor(r).view(-1, 1)
        d      = torch.FloatTensor(d).view(-1, 1)
        curr_q = self.model(s).gather(1, a)
        max_next_q = self.target_model(ns).max(1)[0].detach().view(-1, 1)
        target_q   = r + self.gamma * max_next_q * (1 - d)

        loss = (weights * nn.MSELoss(reduction='none')(curr_q, target_q)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        td_errors = (curr_q - target_q).detach().abs().squeeze().tolist()
        if isinstance(td_errors, float):
            td_errors = [td_errors]
        for i, idx in enumerate(indices):
            if idx < len(self.priorities):
                self.priorities[idx] = float(td_errors[i]) + 1.0

        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_iter == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        torch.save({'model': self.model.state_dict(), 'epsilon': self.epsilon}, path)

    def load(self, path):
        if os.path.exists(path):
            ck = torch.load(path, weights_only=True)
            self.model.load_state_dict(ck['model'])
            self.target_model.load_state_dict(ck['model'])
            self.epsilon = ck['epsilon']
            print(f"✓ Loaded {path}")


# ----------------- 6. 训练 -----------------
def train(num_episodes=2000, verbose_episodes=3):
    env    = CrossBorderDQNMDP()
    agent  = DQNAgent(env.state_dim, env.action_dim)
    save_path = "agent_v3.pth"
    rewards_history, best_mean = [], -float('inf')

    for e in range(num_episodes):
        state_dict = env.reset()
        state_vec  = env.encoder.encode(state_dict)
        assert state_vec.shape[0] == env.state_dim, \
            f"State dim mismatch: {state_vec.shape[0]} vs {env.state_dim}"

        total_r, done, steps = 0, False, 0
        if (e + 1) <= verbose_episodes:
            print(f"\n{'=' * 80}\n📊 Episode {e + 1}\n{'=' * 80}")

        while not done and steps < 80:
            valid  = env.get_valid_actions(state_dict)
            action = agent.act(state_vec, valid_actions=valid)
            next_dict, reward, done = env.step(state_dict, action)
            next_vec = env.encoder.encode(next_dict)
            agent.remember(state_vec, action, reward, next_vec, done,
                           action_name=env.actions[action])
            state_vec, state_dict = next_vec, next_dict
            total_r += reward
            steps   += 1
            agent.replay()

        rewards_history.append(total_r)
        if (e + 1) <= verbose_episodes:
            print(f"\n✅ Episode {e + 1} 结束: 总奖励={total_r:.1f}, 步数={steps}\n")

        if (e + 1) >= 100:
            mean100 = np.mean(rewards_history[-100:])
            if mean100 > best_mean:
                best_mean = mean100
                agent.save(save_path)
                print(f"⭐ New Best! Ep {e + 1}: Mean={best_mean:.2f}, ε={agent.epsilon:.3f}")

        if (e + 1) % 200 == 0:
            print(f"Ep {e + 1}/{num_episodes}: "
                  f"Last100={np.mean(rewards_history[-100:]):.1f}, "
                  f"ε={agent.epsilon:.3f}")

    print(f"\n✓ Training done! Best Mean Reward: {best_mean:.2f}")
    return agent


# ----------------- 7. 评估 -----------------
def evaluate(agent, num_episodes=1000):
    env = CrossBorderDQNMDP()
    agent.epsilon = 0.0

    stats = {m: {'accept': 0, 'reject': 0, 'mitig': 0,
                 'total_evals': 0, 'rewards': [], 'decisions': 0}
             for m in ['M1', 'M2', 'M3', 'M4']}

    for _ in range(num_episodes):
        state_dict = env.reset()
        state_vec  = env.encoder.encode(state_dict)
        done, steps = False, 0

        while not done and steps < 80:
            valid  = env.get_valid_actions(state_dict)
            action = agent.act(state_vec, valid_actions=valid)
            next_dict, reward, done = env.step(state_dict, action)
            next_vec = env.encoder.encode(next_dict)

            an = env.actions[action]
            mt = state_dict.get('model_type', 'M1')
            if an == 'a_eval'   and mt in stats: stats[mt]['total_evals'] += 1
            if an == 'a_mitig'  and mt in stats: stats[mt]['mitig'] += 1
            if an == 'a_accept' and mt in stats:
                stats[mt]['accept'] += 1
                stats[mt]['rewards'].append(reward)
                stats[mt]['decisions'] += 1
            if an == 'a_reject' and mt in stats:
                stats[mt]['reject'] += 1
                stats[mt]['rewards'].append(reward)
                stats[mt]['decisions'] += 1

            state_vec, state_dict = next_vec, next_dict
            steps += 1

    print("\n【各模型决策统计】")
    for m in ['M1', 'M2', 'M3', 'M4']:
        s   = stats[m]
        tot = s['accept'] + s['reject']
        if tot == 0: continue
        avg_eval = s['total_evals'] / tot
        avg_r    = np.mean(s['rewards']) if s['rewards'] else 0.0
        print(f"  {m}: "
              f"Accept {s['accept']/tot*100:.1f}% | "
              f"Reject {s['reject']/tot*100:.1f}% | "
              f"Mitig {s['mitig']} times | "
              f"Avg Evals/Decision {avg_eval:.1f} | "
              f"Avg Reward {avg_r:.2f}")
    return stats


if __name__ == "__main__":
    trained_agent = train(num_episodes=6000, verbose_episodes=3)
    print("\n" + "=" * 80)
    # evaluate(trained_agent, num_episodes=1000)