# new_cross.py  ── 重构：episode 以单条输入数据为驱动单元
# 核心变化：
#   - candidate_pool 由"模型类型列表"改为"同一输入数据按相似度排序后的模型尝试序列"
#   - reject 后：同一输入回到 s0，用下一个模型模板重新初始化状态，继续评估
#   - accept 后：终态 Accept_Mx_x，episode 结束
#   - 所有候选模型均被 reject：终态 Reject，episode 结束

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
        self.d_map  = {'structured': 0, 'unstructured_text': 1, 'image': 2,
                       'video': 3, 'audio': 4, 'multimodal': 5}
        self.b_map  = {'none': 0, 'verification': 1, 'remote_id': 2, 'categorisation': 3}
        self.r_map  = {'false': 0, 'high-risk': 1}
        self.c_map  = {'EU_only': 0, 'adequacy': 1, 'inadequate_SCC': 2, 'non_compliant': 3}
        self.k_map  = {'start': 0, 'evidence': 1, 'proposed': 2, 'final': 3}
        self.m_map  = {'optimal': 0, 'suboptimal': 1, 'insufficient': 2}
        self.mt_map = {'M1': 0, 'M2': 1, 'M3': 2, 'M4': 3}

    def encode(self, state_dict):
        d, b, r, c = state_dict['d'], state_dict['b'], state_dict['r'], state_dict['c']
        m, k = state_dict['m'], state_dict['k']

        g_vector = state_dict.get('g', [0.25] * 4)
        if not isinstance(g_vector, list) or len(g_vector) != 4:
            g_vector = [0.25] * 4

        vec = [self.d_map[d], self.b_map[b], self.r_map[r],
               self.c_map[c], self.k_map[k], self.m_map[m]]
        vec.extend(g_vector)                                              # +4 = 10

        m_metrics = state_dict.get('m_metrics', [0.8, 0.8, 0.8, 0.1])
        if len(m_metrics) != 4:
            m_metrics = (m_metrics + [0.0] * 4)[:4]
        vec.extend(m_metrics)                                             # +4 = 14

        vec.append(min(state_dict.get('eval_count', 0) / 8.0, 1.0))      # 15
        mt = state_dict.get('model_type', 'M1')
        vec.append(self.mt_map.get(mt, 0) / 3.0)                         # 16
        vec.append(min(state_dict.get('candidates_remaining', 0) / 4.0, 1.0))  # 17
        vec.append(min(state_dict.get('mitig_count', 0) / 3.0, 1.0))     # 18

        return np.array(vec, dtype=np.float32)


# ----------------- 3. MDP 环境 -----------------
class CrossBorderDQNMDP:
    # 模型模板：各模型的标准特征
    TEMPLATES = {
        'M1': {'d': 'structured',  'b': 'none',           'r': 'false',
               'c': 'EU_only',     'm': 'optimal'},
        'M2': {'d': 'multimodal',  'b': 'remote_id',      'r': 'high-risk',
               'c': 'adequacy',    'm': 'suboptimal'},
        'M3': {'d': 'multimodal',  'b': 'remote_id',      'r': 'high-risk',
               'c': 'non_compliant','m': 'suboptimal'},
        'M4': {'d': 'multimodal',  'b': 'categorisation', 'r': 'high-risk',
               'c': 'adequacy',    'm': 'insufficient'},
    }
    FIELDS = ['d', 'b', 'r', 'c', 'm']

    def __init__(self):
        self.encoder    = StateEncoder()
        self.actions    = ['a_next', 'a_eval', 'a_mitig', 'a_accept', 'a_reject']
        self.action_dim = len(self.actions)
        self.state_dim  = 18

        # 奖励参数
        self.R_CORRECT       =  50
        self.R_MISCLASS      = -80
        self.R_FALSE_REJECT  = -55   # 加重：本应 accept 却 reject，激励 Agent 多 accept
        self.R_FALSE_ACCEPT  = -60
        self.R_INFO          =  -1
        self.R_mitig_base    =  -3
        self.R_MITIG_SUCCESS =  15
        self.R_MITIG_FAIL    =  -5
        self.R_EVAL_EXCEEDED = -40
        self.R_RECKLESS      = -10
        self.R_XB            = -25       # 跨境不合规未缓解直接决策
        self.NON_COMPLIANT_C = {'non_compliant', 'inadequate_SCC'}

        self.MAX_MITIG_PER_MODEL = 3
        self.MAX_EVAL_PER_MODEL  = 5
        self.K_PROGRESSION = ['start', 'evidence', 'proposed', 'final']
        self.TIER_ORDER    = {'M1': 0, 'M2': 1, 'M3': 2, 'M4': 3}

        self.EVAL_STRUCTURE = {
            'M1': ['M1', 'M2'],
            'M2': ['M2', 'M3', 'M4'],
            'M3': ['M3', 'M2', 'M4'],
            'M4': ['M4', 'M3', 'M2'],
        }
        self.MITIG_STRUCTURE = {
            'M1': ['M1'],
            'M2': ['M1', 'M3', 'M2'],
            'M3': ['M2', 'M3'],
            'M4': ['M4', 'M2'],
        }
        self.MITIG_FIELD_CHANGES = {
            ('M2', 'M1'): {'d': 'structured', 'b': 'none', 'c': 'EU_only', 'm': 'optimal'},
            ('M2', 'M3'): {'c': 'non_compliant'},
            ('M3', 'M2'): {'c': 'adequacy'},
            ('M4', 'M2'): {'b': 'remote_id', 'c': 'adequacy', 'm': 'suboptimal'},
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
            'M4': {'accept': 0.1, 'reject': 0.9},
        }
        self.sub_types = ['a', 'b', 'c', 'd']
        self.SUBTYPE_NORMAL_PARAMS = {
            'M1': {'mean': 0.5, 'std': 0.6},
            'M2': {'mean': 1.5, 'std': 0.7},
            'M3': {'mean': 2.3, 'std': 0.6},
            'M4': {'mean': 3.0, 'std': 0.5},
        }
        self.DIRICHLET_ALPHA = 2.0

        # episode 级状态
        self.candidate_pool        = []   # 按相似度排序的模型类型列表
        self.current_candidate_idx = -1
        self.current_model_state   = None
        self.episode_true_tiers    = {}
        self.phase                 = 'select'

    # ── 候选池构建 ────────────────────────────────────────
    # 低风险偏置加分：叠加在相似度得分上，使低风险模型在同等相似度下优先尝试。
    # M1=+2, M2=+1 保证当输入特征与多个模型相似度接近时，低风险候选排前。
    # 当输入特征与高风险模型高度匹配时（相似度差距 >2），相似度仍占主导。
    RISK_BONUS = {'M1': 2.0, 'M2': 1.0, 'M3': 0.0, 'M4': 0.0}

    @classmethod
    def build_candidate_pool(cls, input_row: dict, top_n: int = 4) -> list:
        """
        构建候选模型列表，按"加权得分"降序排列：
            加权得分 = 特征相似度（0~5分）+ 低风险偏置加分（M1=+2, M2=+1）

        效果：
          - 输入特征与某模型高度匹配（相似度领先 >2）→ 该模型排前，相似度主导
          - 输入特征相似度接近时 → 低风险模型（M1/M2）优先，加分起决定作用
          - 完全像 M4 的特征（得分4.0）仍排 M4 第一，但 M2(3.0) 优先于 M3(1.0)

        top_n：最多尝试前 N 个模型（默认 4，即全部）。
        """
        top_n = max(1, min(top_n, len(cls.TEMPLATES)))
        scores = {
            mt: sum(1 for f in cls.FIELDS if input_row.get(f) == tmpl.get(f))
                + cls.RISK_BONUS[mt]
            for mt, tmpl in cls.TEMPLATES.items()
        }
        ranked = sorted(scores.keys(), key=lambda m: -scores[m])
        return ranked[:top_n]

    # ── 工具方法 ──────────────────────────────────────────
    def _uniform_probs(self, n):
        return np.ones(n, dtype=np.float32) / n

    def _get_noisy_prob(self, p, noise=0.1):
        return float(np.clip(p + np.random.normal(0, noise), 0.05, 0.95))

    def _get_or_sample_true_tier(self, model_type):
        """同一 episode 内同一模型类型的 Ground Truth 只采样一次"""
        if model_type not in self.episode_true_tiers:
            self.episode_true_tiers[model_type] = int(
                np.random.choice(4, p=self.TRUE_TIER[model_type]))
        return self.episode_true_tiers[model_type]

    def _compute_g_calibration_bonus(self, g_array, true_tier):
        calibration     = float(g_array[true_tier])
        entropy         = float(-np.sum(g_array * np.log(g_array + 1e-8)))
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
        p   = self.SUBTYPE_NORMAL_PARAMS[model_type]
        mu  = float(np.clip(
            p['mean'] + (-0.3 if decision == 'accept' else 0.3), 0.0, 3.0))
        pos = np.array([0., 1., 2., 3.])
        w   = np.exp(-0.5 * ((pos - mu) / p['std']) ** 2)
        w  /= w.sum()
        alpha = np.clip(w * self.DIRICHLET_ALPHA, 1e-3, None)
        pw    = np.random.dirichlet(alpha)
        return self.sub_types[int(np.random.choice(4, p=pw))]

    def _compute_xb_penalty(self, s):
        """跨境不合规且未缓解直接决策时追加惩罚"""
        if s.get('c') in self.NON_COMPLIANT_C and s.get('mitig_count', 0) == 0:
            return self.R_XB
        return 0

    # ── 状态构建 ──────────────────────────────────────────
    def _make_model_state(self, model_type, candidates_remaining=0, mitig_count=0):
        """用模型模板初始化候选状态（eval/mitig 计数从0开始）"""
        base = self.TEMPLATES[model_type].copy()
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
        """select 阶段占位状态"""
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
            return [0]   # 只能 a_next
        valid  = [3, 4]  # a_accept / a_reject 始终可用
        g_conf = max(state_dict.get('g', [0.25] * 4))
        ec     = state_dict.get('eval_count', 0)
        if ec < self.MAX_EVAL_PER_MODEL and g_conf < 0.85:
            valid.append(1)
        if state_dict.get('mitig_count', 0) < self.MAX_MITIG_PER_MODEL:
            valid.append(2)
        return sorted(valid)

    # ── Episode 管理 ──────────────────────────────────────
    # 各特征字段的可选值（训练时随机组合模拟真实数据）
    D_VALUES = ['structured', 'unstructured_text', 'image', 'video', 'audio', 'multimodal']
    B_VALUES = ['none', 'verification', 'remote_id', 'categorisation']
    R_VALUES = ['false', 'high-risk']
    C_VALUES = ['EU_only', 'adequacy', 'inadequate_SCC', 'non_compliant']
    M_VALUES = ['optimal', 'suboptimal', 'insufficient']

    def reset(self, top_n: int = 4):
        """
        训练时：随机组合各字段值生成输入特征（模拟真实数据分布），
        使候选池顺序固定为 M1→M2→M3→M4，与测试时行为一致。

        70% 概率偏向低风险特征（r=false / b=none / c=EU_only），
        使训练样本 Accept/Reject 分布均衡。
        """
        low_risk_bias = np.random.random() < 0.7
        fake_row = {
            'd': np.random.choice(self.D_VALUES),
            'b': np.random.choice(['none', 'verification'])
                 if low_risk_bias else np.random.choice(self.B_VALUES),
            'r': 'false'     if low_risk_bias else np.random.choice(self.R_VALUES),
            'c': np.random.choice(['EU_only', 'adequacy'])
                 if low_risk_bias else np.random.choice(self.C_VALUES),
            'm': np.random.choice(self.M_VALUES),
        }
        self.candidate_pool        = self.build_candidate_pool(fake_row, top_n=top_n)
        self.current_candidate_idx = -1
        self.current_model_state   = None
        self.episode_true_tiers    = {}
        self.phase = 'select'
        print(f"🎬 [Episode Start] 输入特征={fake_row} | top_n={top_n} | "
              f"候选顺序: {self.candidate_pool}")
        return self._s0_state()

    def reset_with_input(self, input_row: dict, top_n: int = 4):
        """
        测试时：给定真实输入行，按相似度排序构建候选池。
        top_n 控制最多尝试几个模型（默认 4，即全部）。
        """
        self.candidate_pool        = self.build_candidate_pool(input_row, top_n=top_n)
        self.current_candidate_idx = -1
        self.current_model_state   = None
        self.episode_true_tiers    = {}
        self.phase = 'select'
        return self._s0_state()

    # ── 核心步进 ──────────────────────────────────────────
    def step(self, current_state_dict, action_idx):
        action       = self.actions[action_idx]
        reward, done = self.R_INFO, False

        valid = self.get_valid_actions(current_state_dict)
        if action_idx not in valid:
            print(f"❌ [Invalid] {action} 不可用，合法: {[self.actions[i] for i in valid]}")
            return current_state_dict, -50, False

        # ── a_next：取下一个候选模型，用其模板初始化状态 ──────────────────
        if action == 'a_next':
            self.current_candidate_idx += 1
            model_type = self.candidate_pool[self.current_candidate_idx]
            remaining  = len(self.candidate_pool) - self.current_candidate_idx - 1
            self.current_model_state = self._make_model_state(
                model_type, candidates_remaining=remaining)
            self.phase = 'evaluate'
            print(f"🚀 [Next] 尝试模型 #{self.current_candidate_idx + 1}"
                  f"/{len(self.candidate_pool)}: {model_type} | 剩余: {remaining}")
            return self.current_model_state, self.R_INFO, False

        # ── a_eval ────────────────────────────────────────────────────────
        elif action == 'a_eval':
            s = self.current_model_state
            if s['eval_count'] >= self.MAX_EVAL_PER_MODEL:
                print(f"⚠️  [Eval limit] 已评估 {s['eval_count']} 次，必须决策！")
                return self.current_model_state, self.R_EVAL_EXCEEDED, False

            s['eval_count'] += 1
            curr_type      = s['model_type']
            possible_types = self.EVAL_STRUCTURE[curr_type]
            next_type      = str(np.random.choice(
                possible_types, p=self._uniform_probs(len(possible_types))))

            if next_type != curr_type:
                print(f"🔄 [Reclassify] {curr_type} → {next_type}")
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
            prog_pen = -float(s['eval_count'])
            conf_pen = -4.0 * max(0, g_conf - 0.65)
            early_b  = 1.5 if s['eval_count'] <= 2 else 0.0
            reward   = prog_pen + conf_pen + early_b

            tier_names = ['minimal', 'limited', 'high', 'banned']
            dominant   = tier_names[int(np.argmax(g_array))]
            kc         = " (k↑)" if s['k'] != old_k else ""
            es = "🟡" if s['eval_count'] < 3 else "🟠" if s['eval_count'] < 5 else "🔴"
            print(f"{es} [Eval {s['eval_count']}/{self.MAX_EVAL_PER_MODEL}] "
                  f"{s['model_type']} | k={s['k']}{kc} | "
                  f"g→{dominant}({g_conf:.2f}) | reward={reward:.2f}")
            if s['eval_count'] == self.MAX_EVAL_PER_MODEL:
                print("🚨 [Max evals] 下一步必须决策！")
                reward -= 3
            return self.current_model_state, reward, False

        # ── a_mitig ───────────────────────────────────────────────────────
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
                new_features = {f: s[f] for f in self.FIELDS}
                new_features.update(field_changes)
                new_g_prior = RiskClassifier.compute_initial_g(
                    new_features['d'], new_features['b'],
                    new_features['r'], new_features['c'], new_features['m'])
                blended_g = [(prev_g[i] + new_g_prior[i]) / 2.0 for i in range(4)]
                g_sum     = sum(blended_g)
                blended_g = [x / g_sum for x in blended_g]
                s = {**new_features,
                     'k': prev_k, 'g': blended_g,
                     'm_metrics': [0.8, 0.8, 0.8, 0.1],
                     'eval_count': prev_eval, 'mitig_count': mitig_count,
                     'model_type': next_type, 'candidates_remaining': remaining}
                self.current_model_state = s
                changes_desc = ', '.join(f"{k}:→{v}" for k, v in field_changes.items())
                print(f"🔧 [Mitig #{s['mitig_count']}] [{changes_desc}]")
            else:
                print(f"🔧 [Mitig #{s['mitig_count']}] 缓解无效果")

            is_sideshift = (curr_type == 'M2' and next_type == 'M3')
            if next_tier < curr_tier:
                reward = self.R_mitig_base + self.R_MITIG_SUCCESS
                print(f"🔧✅ {curr_type}→{next_type} 风险下降 | reward={reward}")
            elif next_tier == curr_tier or is_sideshift:
                reward = self.R_mitig_base
                print(f"🔧⚪ {curr_type}→{next_type} 无效/侧移 | reward={reward}")
            else:
                reward = self.R_mitig_base + self.R_MITIG_FAIL
                print(f"🔧❌ {curr_type}→{next_type} 风险上升 | reward={reward}")

            if s['mitig_count'] >= self.MAX_MITIG_PER_MODEL:
                print(f"⚠️  [Mitig limit] 已缓解 {s['mitig_count']} 次")
            return self.current_model_state, reward, False

        # ── a_accept：输入数据通过当前模型，episode 成功结束 ──────────────
        elif action == 'a_accept':
            s          = self.current_model_state
            model_type = s['model_type']
            true_tier  = self._get_or_sample_true_tier(model_type)
            tier_names = ['minimal', 'limited', 'high', 'banned']

            direction_correct = (true_tier in self.ACCEPT_CORRECT_TIERS)
            g_array    = np.array(s['g'])
            g_bonus    = self._compute_g_calibration_bonus(g_array, true_tier)
            k_bonus    = {'start': 0, 'evidence': 0.05,
                          'proposed': 0.1, 'final': 0.15}[s['k']]
            eval_bonus = (0.1 if 3 <= s['eval_count'] <= 6
                          else -0.1 if s['eval_count'] < 3 else -0.05)
            base_p     = self.DECISION_BASE_PROBS[model_type]['accept']
            p_correct  = self._get_noisy_prob(base_p + k_bonus + eval_bonus + g_bonus)

            reckless   = (self.R_RECKLESS
                          if model_type in ['M3', 'M4'] and s['eval_count'] < 2 else 0)
            xb_penalty = self._compute_xb_penalty(s)

            if direction_correct:
                is_correct = np.random.random() < p_correct
                reward     = (self.R_CORRECT if is_correct else self.R_MISCLASS)
            else:
                is_correct = np.random.random() < 0.05
                reward     = (self.R_CORRECT if is_correct else self.R_FALSE_ACCEPT)
            reward += reckless + xb_penalty

            sub      = self._select_subtype(model_type, 'accept')
            dominant = tier_names[int(np.argmax(g_array))]
            g_match  = "✓" if int(np.argmax(g_array)) == true_tier else "✗"
            extras   = (f" ⚡reckless={reckless}" if reckless else "") + \
                       (f" 🌐xb={xb_penalty}"    if xb_penalty < 0 else "")
            print(f"{'✅' if is_correct else '❌'} [Accept] {model_type} → "
                  f"Accept_{model_type}_{sub} | "
                  f"true={tier_names[true_tier]}, g→{dominant}[{g_match}] | "
                  f"k={s['k']}, evals={s['eval_count']}{extras}")
            return self._terminal_state(), reward, True

        # ── a_reject：该输入不通过当前模型 ───────────────────────────────
        #   - 还有候选模型：回到 s0，等待 a_next 尝试下一个模型
        #   - 所有模型耗尽：episode 失败结束（Reject 终态）
        elif action == 'a_reject':
            s          = self.current_model_state
            model_type = s['model_type']
            true_tier  = self._get_or_sample_true_tier(model_type)
            tier_names = ['minimal', 'limited', 'high', 'banned']

            direction_correct = (true_tier in self.REJECT_CORRECT_TIERS)
            g_array    = np.array(s['g'])
            g_bonus    = self._compute_g_calibration_bonus(g_array, true_tier)
            k_bonus    = {'start': 0, 'evidence': 0.05,
                          'proposed': 0.1, 'final': 0.15}[s['k']]
            eval_bonus = (0.1 if 3 <= s['eval_count'] <= 6
                          else -0.1 if s['eval_count'] < 3 else -0.05)
            base_p     = self.DECISION_BASE_PROBS[model_type]['reject']
            p_correct  = self._get_noisy_prob(base_p + k_bonus + eval_bonus + g_bonus)

            reckless   = (self.R_RECKLESS
                          if model_type in ['M3', 'M4'] and s['eval_count'] < 2 else 0)
            xb_penalty = self._compute_xb_penalty(s)

            if direction_correct:
                is_correct = np.random.random() < p_correct
                reward     = (self.R_CORRECT if is_correct else self.R_MISCLASS)
            else:
                is_correct = np.random.random() < 0.05
                reward     = (self.R_CORRECT if is_correct else self.R_FALSE_REJECT)
            reward += reckless + xb_penalty

            remaining = len(self.candidate_pool) - self.current_candidate_idx - 1
            dominant  = tier_names[int(np.argmax(g_array))]
            g_match   = "✓" if int(np.argmax(g_array)) == true_tier else "✗"
            extras    = (f" ⚡reckless={reckless}" if reckless else "") + \
                        (f" 🌐xb={xb_penalty}"    if xb_penalty < 0 else "")

            if remaining > 0:
                # 还有候选模型：同一输入回到 s0，准备尝试下一个模型
                self.phase = 'select'
                self.current_model_state = None
                next_state = self._s0_state()
                done = False
                print(f"{'✅' if is_correct else '❌'} [Reject] {model_type} | "
                      f"true={tier_names[true_tier]}, g→{dominant}[{g_match}] | "
                      f"剩余候选={remaining}{extras} → 回到 s0 尝试下一模型")
            else:
                # 所有模型均被拒绝：episode 彻底失败，记录最终拒绝的子类标签
                sub        = self._select_subtype(model_type, 'reject')
                reject_label = f"Reject_{model_type}_{sub}"
                next_state = self._terminal_state()
                next_state['final_label'] = reject_label   # 供调用方读取
                done = True
                print(f"{'✅' if is_correct else '❌'} [Reject+AllExhausted] "
                      f"{model_type} → {reject_label} | "
                      f"true={tier_names[true_tier]}, g→{dominant}[{g_match}]{extras}")
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


# ----------------- 5. DQN Agent -----------------
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
            print(f"Shape error: {e}"); return

        a        = torch.LongTensor(a).view(-1, 1)
        r        = torch.FloatTensor(r).view(-1, 1)
        d        = torch.FloatTensor(d).view(-1, 1)
        curr_q   = self.model(s).gather(1, a)
        max_next = self.target_model(ns).max(1)[0].detach().view(-1, 1)
        target_q = r + self.gamma * max_next * (1 - d)

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
    env   = CrossBorderDQNMDP()
    agent = DQNAgent(env.state_dim, env.action_dim)
    save_path = "agent_v3.pth"
    rewards_history, best_mean = [], -float('inf')

    for e in range(num_episodes):
        state_dict = env.reset()
        state_vec  = env.encoder.encode(state_dict)
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


if __name__ == "__main__":
    trained_agent = train(num_episodes=1000, verbose_episodes=3)