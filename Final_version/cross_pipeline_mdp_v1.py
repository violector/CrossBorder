# cross_pipeline_mdp.py
# Pipeline MDP Model for Cross-Border AI Governance
# Based on: MDP_Flow_CN.md (pipeline interpretation of Nature Communications 2026 paper)
#
# Key design changes from cross_from_pdf.py:
#   1. All inputs enter via S0 → a_next → always M1 (pipeline, not candidate pool)
#   2. M1–M4 are sequential governance stages, not independent model types
#   3. No candidate pool, no feature-based routing — each request is independent
#   4. Stage transitions are action-driven, bidirectional (forward via eval, backward via mitig)
#   5. M4 a_accept probability = 0.1 (per mermaid.txt, not blocked)
#   6. Reward values per spec §8
#   7. Output O = {verified, rejected, uncertain, abstained, escalated}

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os

# ─── 1. Risk Classifier ───────────────────────────────────────────────────────
class RiskClassifier:
    """Computes initial risk belief g and performs Bayesian updates."""

    @staticmethod
    def compute_initial_g(d, b, r, c, m):
        """Score-based initialisation with softmax normalisation (spec §2, g field)."""
        score = np.array([1.0, 0.5, 0.1, 0.0])

        if b == 'categorisation':
            score += [0.0, 0.0, 0.5, 5.0]
        elif b == 'remote_id':
            score += [0.0, 0.1, 2.5, 0.5]
        elif b == 'verification':
            score += [0.0, 0.8, 0.2, 0.0]

        if r == 'high-risk':
            score += [0.0, 0.15, 3.0, 0.3]

        if c == 'non_compliant':
            score += [0.0, 0.2, 2.0, 0.5]
        elif c == 'inadequate_SCC':
            score += [0.0, 0.5, 1.0, 0.1]
        elif c == 'adequacy':
            score += [0.0, 0.3, 0.2, 0.0]

        if d in ['multimodal', 'video']:
            score += [0.0, 0.3, 1.2, 0.1]
        elif d in ['audio', 'image']:
            score += [0.0, 0.5, 0.3, 0.0]

        if m == 'insufficient':
            score += [0.0, 0.1, 0.5, 1.5]
        elif m == 'suboptimal':
            score += [0.0, 0.5, 0.5, 0.0]
        elif m == 'optimal':
            score += [0.8, 0.5, 0.0, 0.0]

        exp_s = np.exp(score - np.max(score))
        return (exp_s / exp_s.sum()).tolist()

    @staticmethod
    def bayesian_update_g(prior_g, evidence_likelihood, noise_scale=0.05):
        """Bayesian belief update (spec §7, Eq.9)."""
        prior = np.array(prior_g)
        likelihood = np.array(evidence_likelihood)
        likelihood += np.random.normal(0, noise_scale, 4)
        likelihood = np.clip(likelihood, 0.01, 10.0)
        posterior = np.clip(prior * likelihood, 1e-6, 1.0)
        return (posterior / posterior.sum()).tolist()


# ─── 2. State Encoder ────────────────────────────────────────────────────────
class StateEncoder:
    """Encodes the structured state dict into a fixed 22-dim float vector."""

    def __init__(self):
        self.d_map  = {'structured': 0, 'unstructured_text': 1, 'image': 2,
                       'video': 3, 'audio': 4, 'multimodal': 5}
        self.b_map  = {'none': 0, 'verification': 1, 'remote_id': 2, 'categorisation': 3}
        self.r_map  = {'false': 0, 'high-risk': 1}
        self.c_map  = {'EU_only': 0, 'adequacy': 1, 'inadequate_SCC': 2, 'non_compliant': 3}
        self.j_map  = {'EU_member': 0, 'adequacy_country': 1, 'third_country': 2}
        self.k_map  = {'start': 0, 'evidence': 1, 'proposed': 2, 'final': 3}
        self.m_map  = {'optimal': 0, 'suboptimal': 1, 'insufficient': 2}
        self.stage_map = {'S0': 0, 'M1': 1, 'M2': 2, 'M3': 3, 'M4': 4}

    def encode(self, state_dict):
        d, b, r, c = state_dict['d'], state_dict['b'], state_dict['r'], state_dict['c']
        j = state_dict.get('j', 'EU_member')
        m, k = state_dict['m'], state_dict['k']

        g_vector = state_dict.get('g', [0.25] * 4)
        if not isinstance(g_vector, list) or len(g_vector) != 4:
            g_vector = [0.25] * 4

        vec = [self.d_map.get(d, 0), self.b_map.get(b, 0), self.r_map.get(r, 0),
               self.c_map.get(c, 0), self.j_map.get(j, 0), self.k_map.get(k, 0),
               self.m_map.get(m, 0)]
        vec.extend(g_vector)                                              # +4 = 11

        m_metrics = state_dict.get('m_metrics', [0.8, 0.8, 0.8, 0.1])
        if len(m_metrics) != 4:
            m_metrics = [0.8, 0.8, 0.8, 0.1]
        vec.extend(m_metrics)                                             # +4 = 15

        vec.append(min(state_dict.get('eval_count', 0) / 8.0, 1.0))      # 16
        stage = state_dict.get('stage', 'S0')
        vec.append(self.stage_map.get(stage, 0) / 4.0)                    # 17
        vec.append(0.0)                                                    # 18 (padding, was candidates_remaining)
        vec.append(min(state_dict.get('mitig_count', 0) / 3.0, 1.0))     # 19
        vec.append(min(state_dict.get('verify_count', 0) / 3.0, 1.0))    # 20
        vec.append(min(state_dict.get('defer_count', 0) / 3.0, 1.0))     # 21
        vec.append(min(state_dict.get('redact_count', 0) / 1.0, 1.0))    # 22

        return np.array(vec, dtype=np.float32)


# ─── 3. Pipeline MDP Environment ──────────────────────────────────────────────
class PipelineMDP:
    """
    Pipeline MDP per MDP_Flow_CN.md:
    - S0 → a_next → always M1 (pipeline entry, §4.1)
    - M1–M4 are sequential governance stages (§2)
    - Stage transitions via action-driven probabilities (§5)
    - Terminal actions produce O = {verified, rejected, uncertain, abstained, escalated} (§4.2)
    """

    # ── Transition probabilities from MDP_Flow_CN.md §5 ─────────────────

    # §5.1 a_eval — M1→M2 slightly increased for pipeline learnability
    EVAL_PROBS = {
        'M1': {'M1': 0.85, 'M2': 0.15},   # was 0.90/0.10; 0.15 aids exploration
        'M2': {'M2': 0.80, 'M3': 0.10, 'M4': 0.10},
        'M3': {'M3': 0.70, 'M2': 0.25, 'M4': 0.05},
        'M4': {'M4': 1.00},
    }

    # §5.2 a_mitig
    MITIG_PROBS = {
        'M1': {'M1': 1.00},
        'M2': {'M2': 0.60, 'M1': 0.20, 'M3': 0.20},
        'M3': {'M3': 0.50, 'M2': 0.50},
        'M4': {'M4': 0.80, 'M2': 0.20},
    }

    # §5.3 a_verify
    VERIFY_PROBS = {
        'M1': {'M1': 0.95, 'M2': 0.05},
        'M2': {'M2': 0.85, 'M3': 0.10, 'M4': 0.05},
        'M3': {'M3': 0.75, 'M2': 0.20, 'M4': 0.05},
        'M4': {'M4': 0.95, 'M3': 0.05},
    }

    # §5.4 a_escalate
    ESCALATE_PROBS = {
        'M1': {'M1': 0.95, 'M2': 0.05},
        'M2': {'M2': 0.80, 'M1': 0.10, 'M3': 0.10},
        'M3': {'M3': 0.70, 'M2': 0.25, 'M4': 0.05},
        'M4': {'M4': 0.90, 'M3': 0.10},
    }

    # §5.5 a_defer
    DEFER_PROBS = {
        'M1': {'M1': 0.98, 'M2': 0.02},
        'M2': {'M2': 0.90, 'M1': 0.05, 'M3': 0.05},
        'M3': {'M3': 0.85, 'M2': 0.12, 'M4': 0.03},
        'M4': {'M4': 0.92, 'M3': 0.08},
    }

    # §5.6 a_redact
    REDACT_PROBS = {
        'M1': {'M1': 1.00},
        'M2': {'M2': 0.55, 'M1': 0.45},
        'M3': {'M3': 0.45, 'M2': 0.35, 'M1': 0.20},
        'M4': {'M4': 0.40, 'M3': 0.25, 'M2': 0.35},
    }

    # §5.7 a_accept / a_reject decision outcome probabilities
    # M4 accept = 0.1 per mermaid.txt (no hard block)
    DECISION_PROBS = {
        'M1': {'accept': 0.90, 'reject': 0.10},
        'M2': {'accept': 0.70, 'reject': 0.30},
        'M3': {'accept': 0.60, 'reject': 0.40},
        'M4': {'accept': 0.10, 'reject': 0.90},
    }

    # Stage ordering for tier comparison
    STAGE_ORDER = {'M1': 0, 'M2': 1, 'M3': 2, 'M4': 3}
    STAGES = ['M1', 'M2', 'M3', 'M4']
    TIER_NAMES = ['minimal', 'limited', 'high', 'prohibited']

    # Feature change map for stage transitions during mitig/redact
    # When stage downgrades, features become less risky
    STAGE_FEATURE_CHANGES = {
        ('M2', 'M1'): {'d': 'structured', 'b': 'none', 'c': 'EU_only', 'm': 'optimal'},
        ('M2', 'M3'): {'c': 'non_compliant'},
        ('M3', 'M2'): {'c': 'adequacy'},
        ('M3', 'M1'): {'d': 'structured', 'b': 'verification', 'c': 'EU_only', 'm': 'optimal'},
        ('M4', 'M3'): {'b': 'remote_id', 'm': 'suboptimal'},
        ('M4', 'M2'): {'b': 'remote_id', 'c': 'adequacy', 'm': 'suboptimal'},
    }

    # ── Reward parameters (MDP_Flow_CN.md §8) ───────────────────────────
    R_CORRECT_ACCEPT     =  50   # correct accept (true ∈ {minimal, limited})
    R_CORRECT_REJECT     =  50   # correct reject (true ∈ {high, prohibited})
    R_MISCLASS           = -80   # decision contradicts true tier
    R_FALSE_ACCEPT       = -60   # accept on high/prohibited
    R_FALSE_REJECT       = -55   # reject on minimal/limited
    R_STEP               =  -1   # each non-terminal step
    R_MITIG_SUCCESS      =  12   # mitig causes stage downgrade
    R_MITIG_FAIL         =  -8   # stage upgrade or no change
    R_VERIFY_BONUS       =   4   # verify when g_conf > 0.6
    R_VERIFY_PROGRESS    =   3   # verify causes stage upgrade (encourages pipeline)
    R_ESCALATE_COST      = -20   # each escalate (increased to dissuade over-use)
    R_ABSTAIN_CORRECT    = -15   # abstain when true ∈ {high, prohibited}
    R_ABSTAIN_WRONG      = -25   # abstain when true ∈ {minimal, limited}
    R_DEFER_CORRECT      =  -4   # high risk, high uncertainty
    R_DEFER_WRONG         = -15   # unnecessary delay
    R_DEFER_CONSECUTIVE   =  -3   # extra penalty per consecutive defer (escalating)
    R_REDACT_SUCCESS     =   4   # redact causes stage downgrade
    R_XB_PENALTY         = -25   # cross-border non-compliant, unmitigated
    R_RECKLESS           = -10   # M3/M4 with eval < 2

    # Resource limits (spec §9)
    MAX_EVAL_PER_STAGE   = 5
    MAX_MITIG_PER_STAGE  = 3
    MAX_VERIFY_PER_STAGE = 3
    MAX_DEFER_PER_STAGE  = 2
    MAX_REDACT_PER_STAGE = 1

    K_PROGRESSION = ['start', 'evidence', 'proposed', 'final']

    # Input feature value domains
    D_VALUES = ['structured', 'unstructured_text', 'image', 'video', 'audio', 'multimodal']
    B_VALUES = ['none', 'verification', 'remote_id', 'categorisation']
    R_VALUES = ['false', 'high-risk']
    C_VALUES = ['EU_only', 'adequacy', 'inadequate_SCC', 'non_compliant']
    J_VALUES = ['EU_member', 'adequacy_country', 'third_country']
    M_VALUES = ['optimal', 'suboptimal', 'insufficient']

    NON_COMPLIANT_C = {'non_compliant', 'inadequate_SCC'}
    ACCEPT_CORRECT_TIERS = {0, 1}    # minimal, limited
    REJECT_CORRECT_TIERS = {2, 3}    # high, prohibited

    def __init__(self):
        self.encoder    = StateEncoder()
        # 10 actions (spec §3)
        self.actions    = ['a_next', 'a_eval', 'a_mitig', 'a_verify',
                           'a_escalate', 'a_abstain', 'a_accept', 'a_reject',
                           'a_defer', 'a_redact']
        self.action_dim = len(self.actions)
        self.state_dim  = 22

        # Episode-level state
        self.episode_true_tier = None
        self.input_features    = None

    # ── Helpers ───────────────────────────────────────────────────────────
    @staticmethod
    def _weighted_choice(prob_dict):
        items = list(prob_dict.items())
        keys  = [k for k, _ in items]
        probs = [v for _, v in items]
        return str(np.random.choice(keys, p=probs))

    def _compute_true_tier(self, d, b, r, c, m):
        """Compute ground-truth risk tier from input features (deterministic core + noise)."""
        score = 0.0
        if b == 'categorisation':       score += 2.5
        elif b == 'remote_id':          score += 1.5
        elif b == 'verification':       score += 0.5

        if r == 'high-risk':            score += 1.5

        if c == 'non_compliant':        score += 1.5
        elif c == 'inadequate_SCC':     score += 1.0

        if m == 'insufficient':         score += 1.5
        elif m == 'suboptimal':         score += 0.5

        if d in ['multimodal', 'video']: score += 0.5

        score += np.random.normal(0, 0.3)

        if score < 1.0:      return 0   # minimal
        elif score < 2.0:    return 1   # limited
        elif score < 3.5:    return 2   # high
        else:                return 3   # prohibited

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

    def _generate_evidence_likelihood(self, stage, eval_count):
        """Generate evidence likelihood with realistic ambiguity.
        Evidence generated per spec noise levels — not all cases converge to certainty."""
        true_tier = self.episode_true_tier
        # True tier signal is weak relative to noise — realistic evidence quality
        base = np.ones(4) * 0.1
        base[true_tier] = 1.0

        # Higher base noise prevents perfect convergence
        noise_scale = max(0.03, 0.10 - eval_count * 0.01)
        return np.clip(base + np.random.normal(0, noise_scale, 4), 0.01, 2.0).tolist()

    def _compute_xb_penalty(self, s):
        if s.get('c') in self.NON_COMPLIANT_C and s.get('mitig_count', 0) == 0:
            return self.R_XB_PENALTY
        return 0

    # ── State builders ────────────────────────────────────────────────────
    def _s0_state(self):
        return {
            'd': 'structured', 'b': 'none', 'r': 'false', 'c': 'EU_only',
            'j': 'EU_member', 'k': 'start', 'm': 'optimal',
            'g': [0.25] * 4, 'm_metrics': [0.0] * 4,
            'eval_count': 0, 'mitig_count': 0, 'verify_count': 0,
            'defer_count': 0, 'redact_count': 0,
            'stage': 'S0', 'escalated': False,
        }

    def _make_m1_state(self):
        """Create M1 state from episode input features."""
        f = self.input_features
        g = RiskClassifier.compute_initial_g(f['d'], f['b'], f['r'], f['c'], f['m'])
        return {
            'd': f['d'], 'b': f['b'], 'r': f['r'], 'c': f['c'],
            'j': f.get('j', 'EU_member'), 'm': f['m'],
            'k': 'start', 'g': g,
            'm_metrics': [0.8, 0.8, 0.8, 0.1],
            'eval_count': 0, 'mitig_count': 0, 'verify_count': 0,
            'defer_count': 0, 'redact_count': 0,
            'stage': 'M1', 'escalated': False,
        }

    def _terminal_state(self, paper_output):
        s = self._s0_state()
        s['paper_output'] = paper_output
        s['stage'] = 'S0'
        return s

    # ── Episode management ────────────────────────────────────────────────
    def reset(self):
        """Training reset: random input feature generation (50/50 low/high risk)."""
        low_risk_bias = np.random.random() < 0.5
        self.input_features = {
            'd': np.random.choice(self.D_VALUES),
            'b': np.random.choice(['none', 'verification'])
                 if low_risk_bias else np.random.choice(self.B_VALUES),
            'r': 'false' if low_risk_bias else np.random.choice(self.R_VALUES),
            'c': np.random.choice(['EU_only', 'adequacy'])
                 if low_risk_bias else np.random.choice(self.C_VALUES),
            'm': np.random.choice(self.M_VALUES),
            'j': np.random.choice(['EU_member', 'adequacy_country'])
                 if low_risk_bias else np.random.choice(self.J_VALUES),
        }
        self.episode_true_tier = self._compute_true_tier(
            self.input_features['d'], self.input_features['b'],
            self.input_features['r'], self.input_features['c'],
            self.input_features['m'])
        return self._s0_state()

    def reset_with_input(self, input_row: dict):
        """Test-time reset with given input features."""
        self.input_features = {
            'd': input_row.get('d', 'structured'),
            'b': input_row.get('b', 'none'),
            'r': input_row.get('r', 'false'),
            'c': input_row.get('c', 'EU_only'),
            'm': input_row.get('m', 'optimal'),
            'j': input_row.get('j', 'EU_member'),
        }
        self.episode_true_tier = self._compute_true_tier(
            self.input_features['d'], self.input_features['b'],
            self.input_features['r'], self.input_features['c'],
            self.input_features['m'])
        return self._s0_state()

    # ── Apply stage transition with feature changes ───────────────────────
    def _apply_stage_transition(self, current_state, next_stage):
        """Move to a new governance stage, updating features and blending beliefs."""
        curr_stage = current_state['stage']
        if next_stage == curr_stage:
            return current_state

        # Apply feature changes for this transition
        field_changes = self.STAGE_FEATURE_CHANGES.get((curr_stage, next_stage), {})
        new_state = dict(current_state)
        for f, v in field_changes.items():
            new_state[f] = v

        # Blend belief vectors
        prev_g = current_state['g'][:]
        new_g_prior = RiskClassifier.compute_initial_g(
            new_state['d'], new_state['b'], new_state['r'],
            new_state['c'], new_state['m'])
        blended_g = [(prev_g[i] + new_g_prior[i]) / 2.0 for i in range(4)]
        g_sum = sum(blended_g)
        new_state['g'] = [x / g_sum for x in blended_g] if g_sum > 0 else new_g_prior
        new_state['stage'] = next_stage
        return new_state

    # ── Valid actions per stage (§6) ──────────────────────────────────────
    def get_valid_actions(self, state_dict):
        stage = state_dict.get('stage', 'S0')

        if stage == 'S0':
            return [0]   # only a_next

        valid = []
        g_conf = max(state_dict.get('g', [0.25] * 4))
        ec     = state_dict.get('eval_count', 0)

        # a_eval (all stages)
        if ec < self.MAX_EVAL_PER_STAGE:
            valid.append(1)

        # a_mitig (all stages)
        if state_dict.get('mitig_count', 0) < self.MAX_MITIG_PER_STAGE:
            valid.append(2)

        # a_verify (M2/M3 only, §6)
        if (stage in ['M2', 'M3'] and
            state_dict.get('verify_count', 0) < self.MAX_VERIFY_PER_STAGE):
            valid.append(3)

        # a_escalate (genuine uncertainty or after multiple evals, §6)
        if g_conf < 0.70 or (ec >= 3 and g_conf < 0.85):
            valid.append(4)

        # a_abstain (always available, §6)
        valid.append(5)

        # a_accept (all stages)
        valid.append(6)

        # a_reject (all stages)
        valid.append(7)

        # a_defer (M1/M2/M3 only, not M4, §6)
        if stage != 'M4' and state_dict.get('defer_count', 0) < self.MAX_DEFER_PER_STAGE:
            valid.append(8)

        # a_redact (b ∈ {categorisation, remote_id} and redact_count < 1, §6)
        if (state_dict.get('redact_count', 0) < self.MAX_REDACT_PER_STAGE
            and state_dict.get('b') in ['categorisation', 'remote_id']):
            valid.append(9)

        return sorted(valid)

    # ── Core step ─────────────────────────────────────────────────────────
    def step(self, current_state_dict, action_idx):
        action = self.actions[action_idx]
        reward = self.R_STEP
        done   = False

        valid = self.get_valid_actions(current_state_dict)
        if action_idx not in valid:
            return current_state_dict, -50, False

        # ── a_next (S0 → M1 always, §4.1) ────────────────────────────────
        if action == 'a_next':
            next_state = self._make_m1_state()
            return next_state, self.R_STEP, False

        # For actions below, we're in a governance stage
        s = current_state_dict
        stage = s['stage']

        # ── a_eval (§5.1) ────────────────────────────────────────────────
        if action == 'a_eval':
            if s['eval_count'] >= self.MAX_EVAL_PER_STAGE:
                return s, -40, False

            s = dict(s)
            s['eval_count'] += 1
            next_stage = self._weighted_choice(self.EVAL_PROBS[stage])

            if next_stage != stage:
                s = self._apply_stage_transition(s, next_stage)

            noise = max(0.06, 0.18 - s['eval_count'] * 0.02)
            evidence = self._generate_evidence_likelihood(s['stage'], s['eval_count'])
            s['g'] = RiskClassifier.bayesian_update_g(
                prior_g=s['g'], evidence_likelihood=evidence, noise_scale=noise)
            old_k = s['k']
            s['k'] = self._update_k_value(s['k'], s['eval_count'])

            g_array = np.array(s['g'])
            g_conf  = float(np.max(g_array))
            prog_pen = -float(s['eval_count']) * 0.5
            conf_pen = -4.0 * max(0, g_conf - 0.65)
            early_b  = 3.0 if s['eval_count'] <= 2 else 0.0
            reward   = self.R_STEP + prog_pen + conf_pen + early_b

            # Stage progression bonus: encourage pipeline traversal
            if next_stage != stage and self.STAGE_ORDER.get(next_stage, 0) > self.STAGE_ORDER.get(stage, 0):
                reward += 5  # stage upgrade bonus

            if s['eval_count'] == self.MAX_EVAL_PER_STAGE:
                reward -= 3
            return s, reward, False

        # ── a_mitig (§5.2) ───────────────────────────────────────────────
        elif action == 'a_mitig':
            s = dict(s)
            s['mitig_count'] += 1
            next_stage = self._weighted_choice(self.MITIG_PROBS[stage])
            curr_tier = self.STAGE_ORDER[stage]
            next_tier = self.STAGE_ORDER[next_stage]

            if next_stage != stage:
                s = self._apply_stage_transition(s, next_stage)
            else:
                evidence = self._generate_evidence_likelihood(stage, s['eval_count'])
                s['g'] = RiskClassifier.bayesian_update_g(
                    prior_g=s['g'], evidence_likelihood=evidence, noise_scale=0.05)

            if next_tier < curr_tier:
                reward = self.R_STEP + self.R_MITIG_SUCCESS
            elif next_tier == curr_tier:
                reward = self.R_STEP - 3   # base mitig cost
            else:
                reward = self.R_STEP + self.R_MITIG_FAIL
            return s, reward, False

        # ── a_verify (§5.3) ──────────────────────────────────────────────
        elif action == 'a_verify':
            s = dict(s)
            s['verify_count'] += 1
            next_stage = self._weighted_choice(self.VERIFY_PROBS[stage])

            if next_stage != stage:
                s = self._apply_stage_transition(s, next_stage)

            evidence = self._generate_evidence_likelihood(s['stage'], s['eval_count'])
            s['g'] = RiskClassifier.bayesian_update_g(
                prior_g=s['g'], evidence_likelihood=evidence, noise_scale=0.03)

            g_array = np.array(s['g'])
            g_conf  = float(np.max(g_array))
            reward  = self.R_STEP - 1  # base verify cost
            if g_conf > 0.6:
                reward += self.R_VERIFY_BONUS
            if next_stage != stage and self.STAGE_ORDER.get(next_stage, 0) > self.STAGE_ORDER.get(stage, 0):
                reward += self.R_VERIFY_PROGRESS
            return s, reward, False

        # ── a_escalate (§5.4) ────────────────────────────────────────────
        elif action == 'a_escalate':
            s = dict(s)
            was_escalated = s.get('escalated', False)
            s['escalated'] = True
            next_stage = self._weighted_choice(self.ESCALATE_PROBS[stage])

            if next_stage != stage:
                s = self._apply_stage_transition(s, next_stage)

            evidence = self._generate_evidence_likelihood(s['stage'], max(s['eval_count'], 5))
            s['g'] = RiskClassifier.bayesian_update_g(
                prior_g=s['g'], evidence_likelihood=evidence, noise_scale=0.02)
            old_k = s['k']
            k_idx = self.K_PROGRESSION.index(old_k)
            s['k'] = self.K_PROGRESSION[min(k_idx + 1, 3)]

            # Terminal escalation (already escalated or M4, §5.4)
            if was_escalated or s['stage'] == 'M4':
                ts = self._terminal_state('escalated')
                ts['final_stage'] = s['stage']
                return ts, self.R_ESCALATE_COST, True

            return s, self.R_ESCALATE_COST, False

        # ── a_defer (§5.5) ───────────────────────────────────────────────
        elif action == 'a_defer':
            s = dict(s)
            s['defer_count'] += 1
            next_stage = self._weighted_choice(self.DEFER_PROBS[stage])

            if next_stage != stage:
                s = self._apply_stage_transition(s, next_stage)

            evidence = self._generate_evidence_likelihood(s['stage'], max(s['eval_count'], 2))
            s['g'] = RiskClassifier.bayesian_update_g(
                prior_g=s['g'], evidence_likelihood=evidence, noise_scale=0.06)

            g_array = np.array(s['g'])
            g_conf  = float(np.max(g_array))
            true_tier = self.episode_true_tier
            direction_correct = (true_tier in self.REJECT_CORRECT_TIERS and g_conf < 0.7)
            if direction_correct:
                reward = self.R_STEP + self.R_DEFER_CORRECT
            else:
                reward = self.R_STEP + self.R_DEFER_WRONG
            # Escalating penalty for consecutive defers (discourages defer spam)
            if s['defer_count'] >= 2:
                reward += self.R_DEFER_CONSECUTIVE * (s['defer_count'] - 1)
            return s, reward, False

        # ── a_redact (§5.6) ──────────────────────────────────────────────
        elif action == 'a_redact':
            s = dict(s)
            s['redact_count'] += 1
            next_stage = self._weighted_choice(self.REDACT_PROBS[stage])
            curr_tier = self.STAGE_ORDER[stage]
            next_tier = self.STAGE_ORDER[next_stage]

            if next_stage != stage:
                s = self._apply_stage_transition(s, next_stage)
            else:
                if s['b'] == 'categorisation':
                    s['b'] = 'remote_id'
                elif s['b'] == 'remote_id':
                    s['b'] = 'verification'
                s['g'] = RiskClassifier.compute_initial_g(
                    s['d'], s['b'], s['r'], s['c'], s['m'])

            if next_tier < curr_tier:
                reward = self.R_STEP + self.R_REDACT_SUCCESS - 8
            elif next_tier == curr_tier and s.get('b') != current_state_dict.get('b', ''):
                reward = self.R_STEP - 3
            else:
                reward = self.R_STEP - 14   # redact fail
            return s, reward, False

        # ── a_abstain (§5.9) ─────────────────────────────────────────────
        elif action == 'a_abstain':
            true_tier = self.episode_true_tier
            if true_tier in self.REJECT_CORRECT_TIERS:
                reward = self.R_ABSTAIN_CORRECT
            else:
                reward = self.R_ABSTAIN_WRONG

            ts = self._terminal_state('abstained')
            ts['final_stage'] = stage
            return ts, reward, True

        # ── a_accept (§5.7, §5.9) ────────────────────────────────────────
        elif action == 'a_accept':
            true_tier = self.episode_true_tier
            direction_correct = (true_tier in self.ACCEPT_CORRECT_TIERS)

            g_array = np.array(s['g'])
            g_bonus = self._compute_g_calibration_bonus(g_array, true_tier)
            k_bonus = {'start': 0, 'evidence': 0.05,
                       'proposed': 0.1, 'final': 0.15}[s['k']]
            eval_bonus = (0.1 if 3 <= s['eval_count'] <= 6
                          else -0.1 if s['eval_count'] < 3 else -0.05)
            base_p = self.DECISION_PROBS[stage]['accept']
            p_correct = float(np.clip(
                base_p + k_bonus + eval_bonus + g_bonus
                + (0.1 if s.get('escalated') else 0.0),
                0.05, 0.95))

            reckless = (self.R_RECKLESS
                        if stage in ['M3', 'M4'] and s['eval_count'] < 2 else 0)
            xb_penalty = self._compute_xb_penalty(s)
            # Hefty penalty for deciding without any evidence gathering
            no_evidence_penalty = -30 if s['eval_count'] == 0 else 0
            # Bonus for well-informed terminal decision
            g_conf = float(np.max(np.array(s['g'])))
            well_informed_bonus = 12.0 if (s['eval_count'] >= 3 and g_conf > 0.7) else 0.0

            if direction_correct:
                is_correct = np.random.random() < p_correct
                if is_correct:
                    reward = self.R_CORRECT_ACCEPT
                else:
                    reward = self.R_MISCLASS
            else:
                is_correct = np.random.random() < 0.05
                if is_correct:
                    reward = self.R_CORRECT_ACCEPT
                else:
                    reward = self.R_FALSE_ACCEPT
            reward += reckless + xb_penalty + no_evidence_penalty + well_informed_bonus

            ts = self._terminal_state('verified' if is_correct or direction_correct else 'rejected')
            ts['final_stage'] = stage
            ts['paper_output'] = 'verified'
            return ts, reward, True

        # ── a_reject (§5.7, §5.9) ────────────────────────────────────────
        elif action == 'a_reject':
            true_tier = self.episode_true_tier
            direction_correct = (true_tier in self.REJECT_CORRECT_TIERS)

            g_array = np.array(s['g'])
            g_bonus = self._compute_g_calibration_bonus(g_array, true_tier)
            k_bonus = {'start': 0, 'evidence': 0.05,
                       'proposed': 0.1, 'final': 0.15}[s['k']]
            eval_bonus = (0.1 if 3 <= s['eval_count'] <= 6
                          else -0.1 if s['eval_count'] < 3 else -0.05)
            base_p = self.DECISION_PROBS[stage]['reject']
            p_correct = float(np.clip(
                base_p + k_bonus + eval_bonus + g_bonus
                + (0.1 if s.get('escalated') else 0.0),
                0.05, 0.95))

            reckless = (self.R_RECKLESS
                        if stage in ['M3', 'M4'] and s['eval_count'] < 2 else 0)
            xb_penalty = self._compute_xb_penalty(s)
            no_evidence_penalty = -30 if s['eval_count'] == 0 else 0
            g_conf_rej = float(np.max(np.array(s['g'])))
            well_informed_bonus = 12.0 if (s['eval_count'] >= 3 and g_conf_rej > 0.7) else 0.0

            if direction_correct:
                is_correct = np.random.random() < p_correct
                if is_correct:
                    reward = self.R_CORRECT_REJECT
                else:
                    reward = self.R_MISCLASS
            else:
                is_correct = np.random.random() < 0.05
                if is_correct:
                    reward = self.R_CORRECT_REJECT
                else:
                    reward = self.R_FALSE_REJECT
            reward += reckless + xb_penalty + no_evidence_penalty + well_informed_bonus

            ts = self._terminal_state('rejected')
            ts['final_stage'] = stage
            ts['paper_output'] = 'rejected'
            return ts, reward, True

        return current_state_dict, self.R_STEP, False


# ─── 4. Q-Network ────────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),       nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# ─── 5. DQN Agent ────────────────────────────────────────────────────────────
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim, self.action_dim = state_dim, action_dim
        self.memory     = deque(maxlen=50000)
        self.priorities = deque(maxlen=50000)

        self.gamma              = 0.99
        self.epsilon            = 1.0
        self.epsilon_min        = 0.10
        self.epsilon_decay      = 0.998
        self.batch_size         = 128
        self.target_update_iter = 50
        self.learn_step_counter = 0
        self.priority_alpha     = 0.6
        self.priority_beta      = 0.4

        self.model        = QNetwork(state_dim, action_dim)
        self.target_model = QNetwork(state_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer    = optim.Adam(self.model.parameters(), lr=0.001)

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
        if action_name in ('a_mitig', 'a_verify') and r > 0:
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
        except ValueError:
            return

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
            print(f"[Loaded] {path} (eps={self.epsilon:.4f})")
        else:
            print(f"[Warning] Model not found: {path}, using random weights")


# ─── 6. Training Loop ────────────────────────────────────────────────────────
def train(num_episodes=3000, verbose_episodes=3):
    env   = PipelineMDP()
    agent = DQNAgent(env.state_dim, env.action_dim)
    save_path = "agent_pipeline_v1.pth"
    rewards_history, best_mean = [], -float('inf')

    for e in range(num_episodes):
        state_dict = env.reset()
        state_vec  = env.encoder.encode(state_dict)
        total_r, done, steps = 0.0, False, 0

        if (e + 1) <= verbose_episodes:
            print(f"\n{'='*80}\nEpisode {e+1}\n{'='*80}")

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
            print(f"Ep {e+1} done: total_reward={total_r:.1f}, steps={steps}")

        if (e + 1) >= 100:
            mean100 = np.mean(rewards_history[-100:])
            if mean100 > best_mean:
                best_mean = mean100
                agent.save(save_path)
                print(f"[Best] Ep {e+1}: Mean100={best_mean:.2f}, eps={agent.epsilon:.3f}")

        if (e + 1) % 200 == 0:
            print(f"Ep {e+1}/{num_episodes}: Last100={np.mean(rewards_history[-100:]):.1f}, "
                  f"eps={agent.epsilon:.3f}")

    print(f"\nTraining done. Best Mean100: {best_mean:.2f}")
    return agent


if __name__ == "__main__":
    trained_agent = train(num_episodes=3000, verbose_episodes=3)
