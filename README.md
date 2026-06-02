# CrossBorder AI Governance — Pipeline MDP Framework

A Markov Decision Process framework for governing AI systems through a sequential pipeline of four progressively restrictive governance stages, balancing semantic expressivity against uncertainty.

Published in *Nature Communications* (2026): *"Markov Decision Processes for Governance of Artificial Intelligence"*  
Authors: Minxian Xu, Kan Hu, Vlado Stankovski, Petar Kochovski

---

## Overview

The framework routes each decision request through four governance stages:

| Stage | Semantic Focus | Inference Characteristics |
|-------|---------------|--------------------------|
| **M1** | General contextual reasoning | Broad exploratory inference |
| **M2** | Evidence verification | Consistency-driven low-risk inference |
| **M3** | Multimodal biometric fusion | Cross-modal probabilistic reasoning |
| **M4** | Regulatory governance assurance | Conservative high-confidence validation |

Each stage applies 10 possible governance actions (evaluate, mitigate, verify, escalate, defer, redact, abstain, accept, reject) under hard admissible-action constraints, producing one of five terminal outputs: **verified**, **rejected**, **escalated**, **abstained**, or **uncertain**.

---

## Repository Structure

```
.
├── Final_version/
│   ├── cross_pipeline_mdp_v1.py          # Training environment (Pipeline MDP + DQN agent)
│   ├── cross_analysis_pipeline_split.py  # Figure generation (40 individual panels)
│   ├── results_pipeline_v1.csv           # 33,331-episode simulation results (40 columns)
│   └── transition_pipeline_v1.csv        # Aggregated state-action-transition counts
├── figure_show/                          # Generated figure panels (PDF + PNG)
├── main.tex                              # LaTeX manuscript
├── main.pdf                              # Compiled manuscript
└── README.md
```

---

## Key Features

- **Bayesian belief propagation**: Probabilistic belief vector over latent regulatory risk tiers, updated via Bayes' rule
- **Hard governance constraints**: Deterministic admissible-action filters (stage-restricted actions, resource budgets)
- **Stage reversibility**: Bidirectional transitions (mitigation can downgrade risk tier)
- **DQN agent**: Deep Q-Network learning Q(s,a) over 22-dim augmented state space
- **33,331 simulated episodes**: 475,139 governance steps, 441,808 non-entry state transitions

---

## Reproduction

### Requirements

```bash
pip install torch numpy pandas scipy matplotlib seaborn networkx scikit-learn
```

### Train the DQN Agent

```bash
cd Final_version
python cross_pipeline_mdp_v1.py
```

### Generate Figures

```bash
cd Final_version
python cross_analysis_pipeline_split.py results_pipeline_v1.csv transition_pipeline_v1.csv ../figure_show
```

### Compile the Paper

```bash
pdflatex main.tex
pdflatex main.tex  # second pass for cross-references
```

---

## License

MIT License. See repository for details.

## Citation

```bibtex
@article{xu2026mdp,
  title={Markov Decision Processes for Governance of Artificial Intelligence},
  author={Xu, Minxian and Hu, Kan and Stankovski, Vlado and Kochovski, Petar},
  journal={Nature Communications},
  year={2026}
}
```
