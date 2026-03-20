# Neural Network Modeling of Energy Transfer Distributions (QCT)

This project develops a machine learning pipeline to predict **translational energy distributions** from quasi-classical trajectory (QCT) simulations.

---

## 🎯 Objective

Learn the mapping:

(E_in, v_in, j_in) → P(E_out)

where:

- **E_in**: initial translational energy  
- **v_in, j_in**: initial rovibrational quantum numbers  
- **P(E_out)**: probability distribution of final translational energy  

---

## ⚙️ Workflow

The pipeline consists of four main stages:

### 1. Build energy distributions from QCT trajectories

```bash
python build_et_distributions.py
