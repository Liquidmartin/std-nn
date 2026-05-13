# 🧠 ML-QCT Project: Neural Networks for Energy Transfer Distributions

This project develops a machine learning pipeline to predict **translational energy distributions** obtained from quasi-classical trajectory (QCT) simulations.

The objective is to replace expensive trajectory calculations with fast and accurate neural network predictions.

---

## 📂 Project Structure

```
.
├── O3/                              # Raw QCT trajectory data
│   ├── v0-j0-E0.5-1/
│   │   └── out
│   ├── v0-j0-E0.5-2/
│   │   └── out
│   └── ...
│
├── scripts_et_dis/                  # Pipeline for P(E'_out) distributions
│   ├── build_et_distributions.py    # Build raw QCT energy distributions
│   ├── smooth_et_distributions.py   # Smooth raw distributions
│   ├── train_nn_et_distribution_smoothed.py
│   │                               # Train NN using smoothed distributions
│   │
│   ├── conditions.txt               # Config file for building distributions
│   ├── conditions_smooth.txt        # Config file for smoothing
│   │
│   ├── et_distributions_exchange/   # Raw QCT P(E'_out) distributions
│   │   ├── et_grid.dat
│   │   ├── et_distributions_index.csv
│   │   ├── distributions/
│   │   └── plots/
│   │
│   ├── et_distributions_exchange_smoothed/
│   │   ├── et_distributions_smoothed_index.csv
│   │   ├── distributions/
│   │   └── comparison_plots/
│   │
│   ├── nn_et_distribution_exchange_smoothed/
│   │   ├── best_model.pt
│   │   ├── scalers.json
│   │   ├── energy_grid.dat
│   │   ├── metrics.json
│   │   ├── pred_test.npy
│   │   ├── true_test.npy
│   │   ├── test_sample_ids.npy
│   │   ├── loss_curve.png
│   │   ├── test_examples.png
│   │   ├── parity_mean.png
│   │   └── parity_std.png
│   │
│   └── visualization/
│       └── scripts for plotting raw/smoothed distributions
│
└── README.md
```
---

## 🎯 Objective

Learn the mapping:

(E_in, v_in, j_in) → P(E_out)

where:

- E_in: initial translational energy  
- v_in, j_in: initial rovibrational quantum numbers  
- P(E_out): probability distribution of final translational energy  

---

## ⚙️ Workflow

QCT simulations → Energy distributions → Filtering → Neural Network training → Prediction

---

## 🔧 Main Scripts

### build_et_distributions.py
- Reads QCT outputs (rr*/out)
- Extracts translational energies
- Builds normalized distributions P(E_out)

---

### filtar_data_set.py
- Removes low-quality samples
- Filters by selected trajectories

---

### train_nn_et_distribution.py
- Learns:

  (E_in, v_in, j_in) → P(E_out)

- Uses:
  - Softmax output
  - KL divergence loss

---

## 🚀 Usage

1. Build distributions

python build_et_distributions.py

2. Build smooth distributions

python smooth_et_distributions.py

3. Train models

python train_nn_et_distribution_smoothed.py

---

## 📊 Results

Distribution model:
- Correct peak position
- Good width prediction
- Slight smoothing of sharp peaks

---

## 🧬 Requirements

- Python 3.9+
- numpy
- pandas
- matplotlib
- scikit-learn
- torch

Install with:

pip install numpy pandas matplotlib scikit-learn torch

---

## 📁 Data

Raw QCT data is not included due to size.

Expected structure:

```
O3/
 ├── rr001/
 │   └── out
 ├── rr002/
 │   └── out
```
---

## 🧩 Future Work


- Extension to other systems

---

## 🧑‍💻 Author

Raidel Martin-Barrios  
https://github.com/Liquidmartin

---

## 📄 Acknowledgments

This work was developed within a research environment focused on molecular dynamics and machine learning, using QCT simulations and neural network models form Department of Chemistry
University of Basel of Prof. Dr. Markus Meuwly groupe.
