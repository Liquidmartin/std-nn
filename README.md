# 🧠 ML-QCT Project: Neural Networks for Energy Transfer Distributions

This project develops a machine learning pipeline to predict **translational energy distributions** obtained from quasi-classical trajectory (QCT) simulations.

The objective is to replace expensive trajectory calculations with fast and accurate neural network predictions.

---

## 📂 Project Structure

```
.
├── O3/                         # Raw QCT data (rr*/out files)
│
├── et_distributions/           # Generated energy distributions
│   ├── distributions/          # Individual P(E_out) files
│   └── et_distributions_index.csv
│
├── nn_et_moments/              # Trained model (mean & std)
├── nn_et_distribution/         # Trained model (full distributions)
│
├── build_et_distributions.py
├── build_moments_dataset.py
├── filtar_data_set.py
├── train_nn_et_moments.py
├── train_nn_et_distribution.py
├── predict_et_moments.py
│
├── conditions.txt
├── com_dist.ipynb
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

QCT simulations → Energy distributions → Moments → Filtering → Neural Network training → Prediction

---

## 🔧 Main Scripts

### build_et_distributions.py
- Reads QCT outputs (rr*/out)
- Extracts translational energies
- Builds normalized distributions P(E_out)

---

### build_moments_dataset.py
- Computes:
  - Mean energy ⟨E_out⟩  
  - Standard deviation σ(E_out)

---

### filtar_data_set.py
- Removes low-quality samples
- Filters by number of reactive trajectories

---

### train_nn_et_moments.py
- Learns:

  (E_in, v_in, j_in) → (mean_Eout, std_Eout)

- Fast and stable baseline model

---

### train_nn_et_distribution.py
- Learns:

  (E_in, v_in, j_in) → P(E_out)

- Uses:
  - Softmax output
  - KL divergence loss

---

### predict_et_moments.py
- Loads trained model
- Predicts mean and standard deviation

---

## 🚀 Usage

1. Build distributions

python build_et_distributions.py

2. Compute moments

python build_moments_dataset.py

3. Filter dataset

python filtar_data_set.py

4. Train models

python train_nn_et_moments.py
python train_nn_et_distribution.py

5. Predict

python predict_et_moments.py

---

## 📊 Results

Moments model:
- Near-perfect prediction of mean energy
- Accurate prediction of standard deviation

Distribution model:
- Correct peak position
- Good width prediction
- Slight smoothing of sharp peaks

---

## ⚠️ Key Insight

MSE loss produces averaged, non-physical distributions.

KL divergence correctly captures peak structure and produces physically meaningful predictions.

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

- Hybrid models (moments + distributions)
- Improved peak resolution
- Physics-informed neural networks
- Extension to other systems

---

## 🧑‍💻 Author

Raidel Martin-Barrios  
https://github.com/Liquidmartin

---

## 📄 Acknowledgments

This work was developed within a research environment focused on molecular dynamics and machine learning, using QCT simulations and neural network models.
