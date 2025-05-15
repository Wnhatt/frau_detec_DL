# 🛡️ Adversarial ML Toolkit

This repository provides implementations of various **adversarial attacks** and their corresponding **defense mechanisms** on machine learning models. It is structured for easy experimentation and modular evaluation.

---

## 📌 Implemented Attacks & Defenses

| Attack Type        | Description                             | Defense Mechanism                         |
|--------------------|-----------------------------------------|-------------------------------------------|
| **FGSM Attack**     | Fast Gradient Sign Method               | Adversarial Training                      |
| **FALFA Attack**    | Feature Alignment-based Label Flipping  | kNN-based Defense                         |
| **Backdoor Attack** | Universal Trigger-based Backdoor Attack | Spectral Signature & Pruning-based Defense|

---

## 🧪 How to Use

### 🔍 Run Evaluation Notebook

To evaluate all attack and defense methods in one place:

```bash
# Open the notebook in JupyterLab, VS Code, or Google Colab
evaluation.ipynb
```

This notebook includes visualizations, performance metrics, and comparisons of model accuracy before and after applying defenses against attacks.

---

### 🧪 Run Individual Pipelines

Each attack-defense experiment is organized into separate Python scripts under the `testing/` directory.

#### Example usage:

```bash
cd testing/
```

#### Available Pipelines:

```bash
# Run FGSM Attack with Adversarial Training Defense
python test_fgsm_attack_defense.py

# Run FALFA Attack with kNN-based Defense
python test_knn_defense.py

# Run Backdoor Attack with Spectral Signature + Pruning Defense
python test_pruning.py | python test_ss.py
```

---

## 📂 Project Structure

```
.
├── evaluation.ipynb                # Main evaluation notebook
├── testing/                        # Pipelines for attacks & defenses
│   ├── test_fgsm_attack_defense.py
│   ├── test_knn_defense.py
│   ├── test_pruning.py
│   └── ...
├── models/                         # Neural network model definitions
├── data/                           # Sample dataset (or link to external source)
├── utils/                          # Utility functions for attacks/defenses
└── README.md                       # Project overview
```

---

## 📚 References

- **FGSM** – Fast Gradient Sign Method: [1Konny/FGSM](https://github.com/1Konny/FGSM)
- **Backdoor Attack & Defense** – BATD: [HamidRezaTajalli/BATD](https://github.com/HamidRezaTajalli/BATD)
- **Spectral Signature Defense** – *Detecting Backdoor Attacks on Deep Neural Networks by Activation Clustering*, Tran et al., 2018


---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. Any use of adversarial attacks for malicious purposes is strictly discouraged.

---
