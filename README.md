# Cross-Attention Multimodal Emotion Recognition (PPG + Face)

This repository includes my Master’s thesis work on **multimodal emotion recognition** using:

- Behind-the-ear **PPG** signals (converted to 2D CWT images)
- **Facial expression** images

A **cross-attention deep learning model** is used to fuse both modalities for improved emotion classification performance.

This project is developed at the  
Department of Artificial Intelligence Convergence,  
**Pukyong National University (PKNU), South Korea**

---

## 🚀 Key Features

- Dual-branch CNN for PPG and facial modalities  
- Cross-Attention fusion for feature alignment  
- Subject-independent evaluation using LOSO  
- Python & PyTorch-based modular code  
- Suitable for wearable affective computing research  

---

## 📂 Repository Structure

- `src/` – Python scripts (model, dataloader, training)
- `results/` – Confusion matrices and plots (TBA)
- `notebooks/` – Optional experimental notebooks
- `README.md` – Project documentation
- `requirements.txt` – Dependencies list

> More files will be uploaded and updated soon.

---

## 📦 Requirements

- Python 3.8+
- PyTorch
- NumPy / Pandas / SciPy
- scikit-learn
- OpenCV (for facial images)
- Matplotlib / Seaborn (visualization)

> Install dependencies:  
```bash
pip install -r requirements.txt
