# 👁️ Dry Eye Disease Detection using Deep Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

**A CNN-based deep learning model that detects Dry Eye Disease (DED) from retinal images with 92.6% accuracy.**

</div>

---

## 📌 Problem Statement

Dry Eye Disease affects over **344 million people** worldwide and is often misdiagnosed or detected too late. Traditional diagnosis requires specialist visits and expensive equipment. This project uses Convolutional Neural Networks (CNNs) to automatically classify retinal images — enabling faster, cheaper, and more accessible diagnosis.

---

## 🎯 Results

| Metric | Score |
|--------|-------|
| ✅ Accuracy | **92.6%** |
| 📊 Precision | 91.3% |
| 📈 Recall | 93.1% |
| 🔷 F1 Score | 92.2% |
| 📉 Improvement over baseline | **+8.4%** |

---

## 🧠 Model Architecture

```
Input Image (224x224x3)
        ↓
   [Conv2D + ReLU]  → 32 filters
        ↓
   [MaxPooling2D]
        ↓
   [Conv2D + ReLU]  → 64 filters
        ↓
   [MaxPooling2D]
        ↓
   [Conv2D + ReLU]  → 128 filters
        ↓
   [MaxPooling2D]
        ↓
   [Flatten]
        ↓
   [Dense 512 + Dropout 0.5]
        ↓
   [Output: Sigmoid] → DED / Normal
```

---

## 🔬 Key Techniques

- **CLAHE (Contrast Limited Adaptive Histogram Equalization)** — enhanced retinal image contrast
- **Image Normalization** — standardized pixel values for faster convergence
- **Data Augmentation** — rotation, flipping, zoom to prevent overfitting
- **Transfer Learning** — fine-tuned pre-trained CNN weights

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.8+ |
| Deep Learning | TensorFlow 2.x, Keras |
| Image Processing | OpenCV, PIL |
| Data Handling | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Preprocessing | CLAHE, Image Normalization |

---

## 📂 Project Structure

```
Dry-Eye-Disease-Prediction/
│
├── 📓 dry_eye_detection.ipynb    # Model training notebook
├── 🔬 preprocessing.py           # Image preprocessing with CLAHE
├── 🧠 model/                     # Saved CNN model
│   └── ded_model.h5
├── 📊 data/                      # Dataset directory
│   ├── normal/
│   └── dry_eye/
├── 📈 evaluation.py              # Model evaluation scripts
├── 📋 requirements.txt
└── 📖 README.md
```

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/shiav321/Dry-Eye-Disease-Prediction.git
cd Dry-Eye-Disease-Prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run preprocessing
python preprocessing.py

# 4. Train the model
jupyter notebook dry_eye_detection.ipynb
```

---

## 🌍 Real-World Impact

- Enables **early detection** of DED before it causes permanent damage
- Can be deployed in **rural clinics** with basic smartphone cameras
- Reduces dependency on expensive ophthalmology specialists
- Scalable to other retinal diseases with transfer learning

---

## 👨‍💻 About the Developer

**Shiva Keshava** — B.Tech AI & Data Science Graduate

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin)](https://linkedin.com/in/shiva-keshava-b71355364)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-FF6B6B?style=flat&logo=google-chrome)](https://shivaprofilewebsite.lovable.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github)](https://github.com/shiav321)

---

<div align="center">
⭐ If this project helped you, please star it — it motivates further development!
</div>

