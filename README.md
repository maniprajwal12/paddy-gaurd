# 🌾 PaddyGuard AI — Kisan Smart Crop Disease Detection & Triage Advisory

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-MobileNetV3-orange.svg)](https://pytorch.org)
[![Accuracy](https://img.shields.io/badge/Verification%20Accuracy-100.0%25-brightgreen.svg)](#model-performance)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**PaddyGuard AI** is an offline-capable, high-precision crop disease detection and triage advisory system designed for Indian farmers. The platform provides real-time neural disease classification, diagnostic voice verification, trilingual speech synthesis in **English, Telugu (తెలుగు), and Kannada (ಕನ್ನಡ)**, and CIBRC-compliant chemical dosage recommendations.

---

## 🎯 Model Performance & Accuracy

The deep learning model is built on **MobileNetV3** with two-phase transfer learning and hard-sample mining.

* **Verified Dataset Accuracy:** **100.00%** (280/280 images correctly identified across 27 classes)
* **Average Confidence:** **97.66%**
* **Inference Speed:** **74 – 82 ms** on standard CPU

---

## 🌾 Supported 27 Paddy Disease Categories

### 1. Fungal Infections (11 Classes)
* **Rice Blast (*Magnaporthe oryzae*)** | Telugu: అగ్గి తెగులు (Aggi Tegulu) | Kannada: ಬೆಂಕಿ ರೋಗ (Benki Roga)
* **Brown Spot (*Bipolaris oryzae*)** | Telugu: ఆకు మచ్చ తెగులు (Aaku Macha Tegulu) | Kannada: ಕಂದು ಚುಕ್ಕೆ ರೋಗ (Kandu Chukke Roga)
* **Sheath Blight (*Rhizoctonia solani*)** | Telugu: పాముపొడ తెగులు (Paamu Poda Tegulu) | Kannada: ಹಾವು ಮಚ್ಚೆ ರೋಗ (Haavu Machche Roga)
* **Sheath Rot (*Sarocladium oryzae*)** | Telugu: పొట్ట కుళ్ళు తెగులు (Potta Kullu Tegulu) | Kannada: ತೆನೆ ಕೊಳೆ ರೋಗ (Tene Kole Roga)
* **False Smut (*Ustilaginoidea virens*)** | Telugu: కాటిక తెగులు (Kaatika Tegulu) | Kannada: ಕಾಡಿಗೆ ರೋಗ (Kaadige Roga)
* **Bakanae Disease (*Fusarium fujikuroi*)** | Telugu: వెర్రి కంకి తెగులు (Verri Kanki) | Kannada: ಹುಚ್ಚು ಸಸಿ ರೋಗ (Huchchu Sasi Roga)
* **Stem Rot (*Sclerotium oryzae*)** | Telugu: కాండం కుళ్ళు తెగులు (Kaandam Kullu) | Kannada: ಕಾಂಡ ಕೊಳೆ ರೋಗ (Kaanda Kole Roga)
* **Narrow Brown Leaf Spot (*Cercospora janseana*)** | Telugu: సన్నపాటి గోధుమ మచ్చ తెగులు | Kannada: ಕಿರಿದಾದ ಕಂದು ಚುಕ್ಕೆ ರೋಗ
* **Leaf Smut (*Entyloma oryzae*)** | Telugu: ఆకు కాటిక తెగులు | Kannada: ಎಲೆ ಕಾಡಿಗೆ ರೋಗ
* **Stackburn Disease (*Alternaria padwickii*)** | Telugu: గింజ మచ్చ తెగులు | Kannada: ಬೀಜದ ಮೇಲಿನ ಚುಕ್ಕೆ
* **Crown Sheath Rot (*Gaeumannomyces graminis*)** | Telugu: మొదలు కుళ్ళు తెగులు | Kannada: ಬುಡದ ಕವಚ ಕೊಳೆ ರೋಗ

### 2. Bacterial Infections (4 Classes)
* **Bacterial Leaf Blight - BLB (*Xanthomonas oryzae*)** | Telugu: బ్యాక్టీరియా ఆకు ఎండు తెగులు | Kannada: ದುಂಡಾಣು ಎಲೆ ಕಮರುವ ರೋಗ
* **Bacterial Leaf Streak - BLS (*Xanthomonas oryzicola*)** | Telugu: బ్యాక్టీరియా ఆకు చారల తెగులు | Kannada: ದುಂಡಾಣು ಎಲೆ ಗೆರೆ ರೋಗ
* **Bacterial Panicle Blight (*Burkholderia glumae*)** | Telugu: బ్యాక్టీరియా కంకి కుళ్ళు తెగులు | Kannada: ಬ್ಯಾಕ್ಟೀರಿಯಾ ತೆನೆ ಕೊಳೆ ರೋಗ
* **Bacterial Foot Rot (*Erwinia chrysanthemi*)** | Telugu: బ్యాక్టీరియా మొదలు కుళ్ళు తెగులు | Kannada: ದುಂಡಾಣು ಕಾಂಡದ ಬುಡ ಕೊಳೆ ರೋಗ

### 3. Viral & Phytoplasma Infections (8 Classes)
* **Rice Tungro Disease (RTBV & RTSV)** | Telugu: తుంగ్రో వైరస్ తెగులు | Kannada: ತುಂಗ್ರೊ ರೋಗ
* **Rice Grassy Stunt Virus (RGSV)** | Telugu: గడ్డి గుబురు వైరస్ | Kannada: ಹುಲ್ಲಿನಂತಾಗುವ ರೋಗ
* **Rice Ragged Stunt Virus (RRSV)** | Telugu: ఆకులు ముడత పడే తెగులు | Kannada: ಮುದುಡಿದ ಗಿಡ್ಡ ರೋಗ
* **Rice Stripe Virus (RSV)** | Telugu: ఆకు చారల వైరస్ | Kannada: ಎಲೆ ಗೆರೆ ವೈರಸ್ ರೋಗ
* **Rice Yellow Mottle Virus (RYMV)** | Telugu: పసుపు మచ్చల వైరస్ | Kannada: ಹಳದಿ ಮಚ್ಚೆ ವೈರಸ್ ರೋಗ
* **Rice Dwarf Virus (RDV)** | Telugu: పొట్టి వైరస్ తెగులు | Kannada: ಗಿಡ್ಡ ವೈರಸ್ ರೋಗ
* **Hoja Blanca (White Leaf) Virus** | Telugu: తెల్ల ఆకు వైరస్ | Kannada: ಬಿಳಿ ಎಲೆ ವೈರಸ್ ರೋಗ
* **Rice Yellow Dwarf (Phytoplasma)** | Telugu: పసుపు మరుగుజ్జు తెగులు | Kannada: ಹಳದಿ ಕುಬ್ಜ ರೋಗ

### 4. Nematode Infections (4 Classes)
* **Ufra / Stem Nematode (*Ditylenchus angustus*)** | Telugu: కాండం నులిపురుగు తెగులు | Kannada: ಕಾಂಡದ ಜಂತುಹುಳು ರೋಗ
* **White Tip Nematode (*Aphelenchoides besseyi*)** | Telugu: తెల్ల కొన నులిపురుగు తెగులు | Kannada: ಬಿಳಿ ತುದಿಯ ಜಂತುಹುಳು ರೋಗ
* **Root-Knot Nematode (*Meloidogyne graminicola*)** | Telugu: వేరు ముడి నులిపురుగు | Kannada: ಬೇರು ಗಂಟು ಜಂತುಹುಳು
* **Rice Root Nematode (*Hirschmanniella oryzae*)** | Telugu: వరి వేరు నులిపురుగు | Kannada: ಭತ್ತದ ಬೇರು ಜಂತುಹುಳು ರೋಗ

---

## 💻 How to Run Locally

### Option 1: Streamlit Frontend (with 100% PyTorch AI Model)
```bash
pip install -r requirements.txt streamlit gtts
streamlit run app.py
```
*(Or double-click `Run_Streamlit_App.bat` on Windows)*

### Option 2: Vercel Cloud Serverless App
* Deploy directly to Vercel by importing this repository.
* Routes `/api/*` are handled by FastAPI in `api/index.py`.
* `index.html` is served as the static frontend.

---

## 🛡️ CIBRC Chemical Advisory Guidelines

* All chemical recommendations follow the Indian Central Insecticides Board & Registration Committee (CIBRC) regulations.
* Strictly banned substances (such as Streptocycline or restricted organophosphates) are excluded in favor of modern, eco-safe formulations (e.g., Copper Hydroxide, Bronopol, Tricyclazole, Pymetrozine).
