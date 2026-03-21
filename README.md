# 🛳 Ship Classification from ISAR Images

<div align="center">

![DRDO](https://img.shields.io/badge/DRDO-LRDE%20Division-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-orange?style=for-the-badge&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-2.3%2B-green?style=for-the-badge&logo=flask)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**A CNN-based ship classification system for Inverse Synthetic Aperture Radar (ISAR) imagery,  
developed during internship at LRDE Division, DRDO.**

*Dissertation Project — M.Sc. Data Science & Analytics, Jain University (2023–2025)*

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [Architecture](#-architecture)
- [Ship Classes](#-ship-classes)
- [Dataset Structure](#-dataset-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Acknowledgements](#-acknowledgements)

---

## 🔭 Overview

This project addresses the challenge of **automating naval vessel identification** using deep learning on ISAR radar imagery. Traditional ship classification relies on human interpretation of radar data — a slow and error-prone process. This system replaces it with a **Convolutional Neural Network (CNN)** pipeline deployed as a Flask web application.

**Key capabilities:**
- Upload a ZIP dataset of ISAR ship images → auto-train a CNN model
- Real-time ship-type prediction on new images
- Visual evaluation: confusion matrix, accuracy/loss curves, classification report
- Supports 7 ship categories out of the box

**Applications:** Naval surveillance · Maritime traffic management · Piracy prevention · Port monitoring · Environmental compliance tracking

---

## 🎬 Demo

| Page | Description |
|------|-------------|
| `/`  | Upload a `.zip` dataset and trigger training |
| `/train` | View training metrics, confusion matrix, classification report |
| `/predict` | Upload a single ISAR image to classify it in real time |

---

## 🧠 Architecture

The CNN model follows this pipeline:

```
Input Image (200×500, Grayscale)
        │
        ▼
┌─────────────────────────┐
│  Conv2D(2)  + ReLU      │  ← Low-level edge & texture detection
│  MaxPool2D(2×2)         │
├─────────────────────────┤
│  Conv2D(4)  + ReLU      │  ← Intermediate structural features
│  MaxPool2D(2×2)         │
├─────────────────────────┤
│  Conv2D(8)  + ReLU      │  ← High-level ship-type features
│  MaxPool2D(2×2)         │
├─────────────────────────┤
│  Flatten                │
│  Dense(64) + ReLU       │
│  Dropout(0.1)           │
│  Dense(7)  + Softmax    │  ← 7-class probability output
└─────────────────────────┘
        │
        ▼
  Predicted Ship Class
```

**Optimizer:** Adagrad (lr=0.1)  
**Loss:** Categorical Cross-Entropy  
**Input:** 200 × 500 grayscale images  
**Augmentation:** Horizontal flip · Vertical flip · Rescale [0,1]  
**Train/Val Split:** 80% / 20%

---

## 🚢 Ship Classes

| # | Class | Description |
|---|-------|-------------|
| 1 | **Cargo** | Commercial freight vessels |
| 2 | **Aircraft Carrier** | Naval aviation warships |
| 3 | **Passenger** | Cruise / ferry vessels |
| 4 | **Oil Tanker** | Petroleum transport ships |
| 5 | **Frigate** | Naval escort warships |
| 6 | **Patrol** | Coast guard / patrol boats |
| 7 | **Other** | Miscellaneous vessel types |

---

## 📁 Dataset Structure

Your ZIP file must follow this directory layout:

```
dataset.zip
└── dataset/
    ├── cargo/
    │   ├── cargo_001.png
    │   └── ...
    ├── aircraft_carrier/
    │   └── ...
    ├── passenger/
    │   └── ...
    ├── oil_tanker/
    │   └── ...
    ├── frigate/
    │   └── ...
    ├── patrol/
    │   └── ...
    └── other/
        └── ...
```

> **Image requirements:** PNG or JPG, any resolution (will be auto-resized to 200×500).  
> Grayscale ISAR images recommended. Colour images are auto-converted to grayscale.

---

## ⚙️ Installation

### Prerequisites
- Python 3.9 or higher
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/ship-isar-classification.git
cd ship-isar-classification

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Flask app
python app.py
```

The app will be available at **http://127.0.0.1:5000**

---

## 🚀 Usage

### Step 1 — Prepare Dataset
Organise your ISAR images into class-labelled folders and ZIP them as shown in [Dataset Structure](#-dataset-structure).

### Step 2 — Upload & Train
1. Open the web app at `http://127.0.0.1:5000`
2. Click **Upload & Begin Training** and select your ZIP file
3. The system will extract the dataset, train the CNN, and redirect to the results page

### Step 3 — View Results
The training results page shows:
- Validation accuracy
- Training / validation accuracy & loss curves
- Confusion matrix heatmap
- Full classification report (precision, recall, F1-score per class)

### Step 4 — Predict
1. Navigate to `/predict`
2. Upload a single ISAR image
3. View the predicted ship type + per-class confidence scores

---

## 📊 Results

> Results below are representative of a well-balanced 7-class ISAR dataset.

| Metric | Value |
|--------|-------|
| Validation Accuracy | ~85–92% (dataset dependent) |
| Loss Function | Categorical Cross-Entropy |
| Optimizer | Adagrad |
| Epochs | 30 |

**Observations:**
- CNN significantly outperforms traditional hand-crafted feature methods
- Augmentation (flip) improves generalisation across viewing angles
- Noise in ISAR data causes minor misclassification between visually similar classes (e.g., Frigate vs Patrol)

---

## 📂 Project Structure

```
ship-isar-classification/
│
├── app.py                   # Main Flask application (routes, model, training)
├── requirements.txt         # Python dependencies
├── .gitignore
├── README.md
│
├── templates/               # Jinja2 HTML templates
│   ├── index.html           # Home / dataset upload page
│   ├── train.html           # Training results page
│   └── predict.html         # Prediction page
│
├── static/
│   ├── styles.css           # Global stylesheet (dark radar-terminal theme)
│   ├── logo.png             # (add your logo here)
│   ├── uploads/             # Temp storage for prediction images (git-ignored)
│   ├── confusion_matrix.png # Generated after training (git-ignored)
│   └── training_curves.png  # Generated after training (git-ignored)
│
├── uploaded_dataset/        # Extracted dataset (git-ignored)
└── trained_model.h5         # Saved Keras model (git-ignored / use Releases)
```

---

## 🛠 Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3.9+** | Core language |
| **TensorFlow / Keras** | CNN model building & training |
| **Flask** | Web framework |
| **OpenCV** | Image preprocessing |
| **NumPy** | Numerical computations |
| **Matplotlib / Seaborn** | Visualisation |
| **scikit-learn** | Evaluation metrics (confusion matrix, classification report) |
| **HTML5 / CSS3** | Frontend UI |

---

## 🔮 Future Work

- [ ] Integrate transfer learning (ResNet-50 / EfficientNet) for higher accuracy
- [ ] Add real-time video stream classification
- [ ] Multi-sensor fusion (ISAR + optical imagery)
- [ ] Deploy on cloud (AWS / GCP) with Docker
- [ ] Extend to AIS (Automatic Identification System) data fusion
- [ ] Add Grad-CAM visualisation for model interpretability

---

## 👤 Author

**Vijay N** · `23MSRDS078`  
M.Sc. Data Science & Analytics (2023–2025)  
Jain (Deemed-to-be University), Bangalore

**Internship Guide:** Dr. Dyana A, Scientist — LRDE Division, DRDO  
**Academic Mentor:** Dr. Ghouse Basha, Jain University

---

## 🙏 Acknowledgements

- **LRDE Division, DRDO** — for providing the internship opportunity and domain expertise
- **Dr. Dyana A** — for guidance on ISAR imaging and radar signal processing
- **Dr. Ghouse Basha** — for academic mentorship and project support
- **Dr. Asha Rajiv** — Director, School of Sciences, Jain University
- **Dr. Arathi Sudarshan** — Head of Department, Data Analytics & Mathematical Sciences

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
<sub>Built with ☕ during DRDO LRDE Internship, Bangalore · 2025</sub>
</div>
