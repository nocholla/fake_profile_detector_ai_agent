# 🚩 Fake Profile Detector AI Agent

**Fake Profile Detector** is an AI-powered tool designed for **Africa Love Match** to identify fake user profiles on dating platforms. Built with machine learning and deep learning models, it analyzes user profile data (age, country, subscription status, relationship goals, and bio) to detect anomalies and predict fake profiles with high accuracy. The system integrates **Streamlit** for an interactive UI and supports multiple models including **Random Forest**, **XGBoost**, **Isolation Forest**, and a custom **AnomalyNet**.

---

## 📚 Table of Contents

1. [Features](#-features)
2. [Tech Stack](#-tech-stack)
3. [Project Structure](#-project-structure)
4. [Installation](#-installation)
5. [Usage](#-usage)
6. [Training Models](#-training-models)
7. [Configuration](#-configuration)
8. [Contributing](#-contributing)
9. [License](#-license)
10. [Screenshots](#-screenshots)

---

## ✨ Features

* **Multi-Model Detection** – Combines Random Forest, XGBoost, Isolation Forest, and AnomalyNet for robust fake profile detection.
* **Profile Analysis** – Evaluates features like age, country, subscription status, relationship goals, and bio text.
* **RAG Integration** – Uses Sentence Transformers for semantic similarity search to identify similar profiles.
* **Streamlit UI** – Interactive interface for inputting profile details and viewing predictions.
* **Model Persistence** – Saves trained models, encoders, and embeddings for reuse.
* **Detailed Predictions** – Provides confidence scores and reasons for fake profile classifications.
* **Logging** – Comprehensive logging for debugging and monitoring.
* **Configurable** – Easily adjustable via a YAML configuration file.

---

## 🖥 Tech Stack

**Machine Learning & Deep Learning**

* [Scikit-learn](https://scikit-learn.org/) – Random Forest, Isolation Forest, and preprocessing utilities
* [XGBoost](https://xgboost.readthedocs.io/) – Gradient boosting for classification
* [PyOD](https://pyod.readthedocs.io/) – Anomaly detection with Isolation Forest
* [PyTorch](https://pytorch.org/) – Custom AnomalyNet neural network
* [Sentence Transformers](https://www.sbert.net/) – Semantic embeddings for RAG

**Data Processing**

* [Pandas](https://pandas.pydata.org/) – Data manipulation and storage
* [NumPy](https://numpy.org/) – Numerical computations
* [SciPy](https://scipy.org/) – Sparse matrix handling
* [Joblib](https://joblib.readthedocs.io/) – Model persistence
* [PyYAML](https://pyyaml.org/) – Configuration file parsing

**Frontend**

* [Streamlit](https://streamlit.io/) – Interactive UI for profile input and prediction display

**Utilities**

* Python Logging – Detailed logging for debugging
* [Argparse](https://docs.python.org/3/library/argparse.html) – Command-line argument parsing

---

## 📂 Project Structure

```
fake_profile_detector_ai_agent/
├── config.yaml               # Configuration file
├── requirements.txt          # Python dependencies
├── .gitignore                # Ignores data, models, and virtual env
├── data/                     # Data storage
│   ├── Profiles.csv          # User profile data
│   ├── BlockedUsers.csv      # Blocked user data
│   ├── DeclinedUsers.csv     # Declined user data
│   ├── DeletedUsers.csv      # Deleted user data
│   ├── ReportedUsers.csv     # Reported user data
│   └── fake_profile_predictions.csv # Prediction outputs
├── models/                   # Model storage
│   ├── rf_model.pkl          # Random Forest model
│   ├── xgb_model.pkl         # XGBoost model
│   ├── iforest_model.pkl     # Isolation Forest model
│   ├── anomaly_net.pth       # AnomalyNet model
│   ├── tfidf_vectorizer.pkl  # TF-IDF vectorizer
│   ├── label_encoders.pkl    # Label encoders
│   ├── scaler.pkl            # Standard scaler
│   └── profile_embeddings.pkl # RAG embeddings
├── src/                      # Source code
│   ├── __init__.py
│   ├── load_data.py          # Data loading logic
│   ├── preprocess_data.py    # Data preprocessing
│   ├── train_anomaly_net.py  # AnomalyNet training
│   └── run.py                # Main script for training/prediction
├── ui/                       # Streamlit UI
│   └── streamlit_app.py      # Streamlit UI script
└── README.md                 # Documentation
```

---

## 🛠 Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-repo/fake_profile_detector_ai_agent.git
cd fake_profile_detector_ai_agent
```

**2. Set up a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure settings**

* Create or update `config.yaml`:

```yaml
data_dir: "data"
model_dir: "models"
features:
  - age
  - country
  - subscribed
  - relationshipGoals
  - aboutMe
tfidf_max_features: 100
anomaly_epochs: 100
contamination: 0.1
```

* Ensure the `data/` directory contains required CSV files (`Profiles.csv`, etc.).

**5. Run locally**

To run the Streamlit UI:

```bash
streamlit run ui/streamlit_app.py
```

To train models or make predictions via CLI:

```bash
python src/run.py --mode train
# or
python src/run.py --mode predict --age 25 --country "Kenya" --about_me "Looking for love" --relationship_goals "Long-term" --subscribed
```

---

## 🚀 Usage

### Streamlit UI
1. Open the Streamlit app at **[http://localhost:8501](http://localhost:8501)**.
2. Enter profile details (age, country, about me, relationship goals, subscription status).
3. Submit to receive a prediction (Real/Fake), confidence score, reasons, and similar profile indices.
4. Predictions are saved to `data/fake_profile_predictions.csv`.

### Command Line
- **Train models**:
  ```bash
  python src/run.py --mode train
  ```
- **Predict for a profile**:
  ```bash
  python src/run.py --mode predict --age 25 --country "Kenya" --about_me "Looking for love" --relationship_goals "Long-term" --subscribed
  ```

**Example Output**:
```
Prediction: Fake
Confidence: 0.62
Reasons:
- High Random Forest confidence in fake profile detection
- Neural network detects unusual profile patterns
Similar Profile Indices: [2, 5, 1, 3, 0]
```

---

## 🧠 Training Models

The project trains four models: Random Forest, XGBoost, Isolation Forest, and AnomalyNet. To train:

1. Ensure `data/Profiles.csv` and other CSV files are available.
2. Run:
   ```bash
   python src/run.py --mode train
   ```
3. Models and preprocessors are saved to the `models/` directory.
4. **Note**: The dataset in the logs has only 6 samples, which may lead to overfitting. Consider adding more data for better performance.

**Training Logs Example** (from your input):
```
2025-08-12 12:54:13,299 - INFO - Built RAG index with 6 embeddings
2025-08-12 12:54:13,299 - INFO - Training XGBoost model
2025-08-12 12:54:13,342 - INFO - Training Random Forest model
2025-08-12 12:54:13,371 - INFO - Training Isolation Forest model
2025-08-12 12:54:13,405 - INFO - Training AnomalyNet
```

---

## ⚙️ Configuration

* **`config.yaml`** – Defines data/model paths and parameters:
  ```yaml
  data_dir: "data"
  model_dir: "models"
  features:
    - age
    - country
    - subscribed
    - relationshipGoals
    - aboutMe
  tfidf_max_features: 100
  anomaly_epochs: 100
  contamination: 0.1
  ```
* **`requirements.txt`** – Key dependencies:
  ```plaintext
  pandas>=2.0.0
  numpy>=1.26.0
  scikit-learn>=1.3.0
  xgboost>=2.0.0
  pyod>=1.1.0
  torch>=2.0.0
  sentence-transformers>=2.2.0
  streamlit>=1.20.0
  joblib>=1.2.0
  pyyaml>=6.0.0
  pytest>=7.0.0
  ```

---

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push and open a pull request:
   ```bash
   git push origin feature/your-feature
   ```

Please include tests and ensure code follows PEP 8 style guidelines.

---

## 📜 License

Licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 📷 Screenshots

<img width="1716" height="769" alt="1 Fake Profile Detection" src="https://github.com/user-attachments/assets/3ce40f3e-3cee-47db-808b-d19aa2806a2c" />

<img width="1716" height="769" alt="2 Fake Profile Detection" src="https://github.com/user-attachments/assets/3047570f-a4b4-4cfd-8028-67719936f349" />

<img width="1728" height="1026" alt="3 Fake Profile Detection" src="https://github.com/user-attachments/assets/2c5d3b23-4f25-4d38-ae48-11c613c8e879" />

<img width="1728" height="1026" alt="4 Fake Profile Detection" src="https://github.com/user-attachments/assets/1ad96809-8671-40f3-be37-b397a2b7e2f9" />

---
