# 🩺 AI Health Assistant — Multi-Disease Risk Screening Platform

A production-style Data Science + Analytics portfolio project that demonstrates how to operationalize machine learning models into a recruiter-friendly web product.

## Why this project stands out
- **Business-facing framing:** clear value proposition, risk-screening workflows, and user-oriented UX.
- **End-to-end DS lifecycle:** includes datasets, model training artifacts, packaged inference logic, and deployment setup.
- **Production-minded structure:** modular codebase, basic tests, reproducible dependencies, and CI-ready layout.
- **Portfolio ready:** can be showcased directly in LinkedIn Projects section with concise impact narrative.

## Live capabilities
This Streamlit application provides interactive risk screening for:
1. Diabetes
2. Heart Disease
3. Parkinson's Disease
4. Liver Disease

> ⚠️ **Important:** This app is for educational and demonstration purposes only, not medical diagnosis.

---

## Project Architecture

```text
Disease_Predictor/
├── app.py                          # Streamlit entrypoint
├── src/health_assistant/
│   ├── config.py                   # app-level constants and paths
│   ├── inference.py                # model loading + prediction helpers
│   └── ui.py                       # all Streamlit interface logic
├── dataset/                        # raw training datasets
├── saved_models/                   # serialized trained models
├── tests/                          # baseline unit tests
├── .streamlit/config.toml          # UI theme/server settings
├── .github/workflows/              # CI/CD workflow(s)
├── colab_files_to_train_models/    # original training notebooks/scripts
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1) Clone & install
```bash
git clone <your-repo-url>
cd Disease_Predictor
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Run the app
```bash
PYTHONPATH=src streamlit run app.py
```

### 3) Run tests
```bash
PYTHONPATH=src pytest -q
```

---

## Recruiter-Focused Project Highlights

- Built a **multi-model clinical risk screening system** with modular ML inference architecture.
- Designed **user-friendly analytics UI** for non-technical stakeholders using Streamlit.
- Added **input validation and error handling** to improve reliability and user trust.
- Structured repository for **maintainability, deployment, and professional presentation**.

---

## Suggested LinkedIn Project Description (copy/paste)

**AI Health Assistant (Multi-Disease Risk Screening Platform)**  
Designed and deployed a production-style ML web app that predicts risk for diabetes, heart disease, Parkinson's, and liver disease. Refactored monolithic code into modular architecture (`config`, `inference`, `ui`), added validation + tests, and prepared a recruiter-friendly repository with deployment assets and documentation. Built with Python, Scikit-learn, and Streamlit.

---

## Tech Stack
- **Language:** Python
- **ML:** Scikit-learn
- **Data:** Pandas, NumPy
- **App Layer:** Streamlit
- **Testing:** Pytest
- **Deployment-ready:** Docker + Render config

---

## Roadmap
- Add model performance dashboard (ROC, PR curves, confusion matrices).
- Add experiment tracking (MLflow/W&B).
- Add API layer (FastAPI) and separate frontend/backend services.
- Add data validation and drift monitoring.

