# 🏦 Credit Risk Classifier

> **Logistic Regression from scratch · Flask REST API · Docker · GitHub Actions CI/CD**

A production-grade machine learning project that predicts the probability of a loan applicant defaulting, built to demonstrate both ML fundamentals and modern software engineering / DevOps practices.

[![CI/CD](https://github.com/YOUR_USERNAME/credit-risk-classifier/actions/workflows/ci_cd.yml/badge.svg)](https://github.com/YOUR_USERNAME/credit-risk-classifier/actions)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [ML Explanation](#-ml-explanation)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [API Usage](#-api-usage)
- [Testing](#-testing)
- [Docker](#-docker)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Deployment](#-deployment)

---

## 🎯 Project Overview

This classifier answers one business question:

> **Given a loan applicant's financial profile, will they default on their loan?**

**Key highlights:**

- Logistic Regression implemented **from scratch** using only NumPy — no sklearn for training
- StandardScaler implemented **from scratch**
- Full REST API with input validation and structured JSON responses
- Comprehensive test suite (unit + integration + edge cases)
- Dockerized with a production WSGI server (Gunicorn)
- Automated CI/CD via GitHub Actions with Docker health checks and Render deployment

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CI/CD Pipeline                           │
│  Push → Install → Generate Data → Train → Test → Docker Build  │
│       → Health Check → Deploy (main branch only)               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Flask REST API (app.py)                     │
│                                                                 │
│   GET /          GET /health      GET /metrics   POST /predict  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    src/predict.py                            │
│               ModelRegistry (lazy loader)                    │
└────────┬────────────────────────────────────────┬───────────┘
         │                                        │
         ▼                                        ▼
┌─────────────────────┐               ┌──────────────────────┐
│   src/model.py      │               │  src/preprocess.py   │
│   LogisticRegression│               │  StandardScaler      │
│   (from scratch)    │               │  (from scratch)      │
└─────────────────────┘               └──────────────────────┘
         │                                        │
         ▼                                        ▼
┌─────────────────────────────────────────────────────────────┐
│                        models/                              │
│          model.pkl    scaler.pkl    metrics.txt             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧠 ML Explanation

### Why Logistic Regression?

Logistic Regression is the ideal baseline for binary classification because:
1. Outputs a **probability** (not just a label) — useful for risk scoring
2. Highly **interpretable** — weights reveal which features matter most
3. Computationally **efficient** — scales to large datasets

### How It Works

**1. Linear combination:**
```
z = w₁·age + w₂·income + w₃·credit_score + ... + b
```

**2. Sigmoid activation (squashes z → [0, 1]):**
```
P(default) = σ(z) = 1 / (1 + e^{-z})
```

**3. Binary Cross-Entropy Loss:**
```
L = -(1/m) Σ [y·log(ŷ) + (1-y)·log(1-ŷ)]
```

**4. Gradient Descent weight update:**
```
w ← w - α · (1/m) · Xᵀ(ŷ - y)
b ← b - α · (1/m) · Σ(ŷ - y)
```

Where `α` is the learning rate. This repeats for `n_iterations` until convergence.

### Features Used

| Feature | Type | Description |
|---|---|---|
| `age` | Numeric | Applicant age (18–100) |
| `income` | Numeric | Annual income in USD |
| `credit_score` | Numeric | FICO-like score (300–850) |
| `loan_amount` | Numeric | Requested loan amount |
| `loan_tenure_months` | Numeric | Loan duration in months |
| `debt_to_income_ratio` | Numeric | Existing debt / income (0–1) |
| `num_open_accounts` | Numeric | Open credit accounts |
| `num_credit_inquiries` | Numeric | Hard inquiries in last 12 months |
| `months_employed` | Numeric | Months at current employer |
| `has_mortgage` | Binary | Whether applicant owns a mortgage |

---

## 📁 Project Structure

```
credit-risk-classifier/
│
├── src/
│   ├── __init__.py
│   ├── model.py               # Logistic Regression (NumPy only)
│   ├── preprocess.py          # StandardScaler + data pipeline
│   ├── train.py               # Training orchestration + metrics
│   └── predict.py             # Inference for API
│
├── data/
│   ├── generate_data.py       # Synthetic dataset generator
│   └── dataset.csv            # Generated dataset (5,000 rows)
│
├── tests/
│   ├── conftest.py            # Shared pytest fixtures
│   ├── test_model.py          # Unit tests (model + scaler)
│   └── test_api.py            # Integration tests (all endpoints)
│
├── models/
│   ├── model.pkl              # Serialized trained model
│   ├── scaler.pkl             # Serialized StandardScaler
│   └── metrics.txt            # Human-readable evaluation metrics
│
├── notebooks/
│   └── explanation.ipynb      # Step-by-step walkthrough
│
├── .github/workflows/
│   └── ci_cd.yml              # GitHub Actions CI/CD pipeline
│
├── app.py                     # Flask application
├── Dockerfile                 # Production Docker image
├── render.yaml                # Render deployment config
├── requirements.txt
├── pytest.ini
├── Makefile                   # Developer convenience commands
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker (optional)
- Make (optional)

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/credit-risk-classifier.git
cd credit-risk-classifier
pip install -r requirements.txt
```

### 2. One-command setup

```bash
make all
# Equivalent to: install + data + train + test
```

### 3. Manual steps

```bash
# Generate dataset
python data/generate_data.py

# Train model
python -m src.train --lr 0.05 --iters 1000

# Run tests
pytest tests/ -v

# Start API server
python app.py
```

The API will be available at `http://localhost:5000`.

---

## 🌐 API Usage

### GET / — Welcome

```bash
curl http://localhost:5000/
```

```json
{
  "status": "success",
  "service": "Credit Risk Classifier API",
  "version": "1.0.0",
  "endpoints": { ... },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

### GET /health — Health Check

```bash
curl http://localhost:5000/health
```

```json
{
  "status": "success",
  "healthy": true,
  "model_loaded": true,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

Returns `503` if model artifacts are not loaded.

---

### GET /metrics — Model Performance

```bash
curl http://localhost:5000/metrics
```

```json
{
  "status": "success",
  "metrics": {
    "accuracy": 0.832,
    "precision": 0.801,
    "recall": 0.779,
    "f1_score": 0.789
  }
}
```

---

### POST /predict — Credit Risk Prediction

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "income": 65000,
    "credit_score": 680,
    "loan_amount": 25000,
    "loan_tenure_months": 36,
    "debt_to_income_ratio": 0.30,
    "num_open_accounts": 5,
    "num_credit_inquiries": 2,
    "months_employed": 48,
    "has_mortgage": 0
  }'
```

**Success response (200):**

```json
{
  "status": "success",
  "input": { ... },
  "result": {
    "prediction": 0,
    "probability": 0.182453,
    "risk_label": "Low Risk",
    "risk_description": "The applicant is unlikely to default on the loan."
  }
}
```

**Error response (400 — missing field):**

```json
{
  "status": "error",
  "message": "Missing required feature(s): ['credit_score']",
  "details": "Required features: [...]"
}
```

### Feature Constraints

| Feature | Min | Max |
|---|---|---|
| `age` | 18 | 100 |
| `credit_score` | 300 | 850 |
| `debt_to_income_ratio` | 0.0 | 1.0 |
| `has_mortgage` | 0 | 1 |
| `num_open_accounts` | 0 | 50 |
| `num_credit_inquiries` | 0 | 50 |
| `loan_tenure_months` | 1 | 360 |

---

## 🧪 Testing

### Run all tests

```bash
pytest tests/ -v
```

### Run specific test files

```bash
pytest tests/test_model.py -v   # Unit tests
pytest tests/test_api.py -v     # API integration tests
```

### Coverage report

```bash
make test-coverage
# Opens htmlcov/index.html
```

### Test categories

| File | Tests | Coverage |
|---|---|---|
| `test_model.py` | Sigmoid, loss, gradient descent, weight init, serialization, scaler | `src/model.py`, `src/preprocess.py` |
| `test_api.py` | All endpoints (valid + invalid inputs, edge cases, error handlers) | `app.py`, `src/predict.py` |

---

## 🐳 Docker

### Build

```bash
docker build -t credit-risk-classifier .
# or
make docker-build
```

### Run

```bash
docker run -d \
  --name credit-risk-classifier \
  -p 5000:5000 \
  credit-risk-classifier:latest

# or
make docker-run
```

### Test container

```bash
curl http://localhost:5000/health
```

### Stop

```bash
make docker-stop
```

### Image details

- **Base**: `python:3.11-slim` (minimal attack surface)
- **WSGI server**: Gunicorn (2 workers × 2 threads)
- **User**: Non-root (`appuser`)
- **Health check**: Native Docker HEALTHCHECK on `/health`

---

## 🔄 CI/CD Pipeline

The GitHub Actions pipeline runs on every push:

```
Push to any branch
       │
       ▼
┌─────────────────────────────────────┐
│  JOB 1: Build & Test                │
│  ─────────────────────────────────  │
│  1. Checkout code                   │
│  2. Set up Python 3.11              │
│  3. Install dependencies            │
│  4. Generate synthetic dataset      │
│  5. Train model                     │
│  6. Verify artifacts exist          │
│  7. Run pytest (fail → stop here)   │
└─────────────────┬───────────────────┘
                  │ (on success)
                  ▼
┌─────────────────────────────────────┐
│  JOB 2: Docker Build + Health Check │
│  ─────────────────────────────────  │
│  8.  Build Docker image             │
│  9.  Run container                  │
│  10. Wait for startup               │
│  11. Health check (GET /health)     │
│  12. Smoke test all endpoints       │
│  13. Stop container                 │
└─────────────────┬───────────────────┘
                  │ (main branch only)
                  ▼
┌─────────────────────────────────────┐
│  JOB 3: Deploy to Render            │
│  ─────────────────────────────────  │
│  14. Trigger Render webhook         │
└─────────────────────────────────────┘
```

### Required GitHub Secrets (for deployment)

| Secret | Description |
|---|---|
| `RENDER_API_KEY` | Your Render API key |
| `RENDER_SERVICE_ID` | Your Render service ID |

---

## ☁️ Deployment

This project is configured for deployment on **Render**.

1. Connect your GitHub repo to Render
2. Render auto-detects `render.yaml`
3. Add environment secrets in Render dashboard
4. Push to `main` → GitHub Actions triggers deployment

The `render.yaml` configures:
- Docker-based deployment
- Health check on `/health`
- `autoDeploy: false` (controlled by GitHub Actions only)

---

## 📊 Model Performance

Trained on 5,000 synthetic samples with 70/10/20 train/val/test split:

| Metric | Score |
|---|---|
| Accuracy | ~83% |
| Precision | ~80% |
| Recall | ~78% |
| F1 Score | ~79% |

Performance varies slightly per run due to synthetic data generation randomness.

---

## 🛠️ Makefile Reference

```bash
make help          # Show all commands
make install       # Install dependencies
make data          # Generate dataset
make train         # Train model
make test          # Run test suite
make test-coverage # Run tests + HTML coverage
make run           # Start dev server
make docker-build  # Build Docker image
make docker-run    # Run container
make docker-stop   # Stop container
make docker-logs   # Tail container logs
make clean         # Remove artifacts
make all           # install + data + train + test
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built to demonstrate ML fundamentals + production software engineering practices.*
