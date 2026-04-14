# Credit Risk Classifier - Presentation Guide (Simple Words)

This guide helps you explain the whole project in class, especially:

- CI/CD pipeline
- ML flow (end-to-end process)
- Flask API and frontend link
- Docker and deployment flow

---

## 1) One-line Project Story

"This project takes loan applicant data, runs it through a trained ML pipeline, and serves results through a Flask API and a web UI, with automated testing and deployment through CI/CD."

---

## 2) High-Level Flow (Start to End)

1. Data is generated (`data/generate_data.py`).
2. Training pipeline runs (`src/train.py`).
3. Artifacts are saved in `models/` (`model.pkl`, `scaler.pkl`, `metrics.txt`).
4. Flask API loads these artifacts lazily (`src/predict.py` + `app.py`).
5. Frontend page (`templates/index.html`) sends input to `POST /predict`.
6. CI/CD (`.github/workflows/ci_cd.yml`) automates test, Docker check, and deploy.

---

## 3) ML Flow (Not ML Model Math, Just System Flow)

Important: This repo does **not** use the MLflow tracking tool/package.  
When your professor says "MLFlow", explain the **machine learning flow** in this project:

### Step A: Data Creation
- File: `data/generate_data.py`
- Function: `generate_credit_risk_dataset(...)`
- What it does:
  - Creates realistic synthetic credit data
  - Creates target label `default`
  - Saves CSV to `data/dataset.csv`

### Step B: Preprocess + Train + Evaluate + Save
- File: `src/train.py`
- Function: `train(...)`
- What it does:
  - Calls preprocessing pipeline
  - Trains model
  - Evaluates validation/test metrics
  - Saves artifacts:
    - `models/model.pkl`
    - `models/scaler.pkl`
    - `models/metrics.txt`

### Step C: Inference Layer
- File: `src/predict.py`
- Class: `ModelRegistry`
- Function: `predict(raw_input)`
- What it does:
  - Checks artifact readiness (`is_ready`)
  - Loads model/scaler lazily
  - Preprocesses one input
  - Returns prediction + probability + risk label

### Step D: Serving via API
- File: `app.py`
- Endpoint: `POST /predict`
- What it does:
  - Validates JSON
  - Validates required fields and numeric/range checks
  - Calls `predict(...)` from `src/predict.py`
  - Returns structured JSON response

---

## 4) CI/CD Pipeline Explained in Simple Words

- File: `.github/workflows/ci_cd.yml`
- Trigger:
  - Push on any branch
  - Pull request to `main`

### Job 1: Build & Test
- Setup Python (`3.11`) and install dependencies
- Generate data
- Train model
- Verify artifacts exist
- Run pytest
- If fail, upload artifacts/log context

### Job 2: Docker Build & Health Check
- Runs only if Job 1 passes (`needs: build-and-test`)
- Re-generates artifacts
- Builds Docker image
- Runs container
- Checks `/health`
- Runs smoke tests for:
  - `/`
  - `/metrics`
  - `/predict`

### Job 3: Deploy to Render
- Runs only on `main` pushes
- Calls Render API using secrets:
  - `RENDER_API_KEY`
  - `RENDER_SERVICE_ID`

---

## 5) How Flask API Is Linked with Frontend

### Backend API
- File: `app.py`
- Key routes:
  - `GET /` -> info
  - `GET /health` -> readiness check
  - `GET /metrics` -> metrics from saved file
  - `POST /predict` -> prediction

### Frontend UI
- File: `templates/index.html`
- Route:
  - `GET /app` in `app.py` returns this page
- In page JavaScript:
  - Takes form values
  - Sends `fetch("/predict", { method: "POST", ... })`
  - Shows JSON result on page

So the frontend and backend are connected by the `POST /predict` endpoint.

---

## 6) Why We Use These Components

- **Flask**: simple and fast REST API for serving ML predictions.
- **ModelRegistry in `src/predict.py`**: avoids loading model every request (better performance).
- **Validation in `app.py`**: protects API from bad input.
- **Docker**: same runtime locally and in cloud, fewer "works on my machine" issues.
- **GitHub Actions CI/CD**: automatic quality checks before deployment.
- **Render deployment**: easy cloud hosting using Docker image.

---

## 7) Quick "Code Location" Map

- Data generation: `data/generate_data.py`
- Training orchestration: `src/train.py`
- Inference and artifact loader: `src/predict.py`
- API routes and validation: `app.py`
- Frontend UI page: `templates/index.html`
- CI/CD automation: `.github/workflows/ci_cd.yml`
- Container build/runtime: `Dockerfile`
- Cloud deployment config: `render.yaml`

---

## 8) Demo Script for Class (2-3 minutes)

1. "I start by generating data and training artifacts."
2. "The API uses those artifacts via a lazy registry."
3. "I can open `/app` and submit user data."
4. "Frontend calls `/predict` and shows result."
5. "In CI/CD, every push runs data -> train -> test -> Docker health check."
6. "If branch is `main`, pipeline triggers Render deploy."

---

## 9) Common Questions + Short Answers

### Q: Where is the ML flow defined?
A: In `data/generate_data.py` (data), `src/train.py` (training/eval/save), and `src/predict.py` (inference loading + prediction).

### Q: How do you ensure API is production-safe?
A: Input validation + error handling in `app.py`, plus Docker health checks and CI smoke tests in `.github/workflows/ci_cd.yml`.

### Q: What fails deployment if something is wrong?
A: CI jobs fail early. Docker health checks and endpoint smoke tests block bad builds before deploy.

### Q: How does frontend connect to backend?
A: The form in `templates/index.html` sends JSON to `POST /predict` in `app.py`.

### Q: Why not load model each request?
A: `ModelRegistry` in `src/predict.py` caches loaded artifacts for speed and stability.

---

## 10) Important Clarification for "MLflow" Questions

If asked "Where is MLflow tracking server?":

- This project currently has **ML flow**, not the **MLflow tool integration**.
- You can say: "We implemented complete ML lifecycle flow, and a future upgrade can add MLflow tracking for experiments and artifact registry."

