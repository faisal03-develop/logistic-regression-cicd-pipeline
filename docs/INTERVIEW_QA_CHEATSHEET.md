# Interview Q&A Cheat Sheet (Simple Words)

## 1) Explain this project in 30 seconds
"This is an end-to-end ML system. We generate data, train and save artifacts, serve predictions through Flask API, connect a frontend UI to that API, and use CI/CD to auto-test, containerize, and deploy."

---

## 2) What exactly happens on each git push?
- CI starts from `.github/workflows/ci_cd.yml`.
- Build & Test job:
  - install deps
  - generate data
  - train
  - verify artifacts
  - run tests
- Docker job:
  - build image
  - run container
  - hit `/health`, `/`, `/metrics`, `/predict`
- Main branch push also triggers Render deploy.

---

## 3) How does request travel from frontend to model?
1. User opens `/app` (`app.py` -> `frontend_app`).
2. Form in `templates/index.html` sends JSON to `/predict`.
3. `app.py` validates input.
4. `app.py` calls `predict(...)` in `src/predict.py`.
5. `src/predict.py` preprocesses and runs model.
6. JSON response returns to page.

---

## 4) Why did you add validation in API?
- To reject bad inputs early.
- To keep model stable and safe.
- Validation logic is in `app.py` under `predict_endpoint`.

---

## 5) What is artifact management here?
- Managed by `ModelRegistry` in `src/predict.py`.
- Checks files exist with `is_ready()`.
- Lazy loads model/scaler/metrics only when needed.
- Avoids repeated heavy loading.

---

## 6) Why Docker in this project?
- Same runtime in dev and production.
- Includes health check (`Dockerfile` HEALTHCHECK).
- Runs with Gunicorn in container (`Dockerfile` CMD).
- Better reliability for deployment.

---

## 7) How do you know deployment is safe?
- CI fails fast if:
  - tests fail
  - artifacts missing
  - health check fails
  - smoke tests fail
- Only after passing checks does deploy trigger on `main`.

---

## 8) What if professor asks: "Where is MLflow used?"
Suggested answer:
"This project implements full ML lifecycle flow but does not yet use the MLflow tracking tool. A next step is integrating MLflow for experiment tracking and model registry."

---

## 9) Strong closing line for viva
"This project is not just model training; it is an MLOps-style pipeline where data, training, serving, testing, containerization, and deployment are automated and connected."

