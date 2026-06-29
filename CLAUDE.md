# CLAUDE.md

This file provides strict guidance to Claude Code (claude.ai/code) when working with code in this repository. 

## 0. Communication & Persona (CRITICAL)
- Speak like a direct, pragmatic senior engineer discussing a problem with a peer.
- Keep explanations extremely concise, practical, and straight to the point.
- NO assumptions. Never guess. If something is missing, stop and ask.
- DO NOT use emojis anywhere (not in code, not in comments, not in chat responses).
- DO NOT use Turkish characters in code, variables, function names, or code comments. Use standard English ASCII only.

## 1. Agent Workflow Protocol (Strict Execution)
You must follow this exact sequence for ANY task, feature, or bug fix:
1. **Plan:** Create or update a plan document under the docs/ directory. Outline the phases and steps required.
2. **Wait for Approval:** Ask the user for explicit approval on the plan. Do NOT write code yet.
3. **Phase Execution:** Execute the plan phase by phase.
4. **Code:** Write the code for the current phase based on the approved plan.
5. **Test:** Write and run tests to verify the code works.
6. **Fix & Execute:** Fix any issues. Once a phase is successful, update the plan document in docs/ with the execution results. Move to the next phase only after user approval.

## 2. Problem Solving Framework
When facing an issue, bug, or metric degradation, follow this scientific method without making assumptions:
1. **Define the problem:** What is failing?
2. **Prove it exists:** Write a minimal script or test to reproduce the exact error.
3. **Find the root cause:** Investigate why it happens (data leakage, null values, type mismatch, etc.).
4. **Solve:** Implement the fix.
5. **Prove it is solved:** Run the exact reproduction script from step 2 and show it now passes.

## 3. Project Architecture & Coding Standards
All code must be written following Object-Oriented Programming (OOP) principles. Code must be highly modular, reusable, and follow industry standards (PEP8, type hinting, docstrings).

Project structure:
- docs/ -> Planning documents, phase tracking, and results.
- src/ -> Core modular packages.
  - features/ -> Feature engineering classes and transformers.
  - models/ -> Model definitions, training logic, and wrappers.
  - preprocess/ -> Data cleaning, extraction, and encoding.
  - calibration/ -> Model calibration (Platt scaling, isotonic regression).
  - evaluation/ -> Metrics, calibration curves, SHAP analysis.
  - scouting/ -> Player-level aggregation and over/under-performance analysis.
- scripts/ -> Executable entry points.
  - extract_shots.py -> Pull shot events from StatsBomb and build the dataset.
  - train.py -> Main training pipeline execution.
  - evaluate.py -> Run evaluation and generate calibration reports.
- notebooks/ -> EDA and presentation only. No business logic here.

## 4. Project Goal & Data
Build an Expected Goals (xG) model from scratch using StatsBomb Open Data, then extend it into a player evaluation / scouting framework.

Goal: Predict the probability that a given shot results in a goal (binary classification: is_goal = 0 or 1). This is a calibration-critical problem — the model must output well-calibrated probabilities, not just good classification.

Data source: StatsBomb Open Data via the `statsbombpy` Python library.
- Event-level data: each shot event contains location (x, y), shot outcome, body part, technique, play pattern, under_pressure flag, and a freeze_frame (positions of all players at the moment of the shot).
- StatsBomb also provides its own `statsbomb_xg` value per shot.

### Critical Data Rules
- **LEAKAGE:** `statsbomb_xg` must NEVER be used as a feature. It is ONLY for benchmarking the final model.
- **Scope:** Exclude penalties from the model (their xG is fixed ~0.76). Start with open-play shots only. Handle free kicks as a separate category.
- **Train/test split:** Split by match or season, NOT by random shot. Stratify on the target variable (goals are ~10% of shots).
- All preprocessing (encoding, scaling) must be fitted on training data only, inside a pipeline.

### Key Features to Engineer
- Distance from shot location to goal center.
- Angle subtended by the goalposts from the shot location (geometric calculation).
- Body part, shot type, play pattern (categorical).
- Under pressure flag.
- From freeze_frame: number of defenders between ball and goal, distance to nearest defender, goalkeeper position relative to goal line.

## 5. Modeling Strategy
This is a probability estimation problem, not a ranking problem. Calibration matters more than AUC.

### Approach
1. **Baseline:** Logistic Regression (interpretable, classic xG baseline). Use coefficients to sanity-check feature directions.
2. **Advanced:** Gradient Boosting (XGBoost or LightGBM). Hyperparameter tuning via cross-validation.
3. **Calibration:** Mandatory post-processing step. Generate reliability diagrams. Apply Platt scaling or isotonic regression if the model is miscalibrated.
4. **Interpretability:** SHAP values on the final model. Distance and angle should dominate — if they don't, investigate.

### Metrics (in priority order)
1. **Log-loss** (primary).
2. **Brier score** (complementary calibration metric).
3. **Calibration curve / reliability diagram** (visual validation — mandatory before declaring a model "done").
4. ROC-AUC (secondary, for discrimination ability).
5. Do NOT rely on accuracy alone — it is misleading with ~10% positive rate.

### Validation & Benchmarking
- Compare model xG to StatsBomb xG (shot-level correlation).
- Aggregate test: does season-total xG per player/team correlate with actual goals scored?
- Document where the model fails and why.

### Extension: Player Scouting
- Per-player: actual goals minus xG = finishing skill signal (over/under-performers).
- Per-team: total xG vs actual goals = which teams are over/under-performing?
- Apply minimum shot threshold to avoid small sample size bias.

## 6. Environment
Python 3.10+ virtual environment. Activate before running anything:

```
python -m venv env
source env/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

Key dependencies: statsbombpy, pandas, scikit-learn, xgboost, lightgbm, shap, matplotlib, seaborn.
