# Time-to-event prediction (survival analysis)

Predict when an event will occur, given features and possibly censored observations (subjects who haven't experienced the event yet).

This directory is unusual in that it offers three different libraries:
- **`lifelines/`** — classical parametric survival models. Interpretable, fast, well-understood.
- **`scikit_survival/`** — Cox proportional hazards with sklearn-style API.
- **`pytorch/`** — neural survival models for non-linear hazard modeling.

## Start here

**New to survival analysis?** Use [`lifelines/weibull_aft.py`](lifelines/weibull_aft.py). Most common parametric survival baseline; trains instantly and gives interpretable coefficients.

For deep-learning approaches, try [`pytorch/deepsurv.py`](pytorch/deepsurv.py) — neural Cox proportional hazards model.

## Models

### lifelines (classical parametric)

| Model | When to pick |
|---|---|
| [`weibull_aft.py`](lifelines/weibull_aft.py) | Most common parametric baseline |
| [`log_normal_aft.py`](lifelines/log_normal_aft.py) | When `log(time)` is approximately normal |
| [`log_logistic_aft.py`](lifelines/log_logistic_aft.py) | Non-monotonic hazards |

### scikit_survival

| Model | When to pick |
|---|---|
| [`cox_ph.py`](scikit_survival/cox_ph.py) | Classical semi-parametric Cox model; interpretable coefficients |

### PyTorch (neural survival)

| Model | When to pick |
|---|---|
| [`deepsurv.py`](pytorch/deepsurv.py) | Neural Cox; non-linear hazards |
| [`multilayer_norm.py`](pytorch/multilayer_norm.py) | Deep model with normalization; moderate-size datasets |
| [`simple_neural.py`](pytorch/simple_neural.py) | Small feed-forward net; starting point for deep survival |
| [`simple_linear.py`](pytorch/simple_linear.py) | Linear survival model; comparison against lifelines |

## Dataset expectations

- **Input**: tabular rows with `num_feature_points` numeric features.
- **Labels**: `(time_to_event, event_observed)` pairs. `event_observed=0` means censored.
