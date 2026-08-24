Trained model artifacts land here (model.pkl).

Workflow:
  docker compose exec detector python -m ml_scorer.train
  docker compose restart ml-scorer

The scorer auto-loads model.pkl when present; otherwise it runs heuristic-v0.
