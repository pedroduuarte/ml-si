# Model artifacts

This folder stores generated artifacts for the bonus deployment phase.

Run the training script after generating `data/cleaned_dataset.csv`:

```bash
python scripts/train_bonus_model.py
```

The script creates `customer_churn_decision_tree.pkl` and a metadata JSON file.
Those generated files are ignored by Git because they can be recreated from the
clean dataset.
