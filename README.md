# SciFact Project

Baseline in https://github.com/allenai/scifact 

Download data: https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz 

## Retrieve the Rationales (sentences in abstracts)

```

python rationale_model.py

python rational_selection.py --dataset data/claims_dev.jsonl --output predictions/predicted_rationale_dev.jsonl --abstract_store predictions/abstract_dev.jsonl --k_rationales 3

python rational_selection.py --dataset data/claims_train.jsonl --output predictions/predicted_rationale_train.jsonl --abstract_store predictions/abstract_train.jsonl --k_rationales 3

python rationale_evaluation.py --corpus data/corpus.jsonl --dataset data/claims_dev.jsonl --rationale-selection predictions/predicted_rationale_dev.jsonl

```

## Verify the Claim: SUPPORTS, REFUTES, NOTENOUGHINFO

```
python label_model.py
```

