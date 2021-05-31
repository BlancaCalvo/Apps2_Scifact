# SciFact Project

Baseline in https://github.com/allenai/scifact 

Download data: https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz 

## Retrieve the Evidence (sentences in abstracts)

```
python rationale_model.py

python rational_selection.py 

python rationale_evaluation.py 
```

## Verify the Claim: SUPPORTS, REFUTES, NOTENOUGHINFO

```
python label_model.py

python rational_selection.py --include-nei --output predicted/predictions_with_nei.jsonl

python label_selection.py

python label_evaluation.py
```

