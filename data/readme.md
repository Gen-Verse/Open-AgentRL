# Dataset

## Download

The training data in coding setting is `CodeContests_train`, the evaluation dataset is `LiveBench-ReasonFlux`. You can simply download them by
```bash
# download the evaluation data
python download_data.py --dataset LiveBench-ReasonFlux
# download the training data
python download_data.py --dataset CodeContests_train
```

We use Stdio input/output format here. For example, for the task to calculate the sum of a list, the input and output are in the following format:
```python
input = "5\n1 2 3 4 5\n"
output = "15"
```

## Preprocess Dataset

You need to preprocess the downloaded dataset in `preprocess_data.ipynb`.
