# PersonalAgent

## Data processing
First download ***Time-based*** data from LaMP website into [./data] directory. A sample directory is as below.
- data
    - LaMP_4
        - train
        - test
        - valid
        - processed (*empty, to save the processed data.)

Then change arguments in data_process.py to process the data.

## Generation
For zeroshot and rag, run run_generation.py as below:

```bash
python run_generation.py --algo zeroshot --arch llama3 
```

```bash
python run_generation.py --algo rag --arch llama3 --k 5
```

## Evaluation
Notice that evaluation should be down in network-type device.

```bash
python run_generation.py --algo zeroshot --arch llama3 --eval
```

```bash
python run_generation.py --algo rag --arch llama3 --k 5 --eval
```

## Todos
- Hydra implementation (in progress)
- Pruning 
    - Ensure that the pruning is "structral".
    - Extract the def pruning_model(xxx) (See the feishu doc)
