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

- Zeroshot

```bash
python run_generation.py --algo zeroshot --arch llama3 
```

- RAG

First, download the contriever model (already in [../model_weights/contriever]), cd the [./ranking] directory,  and run:

```bash
python ranking.py 
```

Then run
```bash
python run_generation.py --algo rag --arch llama3 --k 5
```

## Evaluation
Notice that evaluation should be done in network-type device.

```bash
python run_generation.py --algo zeroshot --arch llama3 --eval
```

```bash
python run_generation.py --algo rag --arch llama3 --k 5 --eval
```

## Prune and Generate
Basically, we prune the base model for each user using the user's past queries, then use the pruned model to make personalized generation. The algorithm is located in [models/PrunePredict.py].

To run the pipeline:

For unstructured:

```bash
python prune_and_predict --algo rag  
```

For structured:
```bash
python prune_and_predict --algo rag  --structured
```
