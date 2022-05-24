# optml-project
In this project, we question the assumptions for data heterogeneity in FedAvg analysis.

## Installation details

- We are using python 3.8.10
- The `requirements.txt` file has all modules with correct versions
- You can set it up as follows:

```
# create the virtual environment
python3 -m venv venv

# activate it
source venv/bin/activate

pip install -r requirements.txt
```

## Datasets

We will use the [leaf FL benchmark](https://github.com/TalwalkarLab/leaf) for performing tests. It contains variety of datasets aimed at different learning tasks. The current code supports the FEMNIST dataset.

LEAF has been added a submodule. To generate the dataset, do the following:

```
# pull git submodule locally
git submodule update --init --recursive

cd leaf/data/feminst

# command description at https://github.com/TalwalkarLab/leaf/tree/master/data/femnist
./preprocess.sh -s niid --sf 1.0 -k 100 -t sample --smplseed 10 --spltseed 10
```

This make take an hour depending on your machine. So only do this where you intend to run the code.

## Instructions for running the code

The script to run is `main.py`. It supports multiple command line arguments:

```
usage: main.py [-h] [-d {femnist}] -traindir TRAINING_DIR -testdir TESTING_DIR -r LEARNING_RATE -b BATCH_SIZE -lb
               LOWER_BOUND -up UPPER_BOUND -l LOGDIR [-n NUM_CLIENTS] [-f FIXED_ROUNDS] [-ee EVALUATE_EVERY]
               [-sm SAVE_MODEL] [-mwf MODEL_WEIGHTS_FILE] -sd SEED

optional arguments:
  -h, --help            show this help message and exit
  -d {femnist}, --dataset {femnist}
                        Dataset on which to run experiment.
  -traindir TRAINING_DIR, --training_dir TRAINING_DIR
                        Absolute path to the directory containing training data.
  -testdir TESTING_DIR, --testing_dir TESTING_DIR
                        Absolute path to the directory containing testing data.
  -r LEARNING_RATE, --learning_rate LEARNING_RATE
                        Learning rate for training.
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch size for training.
  -lb LOWER_BOUND, --lower_bound LOWER_BOUND
                        Lower bound for budget for U(lb, up)
  -up UPPER_BOUND, --upper_bound UPPER_BOUND
                        Uppper bound for budget for U(lb, up)
  -l LOGDIR, --logdir LOGDIR
                        Path to directory for logging. Creates one if not exists.
  -n NUM_CLIENTS, --num_clients NUM_CLIENTS
                        Number of clients to be selected in every round.
  -f FIXED_ROUNDS, --fixed_rounds FIXED_ROUNDS
                        Number of rounds to run if running for fixed rounds.
  -ee EVALUATE_EVERY, --evaluate_every EVALUATE_EVERY
                        Frequency of evaluation on test set.
  -sm SAVE_MODEL, --save_model SAVE_MODEL
                        Set to True to save final global model to log directory. Default is False.
  -mwf MODEL_WEIGHTS_FILE, --model_weights_file MODEL_WEIGHTS_FILE
                        Points to the file containing model weights for same initialisation.
  -sd SEED, --seed SEED
                        Seed for sampling clients and their budgets.
```

Elements to note:
- The number of steps that each client performs locally is sampled from a uniform distribution `[lb, up]`. Thus depending on the values of `-lb` and `-up` parameters, they may perform same or different number of local steps.
- Since model initialisation affects convergence, to conduct reproducible experiments we can specify which initial model to pick via the file path to model weights (`-mwf` parameter). The folder `model_weights` contains a few initial models for every dataset.
- Seed determines the client selection for every communication round.
- Train and test directory paths correspond to how LEAF generates the data. This should generally be `leaf/data/dataset_name/data/{train, test}`.
- The generated logs contain test metrics per `-ee` rounds and train metrics per round. They also contain a tensorboard log (in `\tb` dir) which can be directly used to visualise performance as: `tensorboard --logdir <path to log directory/tb>`.
![Screenshot 2022-05-24 at 16 54 48](https://user-images.githubusercontent.com/24961068/170066838-b7eaaaea-090c-42a5-b106-103f4d611e7f.png)

Rest all parameters are self-explanatory. Checkout `myrun.sh` for an example.

For FEMNIST dataset, one could do:

`./myrun.sh femnist 0.02 20 20 20 15 0 0`
