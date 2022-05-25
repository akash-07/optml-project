import os
import pandas as pd
import numpy as np
import tensorflow as tf
import argparse
import builtins
from models import *
from data_utils import *
from stopping import *
from evaluate import *

my_parser = argparse.ArgumentParser()

my_parser.add_argument(
    '-d',
    '--dataset',
    action='store',
    type=str,
    choices=['femnist'],
    default='femnist',
    help='Dataset on which to run experiment.')

my_parser.add_argument(
    '-traindir',
    '--training_dir',
    action='store',
    type=str,
    required=True,
    help='Absolute path to the directory containing training data.')

my_parser.add_argument(
    '-testdir',
    '--testing_dir',
    action='store',
    type=str,
    required=True,
    help='Absolute path to the directory containing testing data.')

my_parser.add_argument(
    '-r',
    '--learning_rate',
    action='store',
    type=float,
    required=True,
    help='Learning rate for training.')

my_parser.add_argument(
    '-b',
    '--batch_size',
    action='store',
    type=int,
    required=True,
    help='Batch size for training.')

my_parser.add_argument(
    '-lb',
    '--lower_bound',
    action='store',
    type=int,
    required=True,
    help='Lower bound for budget for  U(lb, up)')

my_parser.add_argument(
    '-up',
    '--upper_bound',
    action='store',
    type=int,
    required=True,
    help='Uppper bound for budget for  U(lb, up)')

my_parser.add_argument(
    '-l',
    '--logdir',
    action='store',
    type=str,
    default='./logs',
    required=True,
    help='Path to directory for logging. Creates one if not exists.')

my_parser.add_argument(
    '-n',
    '--num_clients',
    action='store',
    type=int,
    default=20,
    help='Number of clients to be selected in every round.')

my_parser.add_argument(
    '-f',
    '--fixed_rounds',
    action='store',
    type=int,
    help='Number of rounds to run if running for fixed rounds.')

my_parser.add_argument(
    '-ee',
    '--evaluate_every',
    action='store',
    type=int,
    default=3,
    help='Frequency of evaluation on test set.')

my_parser.add_argument(
    '-sm',
    '--save_model',
    action='store',
    type=bool,
    default=False,
    help='Set to True to save final global model to log directory. Default is False.'
)

my_parser.add_argument(
    '-mwf',
    '--model_weights_file',
    action='store',
    type=str,
    default=None,
    help='Points to the file containing model weights for same initialisation.'
)

my_parser.add_argument(
    '-sd',
    '--seed',
    action='store',
    type=int,
    required=True,
    help='Seed for sampling clients and their budgets.'
)

args = my_parser.parse_args()
for k,v in vars(args).items():
    print(k, ":", v)

# Set args from arg parser ---------------
eta = args.learning_rate
B = args.batch_size
lower_bound = args.lower_bound
upper_bound = args.upper_bound
log_dir = args.logdir
dset = args.dataset
hparams = {'batch_size':B}
train_dir = args.training_dir
test_dir = args.testing_dir
    
# Set functions/fields based on the dataset ---------------
check_stopping_criteria = None
dataset = None
preprocess = None
central_test_dataset = None
evaluate = None
builtins.model_fn = None
builtins.keras_model_fn = None
TC = None

#TC = 3597 (Total clients for non-iid)
if(dset == 'femnist'):
    # Make this global across all modules
    builtins.model_fn = tff_femnist_model_fn
    builtins.keras_model_fn = get_femnist_cnn

    TEST_BATCH_SIZE = 2048
    check_stopping_criteria = check_stopping_criteria_femnist
    dataset = FemnistData(train_dir, test_dir)
    preprocess = preprocess_femnist
    central_test_dataset = dataset.create_test_dataset_for_all_clients().batch(TEST_BATCH_SIZE).map(femnist_batch_format_fn)
    evaluate = evaluate_femnist

TC = len(dataset.num_samples)
print("Total number of client in this dataset:", TC)

# import fl_setup with globals set ---------------
# runs dynamic tracing with type-checking
from run import run_fl

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
tensorboard_logdir = os.path.join(log_dir, 'tb') 

with open(os.path.join(log_dir, 'params.txt'), 'w') as f:
    for k,v in vars(args).items():
        f.write(str(k) + " : " + str(v) + "\n")

# For reproducible client selection and budget sampling
rng = np.random.default_rng(args.seed)

_, _, _, state, train_metrics, test_metrics = run_fl(
    rng,
    dataset,
    preprocess,
    central_test_dataset,
    evaluate,
    check_stopping_criteria,
    hparams, 
    tensorboard_logdir,
    lower_bound,
    upper_bound,
    TC,
    num_clients=args.num_clients,
    fixed_rounds=args.fixed_rounds,
    evaluate_every=args.evaluate_every,
    lr_schedule=lambda _: eta,
    model_weights_file=args.model_weights_file)

# Write train, test, budgets and guesses csv ---------------
train_accuracies, train_losses, train_index = train_metrics
test_accuracies, test_losses, test_index = test_metrics
train_d = {'train_accuracies':train_accuracies, 'train_losses':train_losses, 'train_index':train_index}
test_d = {'test_accuracies':test_accuracies, 'test_losses':test_losses, 'test_index':test_index}
df_train = pd.DataFrame(data=train_d)
df_test = pd.DataFrame(data=test_d)
df_train.to_csv(os.path.join(log_dir, 'train.csv'), index=False)
df_test.to_csv(os.path.join(log_dir, 'test.csv'), index=False)

# Save model ---------------
if(args.save_model):
    if(dset == 'femnist'):
        keras_model = get_femnist_cnn()
        keras_model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]  
        )
        keras_model.set_weights(state)

    trained_model_path = os.path.join(log_dir, 'trained_model')
    if not os.path.exists(trained_model_path):
        os.mkdir(trained_model_path)
    keras_model.save(trained_model_path)