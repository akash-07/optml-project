import tensorflow_federated as tff
from fl_setup import *
from data_utils import make_federated_data
from math import floor

# Aggregation weights depend not only on number of samples
# but also on budgets since budgets define how many samples
# get seen.


def get_client_agg_weights(budgets, client_num_samples, batch_size):
    seen_training_samples = []
    tp_i = []
    for i in range(len(budgets)):
        tau_i = budgets[i]
        n_i = client_num_samples[i]
        tau_possible_i = floor(n_i / batch_size)
        tp_i.append(tau_possible_i)
        if(tau_i <= tau_possible_i):
            n_final_i = tau_i * batch_size
        else:
            n_final_i = n_i
        seen_training_samples.append(n_final_i)

    total_seen_training_samples = sum(seen_training_samples)
    agg_weights = [
        x/total_seen_training_samples for x in seen_training_samples]

    return agg_weights

def run_fl(
    # Random number generator for reproducible client and budget selection
    rng: np.random.Generator,
    dataset,  # Standardised dataset object
    preprocess,  # For dataset specfic processing - Dataset Depend (DD)
    central_test_dataset,  # Preprocessed central test dataset - DD
    evaluate,  # Function encapsulating keras compiled model for test set evaluation -  DD
    check_stopping_criteria,  # Function to evaluate stopping criteria - DD
    hparams,  # Hyperparameters, currently just contains batch size
    log_dir,  # Directory for logging
    lower_bound,  # lower bound for client budgets
    upper_bound,  # upper bound for client budgets
    total_clients,  # Total number of clients in the dataset
    num_clients=10,  # No of clients selected per round
    fixed_rounds=None,  # If specified, overrides check for stopping criteria
    evaluate_every=3,  # How often to evaluate on test set
    lr_schedule=None):  # f: int -> float ; takes round number as input
    
    iterative_process = tff.templates.IterativeProcess(
        initialize_fn=initialize_fn,
        next_fn=next_fn
    )

    state = iterative_process.initialize()

    print('[Beginning training]')

    round_num = 1

    # train_accuracies = []
    train_losses = []
    train_index = []

    # test_accuracies = []
    test_losses = []
    test_index = []

    with tf.summary.create_file_writer(log_dir).as_default():
        while(True):
            print("[Round {}]".format(round_num))

            # Evaluate before beginning any training
            if(round_num == 1):
                print('[Evaluating on test data...]')
                test_loss = evaluate(
                    state, central_test_dataset)

                # best_accuracy = test_accuracy
                # best_accuracy_round_num = round_num

                tf.summary.scalar('test_loss', test_loss, step=0)
                # tf.summary.scalar('test_accuracy', best_accuracy, step=0)

                print(
                    '[Test loss {}]'.format(test_loss))
                test_index.append(0)
                # test_accuracies.append(best_accuracy)
                test_losses.append(test_loss)

            client_indexes = rng.integers(
                low=0, high=total_clients-1, size=num_clients)
            
            # Studying effect of non-iid data
            client_indexes[-1] = 19
            client_ids = []
            client_num_samples = []
            for client_index in client_indexes:
                client_ids.append(dataset.client_ids[client_index])
                client_num_samples.append(dataset.num_samples[client_index])

            budgets = rng.integers(
                low=lower_bound, high=upper_bound+1, size=num_clients)

            print("Capacities [", *budgets, "]")
            print("Num samples [", *client_num_samples, "]")

            # ------- Compute/fetch all things necessary for training
            lr_to_clients = [lr_schedule(round_num)]*num_clients
            client_agg_weights = get_client_agg_weights(
                budgets, client_num_samples, hparams['batch_size'])

            federated_train_data = make_federated_data(
                dataset, preprocess, client_ids, client_num_samples, budgets, hparams['batch_size'], round_num)

            # ---------- Train one round
            state, metrics = iterative_process.next(
                state,
                federated_train_data,
                lr_to_clients,
                client_agg_weights
                )

            for name, value in metrics.items():
                tf.summary.scalar('train_' + name, value, step=round_num)
                if('loss' in name):
                    print('[Train loss', value, ']')
                    train_losses.append(value)
                # else:
                #     print('[Train accuracy', value, ']')
                #     train_accuracies.append(value)
            train_index.append(round_num)

            if((round_num+2) % evaluate_every == 0):
                print('[Evaluating on test data...]')
                test_loss = evaluate(
                    state, central_test_dataset)

                tf.summary.scalar('test_loss', test_loss, step=round_num)
                # tf.summary.scalar(
                    # 'test_accuracy', best_accuracy, step=round_num)

                print(
                    '[Test loss {}]'.format(test_loss))
                test_index.append(round_num)
                # test_accuracies.append(best_accuracy)
                test_losses.append(test_loss)

                if(fixed_rounds is not None and round_num >= fixed_rounds):
                    return round_num, state, (train_losses, train_index), (test_losses, test_index)
                elif(fixed_rounds is None and check_stopping_criteria(test_losses[-10:], round_num)):
                    return round_num, state, (train_losses, train_index), (test_losses, test_index)

            round_num = round_num + 1
