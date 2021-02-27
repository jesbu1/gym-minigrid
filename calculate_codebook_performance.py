import numpy as np
from calculate_mdl import *
import argparse
import os
import pandas as pd
from sklearn import metrics as sk_metrics

def calculate_auc(curves):
    aucs = []
    for curve in curves:
        curve = np.array(curve)
        epochs = np.arange(len(curve))
        aucs.append(sk_metrics.auc(epochs, curve))
    return np.mean(aucs), np.std(aucs), np.array(aucs).tolist()

def calculate_regret(curves):
    regrets = []
    for curve in curves:
        curve = np.array(curve)
        upper_bound = 0.7849
        regrets.append(np.sum(upper_bound - curve))
    return np.mean(regrets), np.std(regrets), np.array(regrets).tolist()

def smooth(list_of_list_of_scalars, weight: float):  # Weight between 0 and 1
    list_of_smoothed = []
    for scalars in list_of_list_of_scalars:
        last = scalars[0]  # First value in the plot (first timestep)
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
            smoothed.append(smoothed_val)                        # Save it
            last = smoothed_val                                  # Anchor the last smoothed value
        list_of_smoothed.append(smoothed)
    return list_of_smoothed

def calculate_codebook_metrics(location):
    """
    Return list of smoothed, averaged codebook returns given the codebook location
    """
    dirs_to_search = ['rl_logs_train', 'rl_logs_test']
    metrics = {}
    for log_dir in dirs_to_search:
        for codebook_file in os.listdir(os.path.join(location, log_dir)):
            for seed_dir in os.listdir(os.path.join(location, log_dir, codebook_file)):
                    for inner_file in os.listdir(os.path.join(location, log_dir, codebook_file, seed_dir)):
                        if inner_file.endswith('progress.csv'):
                            progress_csv = os.path.join(location, log_dir, codebook_file, seed_dir, inner_file)
                            df = pd.read_csv(progress_csv)
                            rewards = df['evaluation/Average Returns'].to_numpy()
                            #path_length = df['evaluation/path length Mean'].to_numpy()
                            stripped_codebook_file = codebook_file.replace('rl_', '')
                            stripped_codebook_file += '.npy'
                            if stripped_codebook_file not in metrics:
                                metrics[stripped_codebook_file] = dict(train=[], test=[])
                            if 'train' in log_dir:
                                metrics[stripped_codebook_file]['train'].append(rewards)
                            else: 
                                metrics[stripped_codebook_file]['test'].append(rewards)
    for codebook_file in metrics.keys():
        smoothed_train = smooth(metrics[codebook_file]['train'], 0.5)
        smoothed_test = smooth(metrics[codebook_file]['test'], 0.5)
        metrics[codebook_file]['train'] = smoothed_train # (np.mean(smoothed_train, axis=0), np.var(smoothed_train))
        metrics[codebook_file]['test'] = smoothed_test # (np.mean(smoothed_test, axis=0), np.var(smoothed_test))
    return metrics 

def discover_evaluations(location):
    """
    Return list of evaluations given the codebook location
    """
    codebooks = []
    for codebook_file in os.listdir(location):
        if codebook_file.endswith('.npy'):
            with open(os.path.join(location, codebook_file), 'rb+') as f:
                codebook = np.load(f, allow_pickle=True)
            codebooks.append((codebook_file, codebook.item())) #.item() to extract dictionary from 0d array
    return codebooks

def process_evaluation(evaluation, codec, tree_bits, name, trajectory_dict):
    """
    Adds to trajectory_dict the mappings from start/end positions to the
    various metrics stored for the associated codebook
    """
    test_trajectories = evaluation.pop('test')
    train_trajectories = evaluation.pop('train')
    def process_trajectories(trajectories, traj_type):
        for trajectory, node_cost, start, end in trajectories:
            trajectory_id = (start, end)
            cleaned_trajectory = list(filter(lambda a: a != "", trajectory.split(" ")))
            code_length = len(codec.encode(cleaned_trajectory)) * 8
            num_primitive_actions = len(trajectory.replace(" ", ""))
            num_abstract_actions = len(cleaned_trajectory)
            metrics = dict(
                        num_primitive_actions=num_primitive_actions,
                        num_abstract_actions=num_abstract_actions,
                        code_length=code_length, 
                        description_length=code_length + tree_bits, 
                        node_cost=node_cost)
                        
            if trajectory_id not in trajectory_dict[traj_type]:
                trajectory_dict[traj_type][trajectory_id] = {name: metrics}
            else:
                trajectory_dict[traj_type][trajectory_id][name] = metrics
    process_trajectories(train_trajectories, 'train')
    process_trajectories(test_trajectories, 'test')

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate MDL of a directory of codebooks which are encoded in .npy format')
    parser.add_argument('location', type=str,
                        help='a file location where codebooks are stored')
    args = parser.parse_args()

    codebooks = discover_codebooks(args.location)
    codebook_name_dl_tuples = []
    codebook_dict = {}
    previous_length_range = None
    track_probs = True
    for codebook in codebooks:
        length_range = codebook[1].pop('length_range')
        probabilities = codebook[1].pop('probabilities')
        if track_probs != False:
            # If different lengths, then don't track this
            if previous_length_range != None:
                if length_range != previous_length_range:
                    track_probs = False
            previous_length_range = length_range
        dl, tree_bits, codec, uncompressed_len = calculate_codebook_dl(codebook[1])
        codebook_name_dl_tuples.append((codebook[0], dl, tree_bits, codec, length_range, probabilities, uncompressed_len))
    sorted_codebooks_by_dl = sorted(codebook_name_dl_tuples, key=lambda x: x[1])
    for name, dl, tree_bits, codec, length_range, probabilities, uncompressed_len in sorted_codebooks_by_dl:
        #print(name, dl, uncompressed_len)
        print(name, dl)
        codebook_dict[name] = dict(description_length=dl, 
                                   tree_bits=tree_bits, 
                                   codec=codec, 
                                   length_range=length_range,
                                   probabilities=probabilities,
                                   uncompressed_len=uncompressed_len)

    evaluations = discover_evaluations(os.path.join(args.location, 'evaluations'))
    has_rl = False
    try:
        metrics = calculate_codebook_metrics(args.location)
        has_rl = True
    except FileNotFoundError as e:
        print("No RL Logs Detected")
    trajectory_dict = dict(train={}, test={}, probabilities={})
    for codebook_name, evaluation in evaluations:
        original_name = codebook_name.replace("trajectories_", "")
        codebook_info = codebook_dict[original_name]
        process_evaluation(evaluation, codebook_info['codec'], codebook_info['tree_bits'], original_name, trajectory_dict) 

    # building a pandas dataframe
    pd_index = []
    pd_dict = {'codebook_dl': [], 'num_symbols': [], 'test_rl_auc': [], 'test_rl_regret': [],
               'test_auc_std': [], 'test_regret_std': [], 'test_regrets': []}
    if not has_rl:
        pd_dict.pop('test_rl_auc')
        pd_dict.pop('test_rl_regret')
        pd_dict.pop('test_auc_std')
        pd_dict.pop('test_regret_std')
        pd_dict.pop('test_regrets')
    length_set = set()
    for name, dl, *_ in sorted_codebooks_by_dl:
        pd_index.append(name)
        def accumulate_values(traj_type):
            values_to_track = ['num_primitive_actions', 'num_abstract_actions', 'code_length', 'description_length', 'node_cost']
            values = [0 for _ in values_to_track]
            for start_end_pair in trajectory_dict[traj_type].keys():
                for i, value_name in enumerate(values_to_track):
                    values[i] += trajectory_dict[traj_type][start_end_pair][name][value_name]
            for i in range(len(values)):
                values[i] /= len(trajectory_dict[traj_type].keys())
            values_to_track = [f'{traj_type}_{value_name}' for value_name in values_to_track]
            for i, column_name in enumerate(values_to_track):
                if column_name not in pd_dict:
                    pd_dict[column_name] = []
                pd_dict[column_name].append(values[i])
        accumulate_values('train')
        accumulate_values('test')
        pd_dict['codebook_dl'].append(dl)
        if track_probs:
            for i, length in enumerate(codebook_dict[name]['length_range']):
                length = str(length)
                length_set.add(length)
                if length not in pd_dict:
                    pd_dict[length] = []
                pd_dict[length].append(codebook_dict[name]['probabilities'][i])
        pd_dict['num_symbols'].append(len(codebook_dict[name]['codec'].get_code_table()))
        if has_rl:
            test_auc_mean, test_auc_std, test_aucs = calculate_auc(metrics[name]['test'])
            test_regret_mean, test_regret_std, test_regrets = calculate_regret(metrics[name]['test'])
            pd_dict['test_rl_auc'].append(test_auc_mean)
            pd_dict['test_rl_regret'].append(test_regret_mean)
            pd_dict['test_auc_std'].append(test_auc_std)
            pd_dict['test_regret_std'].append(test_regret_std)
            pd_dict['test_regrets'].append(test_regrets)
    df = pd.DataFrame(data=pd_dict, index=pd_index)
    correlation_method = 'pearson'
    # printing correlation of codebook description length and other metrics
    for column in df.columns:
        if column != 'codebook_dl' and column not in length_set and column != 'test_aucs' and column != 'test_regrets':
            correlation = df['codebook_dl'].corr(df[column], method=correlation_method)
            print(column, correlation)
    """
    if track_probs:
        #printing correlation of frequency of skill length and all metrics
        for col1 in df.columns:
            if col1 in length_set:
                for col2 in df.columns:
                    if col2 not in length_set:
                        correlation = df[col1].corr(df[col2])
                        print(col1, col2, correlation)
    """
    # correlation between primitive and abstract actions
    #for col1 in df.columns:
    #    if "train" in col1:
    #        for col2 in df.columns:
    #            if "train" in col2 and (col1 != col2):
    #                correlation = df[col1].corr(df[col2], method=correlation_method)
    #                print(col1, col2, correlation)
    df.to_csv(os.path.join(args.location, 'analysis_new.csv'))