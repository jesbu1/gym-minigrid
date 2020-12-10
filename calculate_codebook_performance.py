import numpy as np
from calculate_mdl import *
import argparse
import os
import pandas as pd

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
    for codebook in codebooks:
        length_range = codebook[1].pop('length_range')
        probabilities = codebook[1].pop('probabilities')
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
    trajectory_dict = dict(train={}, test={}, probabilities={})
    for codebook_name, evaluation in evaluations:
        original_name = codebook_name.replace("trajectories_", "")
        codebook_info = codebook_dict[original_name]
        process_evaluation(evaluation, codebook_info['codec'], codebook_info['tree_bits'], original_name, trajectory_dict) 
    #print(trajectory_dict.keys(), trajectory_dict['test'].keys())

    aggregate_stats = []

    pd_index = []
    pd_dict = {'codebook_dl': []}
    length_set = set()
    for name, dl, *_ in sorted_codebooks_by_dl:
        pd_index.append(name)
        def accumulate_values(traj_type):
            #values_to_track = ['num_primitive_actions', 'num_abstract_actions', 'code_length', 'description_length', 'node_cost']
            values_to_track = ['node_cost']
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
        for i, length in enumerate(codebook_dict[name]['length_range']):
            length = str(length)
            length_set.add(length)
            if length not in pd_dict:
                pd_dict[length] = []
            pd_dict[length].append(codebook_dict[name]['probabilities'][i])
    df = pd.DataFrame(data=pd_dict, index=pd_index)
    for column in df.columns:
        if column != 'codebook_dl' and column not in length_set:
            correlation = df['codebook_dl'].corr(df[column])
            print(column, correlation)
    for col1 in df.columns:
        if col1 in length_set:
            for col2 in df.columns:
                if col2 not in length_set:
                    correlation = df[col1].corr(df[col2])
                    print(col1, col2, correlation)
    df.to_csv(os.path.join(args.location, 'analysis.csv'))