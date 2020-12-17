import itertools
import random
import subprocess
import os
from absl import logging, flags, app
from multiprocessing import Queue, Manager
from pathos import multiprocessing
import traceback
import time
import sys
import argparse
from calculate_mdl import preprocess_codebook
from collect_data import run_rl
import numpy as np


def _init_device_queue(which_gpus, max_worker_num):
    m = Manager()
    device_queue = m.Queue()
    for i in range(max_worker_num):
        idx = i % len(which_gpus)
        gpu = which_gpus[idx]
        device_queue.put(gpu)
    return device_queue


def run(which_gpus, max_worker_num, data_folder, train, num_seeds, num_actions):

    process_pool = multiprocessing.Pool(
        processes=max_worker_num, maxtasksperchild=1)
    device_queue = _init_device_queue(which_gpus, max_worker_num)

    for file in os.listdir(data_folder):
        for iteration in range(num_seeds):
            if file.endswith('.npy'):
                process_pool.apply_async(
                    func=_worker,
                    args=[data_folder, file, train, device_queue, num_actions],
                    error_callback=lambda e: logging.error(e))
    process_pool.close()
    process_pool.join()


def _worker(data_folder, file_name, train, device_queue, num_actions):
    try:
        time.sleep(random.uniform(0, 3))
        gpu_id = device_queue.get()

        # load codebook
        codebook = np.load(os.path.join(data_folder, file_name), allow_pickle=True).item()
        _, codebook = preprocess_codebook(codebook)
        codebook_sorted = dict(sorted(codebook.items(), key=lambda item: item[1], reverse=True))
        codebook_clipped = dict(list(codebook_sorted.items())[:num_actions])
        skills = [list(map(int, skill)) for skill in codebook.keys()]

        # run experiment
        experiment_name = 'rl_' + file_name.replace('.npy', '')

        name_append = 'train' if train else 'test'
        run_rl(experiment_name, os.path.join(os.getcwd(), data_folder, f'rl_logs_{name_append}', experiment_name), train, skills, gpu_id)

        device_queue.put(gpu_id)

    except Exception as e:
        logging.info(traceback.format_exc())
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--which_gpus',
        help='used gpus',
        default=[0, 1, 2]
    )
    parser.add_argument(
        '--data_folder',
        help='codebooks folder',
        type=str,
        default='data/method6'
    )
    parser.add_argument(
        '--train',
        help='train mode on',
        action='store_true'
    )
    parser.add_argument(
        '--num_actions',
        help='number of top actions to use',
        type=int,
        default=15
    )
    parser.add_argument(
        '--num_seeds',
        help='number of times to train each codebook',
        type=int,
        default=3
    )
    args = parser.parse_args()

    max_worker_num = len(args.which_gpus) * 3
    run(args.which_gpus, max_worker_num, args.data_folder, args.train, args.num_seeds, args.num_actions)