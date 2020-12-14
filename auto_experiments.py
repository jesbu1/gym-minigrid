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
from collect_data import collect_data_rl
import argparse


def _init_device_queue(which_gpus, max_worker_num):
    m = Manager()
    device_queue = m.Queue()
    for i in range(max_worker_num):
        idx = i % len(which_gpus)
        gpu = which_gpus[idx]
        device_queue.put(gpu)
    return device_queue


def run(which_gpus, max_worker_num, data_folder, train):

    process_pool = multiprocessing.Pool(
        processes=max_worker_num, maxtasksperchild=1)
    device_queue = _init_device_queue(which_gpus, max_worker_num)

    for file in os.listdir(data_folder):
        process_pool.apply_async(
            func=collect_data_rl,
            args=[data_folder, file, train, device_queue],
            error_callback=lambda e: logging.error(e))
    process_pool.close()
    process_pool.join()


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
        default='method5'
    )
    parser.add_argument(
        '--train',
        help='train/test mode',
        type=bool,
        default=True
    )
    args = parser.parse_args()

    max_worker_num = len(args.which_gpus) * 3
    run(args.which_gpus, max_worker_num, args.data_folder, args.train)