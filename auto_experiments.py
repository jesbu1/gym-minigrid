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


def _init_device_queue(which_gpus, max_worker_num):
    m = Manager()
    device_queue = m.Queue()
    for i in range(max_worker_num):
        idx = i % len(which_gpus)
        gpu = which_gpus[idx]
        device_queue.put(gpu)
    return device_queue


def run(which_gpus, max_worker_num, commands, seeds):
    """Run trainings with all possible parameter combinations in
    the configured space.
    """

    process_pool = multiprocessing.Pool(
        processes=max_worker_num, maxtasksperchild=1)
    device_queue = _init_device_queue(which_gpus, max_worker_num)

    for command in commands:
        for seed in seeds:
            command += ' --seed %d' % seed
            process_pool.apply_async(
                func=_worker,
                args=[command, device_queue],
                error_callback=lambda e: logging.error(e))
    process_pool.close()
    process_pool.join()


def _worker(command, device_queue):
    # sleep for random seconds to avoid crowded launching
    try:
        time.sleep(random.uniform(0, 3))

        device = device_queue.get()

        logging.set_verbosity(logging.INFO)

        logging.info("command %s" % command)
        os.system("CUDA_VISIBLE_DEVICES=%d " % device + command)

        device_queue.put(device)
    except Exception as e:
        logging.info(traceback.format_exc())
        raise e


if __name__ == "__main__":

    which_gpus = [0, 1, 2, 3]
    max_worker_num = len(which_gpus) * 3
    COMMANDS = []
    seeds = [123, 1, 12, 73, 309]
    run(which_gpus, max_worker_num, COMMANDS, seeds)