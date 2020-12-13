#!/usr/bin/env python3

import time
import argparse
import numpy as np
import gym
import gym_minigrid
from gym_minigrid.wrappers import *
from gym_minigrid.window import Window

def redraw(img):
    if not args.agent_view:
        img = env.render('rgb_array', tile_size=args.tile_size)

    window.show_img(img)

def reset():
    if args.seed != -1:
        env.seed(args.seed)

    obs_image, obs = env.reset()

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)
        print(obs)

    redraw(obs_image)

def step(action):
    (obs_image, obs), reward, done, info = env.step(action)
    print('step=%s, action_id=%d, action=' % (env.step_count, action)
          + str(env._skills[action]) + ', reward=%.2f' % reward)
    print(obs)

    if done:
        print('done!')
        reset()
    else:
        redraw(obs_image)

def key_handler(event):
    print('pressed', event.key)

    if event.key == 'escape':
        window.close()
        return

    if event.key == 'backspace':
        reset()
        return

    if event.key == 'left':
        step(0)
        return

    if event.key == 'right':
        step(1)
        return

    if event.key == 'up':
        step(2)
        return

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="gym environment to load",
    default='MiniGrid-FourRoomsSkills-v0'
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=-1
)
parser.add_argument(
    "--tile_size",
    type=int,
    help="size at which to render tiles",
    default=32
)
parser.add_argument(
    '--agent_view',
    default=False,
    help="draw the agent sees (partially observable view)",
    action='store_true'
)
parser.add_argument(
    "--train",
    default=False,
    help="training mode on/off"
)

args = parser.parse_args()

skills = [[1,1], [2,2], [2,0]]  # 180, double_forward, forward_left
env = gym.make(args.env, train=args.train, skills=skills, visualize=True)

if args.agent_view:
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

window = Window('gym_minigrid - ' + args.env)
window.reg_key_handler(key_handler)

reset()

# Blocking event loop
window.show(block=True)
