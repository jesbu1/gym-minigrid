import gym_minigrid
from gym_minigrid.wrappers import *
import gym
from heapq import heappush, heappop
from itertools import count
import copy
import numpy as np
from gym_minigrid.window import Window

env = gym.make("MiniGrid-FourRooms-v0")
env = GoalPositionWrapper(env)


# If reward is nonzero, then we know that we have reached the goal. env is reset automatically.
# Heuristic: manhattan distance to goal assuming we go forward from current direction. Ignore walls for now, and
# remember to include the one timestep we need to make a turn if needed to go to goal.

# import pdb; pdb.set_trace()

# not consistent
def get_h(entry, goal_pos):
    curr_pos = (entry[0], entry[1])
    curr_dir = entry[2]
    hori_disp = goal_pos[0] - curr_pos[0]
    vert_disp = goal_pos[1] - curr_pos[1]
    if hori_disp==0 and vert_disp==0:
        return 0
    retval = abs(hori_disp) + abs(vert_disp)
    hori_sign = np.sign(hori_disp)
    vert_sign = np.sign(vert_disp)

    num_turn_dict = {
        (1,-1,0):1,
        (1,-1,1):2,
        (1,-1,2):2,
        (1,-1,3):1,
        (1,0,0):0,
        (1,0,1):1,
        (1,0,2):2,
        (1,0,3):1,
        (1,1,0):1,
        (1,1,1):1,
        (1,1,2):2,
        (1,1,3):2,
        (0,1,0):1,
        (0,1,1):0,
        (0,1,2):1,
        (0,1,3):2,
        (-1,1,0):2,
        (-1,1,1):1,
        (-1,1,2):1,
        (-1,1,3):2,
        (-1,0,0):2,
        (-1,0,1):1,
        (-1,0,2):0,
        (-1,0,3):1,
        (-1,-1,0):2,
        (-1,-1,1):2,
        (-1,-1,2):1,
        (-1,-1,3):1,
        (0,-1,0):1,
        (0,-1,1):2,
        (0,-1,2):1,
        (0,-1,3):0
    }

    retval += num_turn_dict[(hori_sign, vert_sign, curr_dir)]
    return retval


def done(entry, goal_pos):
    return entry[0]==goal_pos[0] and entry[1]==goal_pos[1]


def is_valid(entry):
    return env.grid.is_valid(entry[0], entry[1])


def a_star(env):

    agent_pos = env.agent_pos
    goal_pos = env.goal_pos
    agent_dir = env.agent_dir

    open = []
    closed = []
    solution = None

    init_entry = (agent_pos[0], agent_pos[1], agent_dir)
    init_h = get_h(init_entry, goal_pos)
    init_f = 0 + init_h
    print('Agent and goal position:', init_entry, goal_pos)

    counter = count()

    heappush(open, [init_f, next(counter), (init_entry, [])])  # f, counter, ((x, y, dir), action_seq)
    open_dict = {init_entry: init_f}  # (x, y, dir) => f
    actions = [0, 1, 2]  # left, right, forward

    while open:

        curr_f, _, (curr_entry, curr_seq) = heappop(open)
        curr_h = get_h(curr_entry, goal_pos)
        curr_g = curr_f - curr_h

        if curr_entry in closed:
            continue

        closed.append(curr_entry)

        if done(curr_entry, goal_pos):
            solution = curr_seq
            break

        for action in actions:

            if action == 0:
                new_entry = (curr_entry[0], curr_entry[1], 3 if curr_entry[2] == 0 else curr_entry[2] - 1)
            elif action == 1:
                new_entry = (curr_entry[0], curr_entry[1], 0 if curr_entry[2] == 3 else curr_entry[2] + 1)
            else:
                curr_dir = curr_entry[2]
                if curr_dir == 0:
                    new_entry = (curr_entry[0] + 1, curr_entry[1], curr_entry[2])
                elif curr_dir == 1:
                    new_entry = (curr_entry[0], curr_entry[1] + 1, curr_entry[2])
                elif curr_dir == 2:
                    new_entry = (curr_entry[0] - 1, curr_entry[1], curr_entry[2])
                else:
                    new_entry = (curr_entry[0], curr_entry[1] - 1, curr_entry[2])

            if not is_valid(new_entry):
                new_entry = curr_entry

            new_h = get_h(new_entry, goal_pos)
            new_g = curr_g + 1
            new_f = new_g + new_h
            new_seq = copy.deepcopy(curr_seq)
            new_seq.append(action)

            if new_entry in open_dict:
                prev_f = open_dict[new_entry]
                if new_f < prev_f:
                    open_dict[new_entry] = new_f
                    heappush(open, [new_f, next(counter), (new_entry, new_seq)])
            elif new_entry not in closed and new_entry not in open_dict:
                open_dict[new_entry] = new_f
                heappush(open, [new_f, next(counter), (new_entry, new_seq)])

    return solution


def show_init(count):
    window = Window('Env No.%d' % (count))
    window.set_caption(env.mission)
    img = env.render('rgb_array', tile_size=32)
    window.show_img(img)
    window.show(block=True)


def in_room(pos):
    # | 4 | 1 |
    # | 3 | 2 |
    w, h = pos
    if w < 9 and h < 9:
        return 4
    elif w < 9 and h > 9:
        return 3
    elif w > 9 and h < 9:
        return 1
    else:
        return 2


# exclude 4 of the 16 starting agent/goal positions
def training_valid(env):
    agent_pos = env.agent_pos
    goal_pos = env.goal_pos
    agent_goal = (in_room(agent_pos), in_room(goal_pos))
    test_set = {
        (1,3),
        (3,1),
        (2,4),
        (4,2)
    }
    return agent_goal not in test_set


def collect_trajectories(env, num=10, show=False):
    data = []
    count = 0
    while count < num:
        obs = env.reset()

        if training_valid(env):
            count += 1
        else:
            continue

        if show:
            show_init(count)

        print('Collecting %dth trajectory...' % (count))
        actions = a_star(env)
        print('Actions (len=%d):' % len(actions), actions)
        trajectory = Trajectory(actions, env)
        data.append(trajectory)

    # data = np.array(data)
    # np.save('data.npy', data)


class Trajectory:

    def __init__(self, actions, env):
        self.actions = actions
        self.env = env

        self.start_pos = env.agent_pos
        self.start_dir = env.agent_dir
        self.goal_pos = env.goal_pos

        start_entry = (self.start_pos[0], self.start_pos[1], self.start_dir)
        self.agent_states = [start_entry]
        self.rewards = []
        self.done = False

        for action in actions:
            obs, reward, done, info = env.step(action)
            self.rewards.append(reward)
            curr_pos = env.agent_pos
            curr_dir = env.agent_dir
            curr_entry = (curr_pos[0], curr_pos[1], curr_dir)
            self.agent_states.append(curr_entry)

            if done:
                self.done = True
                break

        print('Agent states (len=%d):' % len(self.agent_states), self.agent_states)
        print('Agent rewards (len=%d):' % len(self.rewards), self.rewards, '\n')


collect_trajectories(env, num=50, show=True)