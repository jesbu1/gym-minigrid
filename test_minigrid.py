import gym_minigrid
from gym_minigrid.wrappers import *
import gym
from heapq import heappush, heappop
from itertools import count
import copy
import numpy as np
from gym_minigrid.window import Window

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


def finished(entry, goal_pos):
    return entry[0]==goal_pos[0] and entry[1]==goal_pos[1]


def is_valid(entry):
    return env.grid.is_valid(entry[0], entry[1])


def one_step(curr_entry, action):
    new_entry = None
    if action == 0:
        new_entry = (curr_entry[0], curr_entry[1], 3 if curr_entry[2] == 0 else curr_entry[2] - 1)
    elif action == 1:
        new_entry = (curr_entry[0], curr_entry[1], 0 if curr_entry[2] == 3 else curr_entry[2] + 1)
    elif action == 2:
        curr_dir = curr_entry[2]
        if curr_dir == 0:
            new_entry = (curr_entry[0] + 1, curr_entry[1], curr_entry[2])
        elif curr_dir == 1:
            new_entry = (curr_entry[0], curr_entry[1] + 1, curr_entry[2])
        elif curr_dir == 2:
            new_entry = (curr_entry[0] - 1, curr_entry[1], curr_entry[2])
        else:
            new_entry = (curr_entry[0], curr_entry[1] - 1, curr_entry[2])
    return new_entry


def a_star(env, skills=None):

    agent_pos = env.agent_pos
    goal_pos = env.goal_pos
    agent_dir = env.agent_dir

    open = []
    closed = []
    solution = None

    init_entry = (agent_pos[0], agent_pos[1], agent_dir)
    init_h = get_h(init_entry, goal_pos)
    init_f = 0 + init_h
    # print('Agent and goal position:', init_entry, goal_pos)

    counter = count()

    heappush(open, [init_f, next(counter), (init_entry, [])])  # f, counter, ((x, y, dir), action_seq)
    open_dict = {init_entry: init_f}  # (x, y, dir) => f

    if not skills:
        skills = [0, 1, 2]  # primitive actions: left, right, forward

    while open:

        curr_f, _, (curr_entry, curr_seq) = heappop(open)
        curr_h = get_h(curr_entry, goal_pos)
        curr_g = curr_f - curr_h

        if curr_entry in closed:
            continue

        closed.append(curr_entry)

        if finished(curr_entry, goal_pos):
            solution = curr_seq
            break

        for action in skills:
            new_entry = None
            if isinstance(action, int):
                new_entry = one_step(curr_entry, action)
                if not is_valid(new_entry):
                    new_entry = curr_entry
                    break
            elif isinstance(action, list):
                new_entry = curr_entry
                for a in action: # action: [a1, a2, ...]
                    new_entry = one_step(new_entry, a)
                    if not is_valid(new_entry):
                        new_entry = curr_entry
                        break

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


def show_init(count=1):
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
            show_init(count=count)

        # print('Collecting %dth trajectory...' % (count))
        actions = a_star(env)
        # print('Actions (len=%d):' % len(actions), actions)
        trajectory = Trajectory(actions, env)
        data.append(trajectory)
    return data


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

        # print('Agent states (len=%d):' % len(self.agent_states), self.agent_states)
        # print('Agent rewards (len=%d):' % len(self.rewards), self.rewards, '\n')


def random_cut_up(actions, skill_length):
    l = len(actions)
    if l < skill_length:
        return []
    while 1:
        start = np.random.randint(0,l)
        end = np.random.randint(0,l)
        if abs(end-start) >= skill_length-1:
            return actions[min(start,end):max(end,start)+1]


def build_codebook(trajectories, skill_length=2):
    skills = {}
    for trajectory in trajectories:
        curr_skills = {}
        actions = random_cut_up(trajectory.actions, skill_length)
        index = 0
        while 1:
            if index + skill_length > len(actions):
                break
            skill = tuple(actions[index:index+skill_length])
            if skill in curr_skills:
                curr_skills[skill] += 1
            else:
                curr_skills[skill] = 1
            index += skill_length

        for key, value in curr_skills.items():
            if key in skills:
                skills[key] += value
            else:
                skills[key] = value
    return skills


def evaluate_solution(solution, env):
    # evaluate a solution (of skills) on the start env
    # return true if solution solves the task

    start_pos = env.agent_pos
    start_dir = env.agent_dir
    goal_pos = env.goal_pos

    start_entry = (start_pos[0], start_pos[1], start_dir)
    agent_states = [start_entry]
    rewards = []
    is_done = False

    for action in solution:
        curr_rewards = []
        curr_agent_states = []
        for a in action:
            obs, reward, done, info = env.step(a)
            curr_rewards.append(reward)
            curr_pos = env.agent_pos
            curr_dir = env.agent_dir
            curr_entry = (curr_pos[0], curr_pos[1], curr_dir)
            curr_agent_states.append(curr_entry)

            if done:
                is_done = True
                break

        rewards.append(curr_rewards[-1])
        agent_states.append(curr_agent_states[-1])

        if is_done:
            break

    return len(agent_states)-1 == len(rewards) == len(solution) and rewards[-1] > 0 and is_done


if __name__ == "__main__":

    env = gym.make("MiniGrid-FourRooms-v0")
    env = GoalPositionWrapper(env)

    trajectories = collect_trajectories(env, num=50, show=False)
    code_book = build_codebook(trajectories, skill_length=2)

    print(code_book)
    np.save('./data/code_book_2.npy', code_book)
    cb = np.load('./data/code_book_2.npy', allow_pickle=True).item()
    print(cb)

    skills = cb.keys()
    skills = [list(skill) for skill in skills]  # convert tuples to lists
    print(skills)

    env.reset()
    show_init()
    solution = a_star(env, skills=skills)
    print(evaluate_solution(solution, env))
