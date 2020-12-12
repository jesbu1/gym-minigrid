import gym_minigrid
import multiprocessing as mp
from gym_minigrid.wrappers import *
import gym
from heapq import heappush, heappop
from itertools import count, product
import copy
import os
import random
import datetime as dt
import numpy as np
from gym_minigrid.window import Window
from calculate_mdl import preprocess_codebook, discover_codebooks

NUM_PARALLEL_THREADS = 12

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


def a_star(env, skills=None, codebook=None):

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

    if skills is None and codebook is None:
        skills = [0, 1, 2]  # primitive actions: left, right, forward
    elif skills is None and codebook is not None:
        skills = [list(map(int, skill)) for skill in codebook.keys()]  # convert '1011' to [1,0,1,1]

    cost = 0

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
            if isinstance(action, int): # primitive actions
                new_entry = one_step(curr_entry, action)
                if not is_valid(new_entry):
                    new_entry = curr_entry
            elif isinstance(action, list):
                new_entry = curr_entry
                for a in action: # action: [a1, a2, ...]
                    next_entry = one_step(new_entry, a)
                    if is_valid(next_entry):
                        new_entry = next_entry

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
                cost += 1
                open_dict[new_entry] = new_f
                heappush(open, [new_f, next(counter), (new_entry, new_seq)])

    return solution, cost

def a_star_parallel(env, out_q, skills=None, codebook=None, name=None):

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

    if skills is None and codebook is None:
        skills = [0, 1, 2]  # primitive actions: left, right, forward
    elif skills is None and codebook is not None:
        skills = [list(map(int, skill)) for skill in codebook.keys()]  # convert '1011' to [1,0,1,1]

    cost = 0

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
            if isinstance(action, int): # primitive actions
                new_entry = one_step(curr_entry, action)
                if not is_valid(new_entry):
                    new_entry = curr_entry
            elif isinstance(action, list):
                new_entry = curr_entry
                for a in action: # action: [a1, a2, ...]
                    next_entry = one_step(new_entry, a)
                    if is_valid(next_entry):
                        new_entry = next_entry

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
                cost += 1
                open_dict[new_entry] = new_f
                heappush(open, [new_f, next(counter), (new_entry, new_seq)])

    output_dict = {name: (solution, cost)}
    out_q.put(output_dict)

def show_init(env, count=1):
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


def collect_trajectories(env, skills=None, num=50, show=False, print_every=50):
    start_time = dt.datetime.now()
    data = []
    count = 0
    while count < num:
        env.reset()
        if training_valid(env):
            count += 1
        else:
            continue

        if show:
            show_init(env, count=count)

        actions, _ = a_star(env, skills=skills)
        trajectory = Trajectory(actions, env)
        data.append(trajectory)

        # print('Collecting %dth trajectory...' % (count))
        # print('Actions (len=%d):' % len(actions), actions)

        if count % print_every == 0 and count != 0:
            curr_time = dt.datetime.now()
            time = (curr_time - start_time).total_seconds()
            print('Finished collecting %d trajectories, time elapsed = %f sec' % (count, time))

    return data


class Trajectory:

    def __init__(self, actions, env, simulate=False):
        self.env = env
        self.actions = actions

        self.cut_tail = ''  # stores cut tail after split_actions
        self.action_str = ''
        for action in self.actions:
            if isinstance(action, int):  # primitive actions
                break
            elif isinstance(action, list):
                for a in action:
                    self.action_str += str(a)
                self.action_str += ' '

        start_pos = env.agent_pos
        self.goal_pos = env.goal_pos

        start_entry = (start_pos[0], start_pos[1], env.agent_dir)
        self.start_pos = start_entry
        self.agent_states = [start_entry]
        self.rewards = []
        self.done = False

        curr_entry = start_entry

        for action in actions:
            curr_agent_states = []
            curr_rewards = []

            if isinstance(action, int):  # primitive actions

                if simulate:
                    new_entry = one_step(curr_entry, action)
                    if not is_valid(new_entry):
                        new_entry = curr_entry
                    curr_agent_states.append(new_entry)
                    curr_entry = new_entry

                    if curr_entry[:2] == self.goal_pos:
                        curr_rewards.append(1)
                        self.done = True
                    else:
                        curr_rewards.append(0)

                else:
                    obs, reward, done, info = env.step(action)
                    curr_rewards.append(reward)
                    curr_pos = env.agent_pos
                    curr_dir = env.agent_dir
                    curr_entry = (curr_pos[0], curr_pos[1], curr_dir)
                    curr_agent_states.append(curr_entry)

                    if done:
                        self.done = True

            elif isinstance(action, list):
                new_entry = curr_entry
                for a in action:

                    if simulate:
                        next_entry = one_step(new_entry, a)
                        if is_valid(next_entry):
                            new_entry = next_entry
                        curr_agent_states.append(new_entry)
                        curr_entry = new_entry

                        if curr_entry[:2] == self.goal_pos:
                            curr_rewards.append(1)
                            self.done = True
                            break
                        else:
                            curr_rewards.append(0)

                    else:
                        obs, reward, done, info = env.step(a)
                        curr_rewards.append(reward)
                        curr_pos = env.agent_pos
                        curr_dir = env.agent_dir
                        curr_entry = (curr_pos[0], curr_pos[1], curr_dir)
                        curr_agent_states.append(curr_entry)

                        if done:
                            self.done = True
                            break

            self.rewards.append(curr_rewards[-1])
            self.agent_states.append(curr_agent_states[-1])

            if self.done:
                break

        # print('Agent states (len=%d):' % len(self.agent_states), self.agent_states)
        # print('Agent rewards (len=%d):' % len(self.rewards), self.rewards, '\n')
        # assert len(self.agent_states) - 1 == len(self.rewards) == len(self.actions) \
        #        and self.rewards[-1] > 0, 'something wrong'

    def __str__(self):
        return self.action_str

    def split_actions(self, skill_length_range, biased_probabilities):
        self.action_str = ''
        l = len(self.actions)
        index = 0
        while 1:
            if index >= l:
                break
            length = np.random.choice(skill_length_range, p=biased_probabilities)
            skill = self.actions[index:index+length]
            if len(skill) in skill_length_range:
                skill = ''.join(map(str, skill))
                self.action_str += skill + ' '
            else:
                skill = ''.join(map(str, skill))
                self.cut_tail += skill + ' '
            index += length


def build_codebook_method_1(trajectories, complete_skills):
    # code_book: {
    #     'trajectories': list of str representations of all trajectories,
    #     skill in str form: num of occurrences
    # }
    code_book = {'trajectories': []}

    for trajectory in trajectories:
        code_book['trajectories'].append(str(trajectory))

        actions = trajectory.actions
        for action in actions:
            action = ''.join(map(str, action))
            if action not in code_book:
                code_book[action] = 1
            else:
                code_book[action] += 1

    for skill in complete_skills:
        skill = ''.join(map(str, skill))
        if skill not in code_book:
            code_book[skill] = 0

    return code_book


def build_codebook_method_2(trajectories, skill_length_range):
    # code_book: {
    #     'trajectories': list of str representations of all trajectories,
    #     skill in str form: num of occurrences,
    # }
    code_book = {'trajectories': []}

    v = np.random.random_sample(len(skill_length_range))
    biased_probabilities = v / np.linalg.norm(v, ord=1)  # normalize

    code_book['length_range'] = skill_length_range
    code_book['probabilities'] = biased_probabilities

    for trajectory in trajectories:

        trajectory.split_actions(skill_length_range, biased_probabilities)
        actions = trajectory.action_str
        code_book['trajectories'].append(actions)

        for action in actions.split(' ')[:-1]:  # exclude '' at the end
            if action not in code_book:
                code_book[action] = 1
            else:
                code_book[action] += 1

    return code_book


def generate_skills(primitive_actions=None, length_range=None, max_action_num=200):
    if primitive_actions is None:
        primitive_actions = [0, 1, 2]
    if length_range is None:
        length_range = range(2, 7) # default lengths: from 2 to 6
    skills = []
    for l in length_range:
        curr_skills = list(product(primitive_actions, repeat=l))
        curr_skills = [list(skill) for skill in curr_skills]
        skills.extend(curr_skills)
    random.shuffle(skills)
    size = random.randint(1, max_action_num if len(skills)>max_action_num else len(skills))
    skills = random.sample(skills, size)
    return skills


def evaluate_solution(solution, env):
    # evaluate a solutionon the start env
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
        if isinstance(action, int):  # primitive actions
            obs, reward, done, info = env.step(action)
            curr_rewards.append(reward)
            curr_pos = env.agent_pos
            curr_dir = env.agent_dir
            curr_entry = (curr_pos[0], curr_pos[1], curr_dir)
            curr_agent_states.append(curr_entry)

            if done:
                is_done = True

        elif isinstance(action, list):
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

def evaluate_codebook_parallel(env, codebooks, num_test=500, num_train=500, print_every=50):
    """
    input:
        env: env variable
        codebooks: pre-processed codebook to evaluate
        num_test: number of test start/end pairs to evaluate,
        num_train: number of train start/end pairs to evaluate

    output:
        solutions: dict storing num_test trajectories for test set, num_train trajectories for train set
            for each codebook in codebooks
            entry form: (str form of trajectory, a_star search cost, start_pos, goal_pos)
    """

    start_time = dt.datetime.now()
    count_train, count_test, count = 0, 0, 0
    solutions = {}
    skills = {}
    for file, codebook in codebooks:
        solutions[file] = {'test': [], 'train': []}
        skills[file] = [list(map(int, skill)) for skill in codebook.keys()]
    while 1:
        if count_train >= num_train and count_test >= num_test:
            break
        env.reset()

        if not training_valid(env) and count_test < num_test:  # in test set
            out_q = mp.Queue()
            procs = []
            result_dict = {}
            for i, (file, codebook) in enumerate(codebooks):
                if i % NUM_PARALLEL_THREADS == 0:
                    for _ in procs:
                        result_dict.update(out_q.get())
                    for proc in procs:
                        proc.join()
                    procs = []
                    
                p = mp.Process(
                    target=a_star_parallel,
                    kwargs={'env':env, 'out_q':out_q, 'skills':skills[file], 'name':file},
                )
                procs.append(p)
                p.start()
            
            for _ in procs:
                result_dict.update(out_q.get())
            for proc in procs:
                proc.join()
            for file, (solution, cost) in result_dict.items():
                traj = Trajectory(solution, env, simulate=True)  # simulate without interacting with env
                solutions[file]['test'].append((str(traj), cost, traj.start_pos, traj.goal_pos))
            count_test += 1
        elif training_valid(env) and count_train < num_train:  # in train set
            out_q = mp.Queue()
            procs = []
            result_dict = {}
            for i, (file, codebook) in enumerate(codebooks):
                if i % NUM_PARALLEL_THREADS == 0:
                    for _ in procs:
                        result_dict.update(out_q.get())
                    for proc in procs:
                        proc.join()
                    procs = []
                    
                p = mp.Process(
                    target=a_star_parallel,
                    kwargs={'env':env, 'out_q':out_q, 'skills':skills[file], 'name':file},
                )
                procs.append(p)
                p.start()
            
            for _ in procs:
                result_dict.update(out_q.get())
            for proc in procs:
                proc.join()
            for file, (solution, cost) in result_dict.items():
                traj = Trajectory(solution, env, simulate=True)
                solutions[file]['train'].append((str(traj), cost, traj.start_pos, traj.goal_pos))
            count_train += 1

        count += 1
        if count % print_every == 0 and count != 0:
            curr_time = dt.datetime.now()
            time = (curr_time - start_time).total_seconds()
            print('Total env tried: %d, test trajectories collected = %d, train trajectories collected = %d, time elapsed = %f sec'
                  % (count, count_test, count_train, time))

    return solutions

def evaluate_codebook(env, codebooks, num_test=500, num_train=500, print_every=50):
    """
    input:
        env: env variable
        codebooks: pre-processed codebook to evaluate
        num_test: number of test start/end pairs to evaluate,
        num_train: number of train start/end pairs to evaluate

    output:
        solutions: dict storing num_test trajectories for test set, num_train trajectories for train set
            for each codebook in codebooks
            entry form: (str form of trajectory, a_star search cost, start_pos, goal_pos)
    """

    start_time = dt.datetime.now()
    count_train, count_test, count = 0, 0, 0
    solutions = {}
    skills = {}
    for file, codebook in codebooks:
        solutions[file] = {'test': [], 'train': []}
        skills[file] = [list(map(int, skill)) for skill in codebook.keys()]
    while 1:
        if count_train >= num_train and count_test >= num_test:
            break
        env.reset()

        if not training_valid(env) and count_test < num_test:  # in test set
            for file, codebook in codebooks:
                solution, cost = a_star(env, skills=skills[file])
                traj = Trajectory(solution, env, simulate=True)  # simulate without interacting with env
                solutions[file]['test'].append((str(traj), cost, traj.start_pos, traj.goal_pos))
            count_test += 1
        elif training_valid(env) and count_train < num_train:  # in train set
            for file, codebook in codebooks:
                solution, cost = a_star(env, skills=skills[file])
                traj = Trajectory(solution, env, simulate=True)
                solutions[file]['train'].append((str(traj), cost, traj.start_pos, traj.goal_pos))
            count_train += 1

        count += 1
        if count % print_every == 0 and count != 0:
            curr_time = dt.datetime.now()
            time = (curr_time - start_time).total_seconds()
            print('Total env tried: %d, test trajectories collected = %d, train trajectories collected = %d, time elapsed = %f sec'
                  % (count, count_test, count_train, time))

    return solutions


def collect_data_method1(env, data_folder, seed=None, num_code_books=10, num_trajectories=1000):
    """
    Method 1: randomly define skills, collect A* trajectories with
    these skills , save the trajectories and skill frequencies as codebook
    """

    if seed is not None:
        env.seed(seed)

    for i in range(num_code_books):

        skills = generate_skills()
        print('Generated skills (len=%d):' % len(skills), skills)

        trajectories = collect_trajectories(env, skills, num=num_trajectories, show=False)
        code_book = build_codebook_method_1(trajectories, skills)

        # print(code_book)
        path = os.path.join(data_folder, 'code_book'+str(i+1)+'.npy')
        np.save(path, code_book)
        print('Codebook saved to %s' % path)

        # cb = np.load(path, allow_pickle=True).item()
        # print(cb)


def collect_data_method2(env,
                         data_folder,
                         skill_length_range,
                         num_skills,
                         seed=None,
                         num_code_books=20,
                         num_trajectories=100):
    """
    Method 2: collect A* trajectories with only primitive actions, randomly dissect trajectories
    to form skills, save the trajectories and skill frequencies as codebook
    """

    if seed is not None:
        env.seed(seed)

    trajectories = collect_trajectories(env, num=num_trajectories, show=False)

    count = 0
    while count < num_code_books:

        code_book = build_codebook_method_2(trajectories, skill_length_range)
        # print(code_book)

        if len(code_book) - 2 == num_skills:  # minus 2: 'length_range' & 'probabilities'
            path = os.path.join(data_folder, 'code_book' + str(count + 1) + '.npy')
            np.save(path, code_book)
            print('Codebook saved to %s' % path)

            count += 1



def evaluate_data(env, data_folder, seed=None):
    """Evaluate codebooks"""

    if seed is not None:
        env.seed(seed)

    codebooks_pre = discover_codebooks(data_folder)

    codebooks = [(file_name, preprocess_codebook(codebook)[1]) for file_name, codebook in codebooks_pre]
    solutions = evaluate_codebook_parallel(env, codebooks)
    for file, dict in solutions.items():
        path = os.path.join(data_folder, 'evaluations', 'trajectories_' + file)
        np.save(path, dict)
        print('Trajectories saved to %s' % path)

    # files = [file for file, _ in codebooks_pre]
    # for file in files:
    #     path = os.path.join(data_folder, 'evaluations', 'trajectories_' + file)
    #     dict = np.load(path, allow_pickle=True).item()
    #     print(dict)


if __name__ == "__main__":

    env = gym.make("MiniGrid-FourRooms-v0")
    env = GoalPositionWrapper(env)
    # show_init(env)

    data_folder = './data/method4'

    # collect_data_method2(env, data_folder, range(2,7), 50, num_code_books=40)
    evaluate_data(env, data_folder)
