import os
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit
import numpy as np
from calculate_mdl import preprocess_codebook
from gym_minigrid.window import Window
import gym
from collect_data import a_star


analysis_file = os.path.join(os.getcwd(), 'data/method6/analysis.csv')
df = pd.read_csv(analysis_file)
correlation_method = 'pearson'

# DL against RL
x1 = df['codebook_dl']
y1 = df['test_rl_auc']
correlation1 = x1.corr(y1, method=correlation_method)
print(f'Correlation between DL and RL: {correlation1}')

b1, m1 = polyfit(x1, y1, 1)
plt.figure()
plt.plot(x1, y1, '.')
plt.plot(x1, b1 + m1 * x1, '--')
plt.xlabel('Codebook DL')
plt.ylabel('Area Under Curve (AUC)')
plt.title(f'Correlation between DL and RL agent: {correlation1}')
plt.show()
# plt.savefig('dl_rl_correlation.png')

# DL against augmented A*
y2 = df['test_node_cost']
b2, m2 = polyfit(x1, y2, 1)
correlation2 = x1.corr(y2, method=correlation_method)
print(f'Correlation between DL and A*: {correlation2}')

plt.figure()
plt.plot(x1, y2, '.')
plt.plot(x1, b2 + m2 * x1, '--')
plt.xlabel('Codebook DL')
plt.ylabel('Search Cost (node expanded)')
plt.title(f'Correlation between DL and A* search: {correlation2}')
plt.show()
# plt.savefig('dl_a_star_correlation.png')

# qualitative examples
good_figure_dir = os.path.join(os.getcwd(), 'figures/gif/good/')
bad_figure_dir = os.path.join(os.getcwd(), 'figures/gif/bad/')
good_cb_name = 'code_book4.npy'
bad_cb_name = 'code_book7.npy'
with open(os.path.join(os.getcwd(), 'data/method6', good_cb_name), 'rb+') as f:
    good_cb = np.load(f, allow_pickle=True).item()
with open(os.path.join(os.getcwd(), 'data/method6', bad_cb_name), 'rb+') as f:
    bad_cb = np.load(f, allow_pickle=True).item()
# preprocess codebook
_, good_cb = preprocess_codebook(good_cb)
_, bad_cb = preprocess_codebook(bad_cb)
# clip the first 15 skills
good_cb = dict(sorted(good_cb.items(), key=lambda item: item[1], reverse=True))
good_cb = dict(list(good_cb.items())[:15])
bad_cb = dict(sorted(bad_cb.items(), key=lambda item: item[1], reverse=True))
bad_cb = dict(list(bad_cb.items())[:15])

def run_and_save(codebook, image_dir):

    def redraw():
        img = env.render('rgb_array', tile_size=32)
        window.show_img(img)
        window.save_img(image_dir, env.step_count)
        # print(f'Saving image_{env.step_count}.png to {image_dir}')

    def reset():
        seed = np.random.randint(0,10000)
        env.seed(seed)
        env.reset()
        redraw()

    def step(action):
        _, _, done, _ = env.step(action)
        if done:
            reset()
        else:
            redraw()

    skills = list(codebook.keys())

    env = gym.make('MiniGrid-FourRoomsSkills-v0', train=False, skills=skills, visualize=True)
    window = Window('gym_minigrid - MiniGrid-FourRoomsSkills-v0')
    reset()

    solution, cost = a_star(env, codebook=codebook)
    print(cost)

run_and_save(good_cb, good_figure_dir)