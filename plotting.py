import os
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit
import numpy as np
from calculate_mdl import preprocess_codebook
from gym_minigrid.window import Window
import gym
from collect_data import a_star
from copy import deepcopy


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
plt.title(f'Correlation between DL and RL Reward AUC: {correlation1:.2f}')
#plt.show()
plt.savefig('figures/dl_rl_correlation.png', bbox_inches="tight")
plt.savefig('figures/dl_rl_correlation.pdf', bbox_inches="tight")

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
plt.title(f'Correlation between DL and A* search: {correlation2:.2f}')
#plt.show()
plt.savefig('figures/dl_a_star_correlation.png', bbox_inches="tight")
plt.savefig('figures/dl_a_star_correlation.pdf', bbox_inches="tight")

# qualitative examples
good_figure_dir = os.path.join(os.getcwd(), 'figures/good_3/')
bad_figure_dir = os.path.join(os.getcwd(), 'figures/bad_3/')
good_cb_name = 'code_book4.npy'
bad_cb_name = 'code_book5.npy'
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

# make a fixed test env with trivial skills
env = gym.make('MiniGrid-FourRoomsSkills-v0', train=False, skills=['0','1','2'], visualize=True)
seed = np.random.randint(0,10000)
env.seed(seed)
window = Window('MiniGrid-FourRoomsSkills-v0')

count = 1
total_frame = 10
while count <= total_frame:
    env.reset()
    env_copy = deepcopy(env)

    good_sol, good_cost, good_search_path = a_star(env, codebook=good_cb, save_search_path=True)
    bad_sol, bad_cost, bad_search_path = a_star(env, codebook=bad_cb, save_search_path=True)
    print(f'good codebook cost: {good_cost}, bad codebook cost: {bad_cost}')
    if bad_cost - good_cost < 50:
        continue

    env.add_heat(good_search_path)
    env_copy.add_heat(bad_search_path)

    img = env.render('rgb_array')
    window.show_img(img)
    window.save_img(good_figure_dir, count)

    img = env_copy.render('rgb_array')
    window.show_img(img)
    window.save_img(bad_figure_dir, count)

    print(f'saved {count}th pair of comparison')
    count += 1