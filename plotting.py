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
import imageio
from collect_data import rollout
from ast import literal_eval


analysis_file = os.path.join(os.getcwd(), 'data/rerun/analysis_new.csv')
df = pd.read_csv(analysis_file)
# remove unnecessary columns for debug vis
df_vis = df.drop(['train_num_primitive_actions', 'train_num_abstract_actions', 'train_code_length',
              'train_description_length', 'train_node_cost', 'test_num_primitive_actions',
              'test_num_abstract_actions', 'test_code_length', 'test_description_length',
              'test_auc_std', 'test_regret_std', 'num_symbols', 'test_rl_auc'], axis=1)
correlation_method = 'pearson'
show_img = False

# # DL against RL
dropped = [0,1,2,6,8,9,12,14,19,21]  # outliers
df_rl = df.drop(dropped)
x1 = df_rl['codebook_dl']
y1 = df_rl['test_rl_regret']
v1 = df_rl['test_regret_std']
v1 = v1/np.sqrt(8)  # convert SE
rs = df_rl['test_regrets'].apply(literal_eval).to_numpy()
rs = [np.array(r) for r in rs]

correlation1 = x1.corr(y1, method=correlation_method)
print(f'Correlation between DL and RL: {correlation1}')

plt.figure()
b1, m1 = polyfit(x1, y1, 1)
plt.scatter(x1, y1)
# plt.scatter(x1, y1+v1)
# plt.scatter(x1, y1-v1)
plt.vlines(x1, y1-v1, y1+v1)
plt.hlines(y1-v1, x1-150, x1+150)
plt.hlines(y1+v1, x1-150, x1+150)
x = np.linspace(33000,46000)
plt.plot(x, b1 + m1 * x, '--', c='#ff7f0e')
locs, _ = plt.xticks()
locs = locs[::2]
plt.xticks(locs, fontsize=16)
locs, _ = plt.yticks()
locs = locs[::3]
plt.yticks(locs, fontsize=16)
plt.xlim(33000, 46000)
plt.xlabel('Description Length', fontsize=16)
plt.ylabel('Regret', fontsize=16)
plt.title(f'Correlation between DL and RL Regret: {correlation1:.2f}', fontsize=16)
if show_img:
    plt.show()
else:
    plt.tight_layout()
    plt.savefig('regret_se_caption.pdf')

# plt.figure()
# plt.boxplot(rs, showfliers=False)
# plt.show()
# plt.savefig('regret.png')

# DL against augmented A*
df_a_star = df.drop(dropped)
x2 = df_a_star['codebook_dl']
y2 = df_a_star['test_node_cost']
b2, m2 = polyfit(x2, y2, 1)
correlation2 = x2.corr(y2, method=correlation_method)
print(f'Correlation between DL and A*: {correlation2}')

plt.figure()
plt.scatter(x2, y2)
plt.plot(x, b2 + m2 * x, '--', c='#ff7f0e')
locs, _ = plt.xticks()
locs = locs[::2]
plt.xticks(locs, fontsize=16)
locs, _ = plt.yticks()
locs = locs[::2]
plt.yticks(locs, fontsize=16)
plt.xlim(33000, 46000)
plt.xlabel('Description Length', fontsize=16)
plt.ylabel('A* Search Cost', fontsize=16)
plt.title(f'Correlation between DL and A* Cost: {correlation2:.2f}', fontsize=16)
if show_img:
    plt.show()
else:
    plt.tight_layout()
    plt.savefig(f'a_star_caption.pdf')

'''
# qualitative examples
good_cb_name = 'code_book4_6.npy'
bad_cb_name = 'code_book3_4_8.npy'
with open(os.path.join(os.getcwd(), 'data/rerun', good_cb_name), 'rb+') as f:
    good_cb = np.load(f, allow_pickle=True).item()
with open(os.path.join(os.getcwd(), 'data/rerun', bad_cb_name), 'rb+') as f:
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
total_exps = 1
while count <= total_exps:
    figure_dir = os.path.join(os.getcwd(), f'figures/rerun/exp{count}')
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    env.reset()
    env_copy = deepcopy(env)

    good_sol, good_cost, good_search_path = a_star(env, codebook=good_cb, save_search_path=True)
    bad_sol, bad_cost, bad_search_path = a_star(env, codebook=bad_cb, save_search_path=True)
    print(f'good codebook cost: {good_cost}, bad codebook cost: {bad_cost}')
    if bad_cost - good_cost < 50 or len(good_search_path) < 100:
        continue

    env_copy_2 = deepcopy(env)
    skill_path = rollout(env_copy_2, good_sol)
    for x,y,skill in skill_path:
        env_copy_2.set_path(x,y,skill)
    img = env_copy_2.render('rgb_array', visible_mask=False)
    window.show_img(img)
    window.save_img(figure_dir, f'good_sol_skills.png')

    env_copy_3 = deepcopy(env)
    skill_path = rollout(env_copy_3, bad_sol)
    for x,y,skill in skill_path:
        env_copy_3.set_path(x,y,skill)
    img = env_copy_3.render('rgb_array', visible_mask=False)
    window.show_img(img)
    window.save_img(figure_dir, f'bad_sol_skills.png')

    sep = 30
    i, j = 0, 0
    good_images, bad_images = [], []
    while True:
        img = env.render('rgb_array', visible_mask=False)
        window.show_img(img)
        window.save_img(figure_dir, f'good_frame_{int(i/sep + 1)}.png')
        good_images.append(imageio.imread(os.path.join(figure_dir, f'good_frame_{int(i/sep + 1)}.png')))
        good_chunk = good_search_path[i:i + sep]
        env.add_heat(good_chunk)
        if i > len(good_search_path):
            break
        i += sep

    while True:
        img = env_copy.render('rgb_array', visible_mask=False)
        window.show_img(img)
        window.save_img(figure_dir, f'bad_frame_{int(j/sep + 1)}.png')
        bad_images.append(imageio.imread(os.path.join(figure_dir, f'bad_frame_{int(j/sep + 1)}.png')))
        bad_chunk = bad_search_path[j:j + sep]
        env_copy.add_heat(bad_chunk)
        if j > len(bad_search_path):
            break
        j += sep

    imageio.mimsave(os.path.join(figure_dir, 'good.gif'), good_images, duration=0.5)
    imageio.mimsave(os.path.join(figure_dir, 'bad.gif'), bad_images, duration=0.5)

    for file in os.listdir(figure_dir):
        if 'frame' in file:
            os.remove(os.path.join(figure_dir, file))

    print(f'saved {count}th exp')
    count += 1
'''