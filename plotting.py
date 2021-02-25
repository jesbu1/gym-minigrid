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


analysis_file = os.path.join(os.getcwd(), 'data/rerun/analysis_new.csv')
df = pd.read_csv(analysis_file)
# df = df.drop([0,2,6])  # dropping high var cbs
correlation_method = 'pearson'

# # DL against RL
x1 = df['codebook_dl']
#y1 = df['test_rl_auc']
y1 = df['test_rl_regret']
v1 = df['test_regret_variance']

# aucs = df['test_aucs'].to_numpy()
# regrets = df['test_regrets'].to_numpy()

correlation1 = x1.corr(y1, method=correlation_method)
print(f'Correlation between DL and RL: {correlation1}')

plt.figure()
# b1, m1 = polyfit(x1, y1, 1)
# plt.plot(x1, y1)
# plt.plot(x1, b1 + m1 * x1, '--')

# plt.plot(x1, y1)
plt.errorbar(x1, y1, v1, marker='.')

plt.xlabel('Codebook DL')
plt.ylabel('Regret')
plt.title(f'Correlation between DL and RL Regret: {correlation1:.2f}')
plt.show()
# plt.savefig('figures/rerun/dl_rl_correlation.png', bbox_inches="tight")
# plt.savefig('figures/rerun/dl_rl_correlation.pdf', bbox_inches="tight")

# # DL against augmented A*
# y2 = df['test_node_cost']
# b2, m2 = polyfit(x1, y2, 1)
# correlation2 = x1.corr(y2, method=correlation_method)
# print(f'Correlation between DL and A*: {correlation2}')
#
# plt.figure()
# plt.plot(x1, y2, '.')
# plt.plot(x1, b2 + m2 * x1, '--')
# plt.xlabel('Codebook DL')
# plt.ylabel('Search Cost (node expanded)')
# plt.title(f'Correlation between DL and A* search: {correlation2:.2f}')
# plt.show()
# # plt.savefig('figures/rerun/dl_a_star_correlation.png', bbox_inches="tight")
# # plt.savefig('figures/rerun/dl_a_star_correlation.pdf', bbox_inches="tight")

# # qualitative examples
# good_cb_name = 'code_book4_6.npy'
# bad_cb_name = 'code_book3_4_8.npy'
# with open(os.path.join(os.getcwd(), 'data/rerun', good_cb_name), 'rb+') as f:
#     good_cb = np.load(f, allow_pickle=True).item()
# with open(os.path.join(os.getcwd(), 'data/rerun', bad_cb_name), 'rb+') as f:
#     bad_cb = np.load(f, allow_pickle=True).item()
# # preprocess codebook
# _, good_cb = preprocess_codebook(good_cb)
# _, bad_cb = preprocess_codebook(bad_cb)
# # clip the first 15 skills
# good_cb = dict(sorted(good_cb.items(), key=lambda item: item[1], reverse=True))
# good_cb = dict(list(good_cb.items())[:15])
# bad_cb = dict(sorted(bad_cb.items(), key=lambda item: item[1], reverse=True))
# bad_cb = dict(list(bad_cb.items())[:15])
#
# # make a fixed test env with trivial skills
# env = gym.make('MiniGrid-FourRoomsSkills-v0', train=False, skills=['0','1','2'], visualize=True)
# seed = np.random.randint(0,10000)
# env.seed(seed)
# window = Window('MiniGrid-FourRoomsSkills-v0')
#
# count = 1
# total_exps = 1
# while count <= total_exps:
#     figure_dir = os.path.join(os.getcwd(), f'figures/rerun/exp{count}')
#     if not os.path.exists(figure_dir):
#         os.makedirs(figure_dir)
#
#     env.reset()
#     env_copy = deepcopy(env)
#
#     good_sol, good_cost, good_search_path = a_star(env, codebook=good_cb, save_search_path=True)
#     bad_sol, bad_cost, bad_search_path = a_star(env, codebook=bad_cb, save_search_path=True)
#     print(f'good codebook cost: {good_cost}, bad codebook cost: {bad_cost}')
#     if bad_cost - good_cost < 50 or len(good_search_path) < 100:
#         continue
#
#     env_copy_2 = deepcopy(env)
#     skill_path = rollout(env_copy_2, good_sol)
#     for x,y,skill in skill_path:
#         env_copy_2.set_path(x,y,skill)
#     img = env_copy_2.render('rgb_array', visible_mask=False)
#     window.show_img(img)
#     window.save_img(figure_dir, f'good_sol_skills.png')
#
#     env_copy_3 = deepcopy(env)
#     skill_path = rollout(env_copy_3, bad_sol)
#     for x,y,skill in skill_path:
#         env_copy_3.set_path(x,y,skill)
#     img = env_copy_3.render('rgb_array', visible_mask=False)
#     window.show_img(img)
#     window.save_img(figure_dir, f'bad_sol_skills.png')
#
#     sep = 30
#     i, j = 0, 0
#     good_images, bad_images = [], []
#     while True:
#         img = env.render('rgb_array', visible_mask=False)
#         window.show_img(img)
#         window.save_img(figure_dir, f'good_frame_{int(i/sep + 1)}.png')
#         good_images.append(imageio.imread(os.path.join(figure_dir, f'good_frame_{int(i/sep + 1)}.png')))
#         good_chunk = good_search_path[i:i + sep]
#         env.add_heat(good_chunk)
#         if i > len(good_search_path):
#             break
#         i += sep
#
#     while True:
#         img = env_copy.render('rgb_array', visible_mask=False)
#         window.show_img(img)
#         window.save_img(figure_dir, f'bad_frame_{int(j/sep + 1)}.png')
#         bad_images.append(imageio.imread(os.path.join(figure_dir, f'bad_frame_{int(j/sep + 1)}.png')))
#         bad_chunk = bad_search_path[j:j + sep]
#         env_copy.add_heat(bad_chunk)
#         if j > len(bad_search_path):
#             break
#         j += sep
#
#     imageio.mimsave(os.path.join(figure_dir, 'good.gif'), good_images, duration=0.5)
#     imageio.mimsave(os.path.join(figure_dir, 'bad.gif'), bad_images, duration=0.5)
#
#     for file in os.listdir(figure_dir):
#         if 'frame' in file:
#             os.remove(os.path.join(figure_dir, file))
#
#     print(f'saved {count}th exp')
#     count += 1