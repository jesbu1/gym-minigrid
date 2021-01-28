import os
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit


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





