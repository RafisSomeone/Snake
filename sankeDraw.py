import csv
import matplotlib.pyplot as plt
import numpy as np
def moving_average(x, w=10):
    return np.convolve(x, np.ones(w), 'valid') / w

sigmas = [0.99] 
alphas = [0.01]
epsilons = [0.01]
results =[]
s = 1
with open('Reward1.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        results.append(float(row[0]))
moving_average_result = moving_average(results)

fig = plt.figure(figsize=(16, 6))
plt.plot(moving_average_result)
plt.ylabel('Reward')
plt.xlabel('Attempt')
plt.title(f'Moving average sigma')


plt.savefig(f'TheNewest_buckets_moving_average_{s}.png')
plt.show(block = False)
plt.clf()

# avg = np.average(results, axis=0)
# std = np.std(results, axis=0)
# plt.plot(range(len(avg)), avg)
# plt.fill_between(range(len(avg)), avg-std, avg+std,alpha=0.2, edgecolor='#1B2ACC', facecolor='#089FFF',
# linewidth=4, linestyle='dashdot', antialiased=True)
# plt.ylabel('Reward')
# plt.xlabel('Attempt')
# plt.title(f'Average sigma = {sigmas[s]}, alpha = {alphas[s]}, epsilon = {epsilons[s]}')
# plt.savefig(f'sarsa/TheNewest_buckets_average_{s}.png')
# plt.show(block = False)
# plt.clf()