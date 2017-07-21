import numpy as np
import matplotlib.pyplot as plt

if __name__=='__main__':
  rewards_max = np.loadtxt('max_rewards.csv', delimiter=',')
  rewards_med = np.loadtxt('med_rewards.csv', delimiter=',')
  rewards_avg = np.loadtxt('avg_rewards.csv', delimiter=',')

  x = rewards_max[:, 1]
  max_y = rewards_max[:, 2]
  med_y = rewards_med[:, 2]
  avg_y = rewards_avg[:, 2]

  plt.figure(figsize=(5,4), dpi=80)

#  plt.plot(x, max_y, label='max', linewidth=1)
  plt.plot(x, med_y, label='median', linewidth=1)
  plt.plot(x, avg_y, label='average', linewidth=1)

  plt.legend(loc='upper left', fontsize=8)
  plt.xlim(0, 80)
  plt.ylim(0, 600)

  plt.title('A3C FF breakout')
  plt.xlabel('million steps')
  plt.ylabel('score')

  plt.savefig('breakout_result.png')
  plt.show()
