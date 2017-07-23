from datetime import datetime as dt
import gflags
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

FLAGS = gflags.FLAGS
gflags.DEFINE_string('data_dir', '', 'Graph title')
gflags.DEFINE_string('title', '', 'Graph title')
gflags.DEFINE_string('xlabel', 'million steps', 'Label for x-axis')
gflags.DEFINE_string('ylabel', 'score', 'Label for y-axis')
gflags.DEFINE_string('legend_pos', 'upper left', 'Legend position')
gflags.DEFINE_string('name', '', 'file name of png to save')
gflags.DEFINE_integer('xlim', 80, 'x-axis limit')
gflags.DEFINE_integer('ylim', 1000, 'y-axis limit')

def load_data(directory, file_name):
  full_path = os.path.join(directory, file_name)
  return np.loadtxt(full_path, delimiter=',')

if __name__=='__main__':
  try:
    argv = FLAGS(sys.argv)
  except gflags.FlagsError:
    print 'Incompatible flags were specified'

  rewards_max = load_data(FLAGS.data_dir, 'max_rewards.csv')
  rewards_med = load_data(FLAGS.data_dir, 'med_rewards.csv')
  rewards_avg = load_data(FLAGS.data_dir, 'avg_rewards.csv')

  x = rewards_max[:, 1]
  max_y = rewards_max[:, 2]
  med_y = rewards_med[:, 2]
  avg_y = rewards_avg[:, 2]

  plt.figure(figsize=(5,4), dpi=80)

#  plt.plot(x, max_y, label='max', linewidth=1)
  plt.plot(x, med_y, label='median', linewidth=1)
  plt.plot(x, avg_y, label='average', linewidth=1)

  plt.legend(loc=FLAGS.legend_pos, fontsize=8)
  plt.xlim(0, FLAGS.xlim)
  plt.ylim(0, FLAGS.ylim)

  current_datetime = dt.now()
  title = FLAGS.title if FLAGS.title is not '' else current_datetime.strftime('%Y/%m/%d')
  plt.title(FLAGS.title)
  plt.xlabel(FLAGS.xlabel)
  plt.ylabel(FLAGS.ylabel)

  file_name = FLAGS.name if FLAGS.name is not '' else current_datetime.strftime('%Y%m%d_%H%M%S')
  plt.savefig(file_name + '.png')
  plt.show()
