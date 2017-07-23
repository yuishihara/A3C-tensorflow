#  The MIT License (MIT)
#
#  Copyright (c) 2017 Yu Ishihara
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

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
