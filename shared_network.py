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

from a3c_network import A3CNetwork

import tensorflow as tf

class SharedNetwork(A3CNetwork):
  def __init__(self, image_height, image_width, num_channels, num_actions, thread_id, local_t_max, global_t_max, device='/cpu:0'):
    super(SharedNetwork, self).__init__(image_height, image_width, num_channels, num_actions, thread_id, device)
    scope_name = "a3c_network_%d" % thread_id
    with tf.variable_scope(scope_name):
      self.eta = 7e-4
      self.alpha = 0.99
      self.momentum = 0.0
      self.epsilon = 0.1
      self.local_t_max = local_t_max
      self.global_t_max = global_t_max
      self.shared_counter = self.prepare_shared_counter()
      self.learning_rate = self.learning_rate()
      self.optimizer = self.prepare_optimizer(self.learning_rate)


  def prepare_optimizer(self, learning_rate):
    with tf.device(self.device):
      return tf.train.RMSPropOptimizer(
          learning_rate=learning_rate,
          decay=self.alpha,
          momentum=self.momentum,
          epsilon=self.epsilon,
          name="shared_optimizer")


  def learning_rate(self):
    return self.eta * (1.0 - self.shared_counter.ref() * self.local_t_max / self.global_t_max)


  def prepare_shared_counter(self):
    return tf.get_variable('counter', [], initializer=tf.constant_initializer(0), trainable=False)
