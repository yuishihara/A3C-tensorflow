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

import tensorflow as tf
import numpy as np

class A3CNetwork:
  def __init__(self, image_height, image_width, num_channels, num_actions, thread_id, device='/cpu:0'):
    scope_name = "a3c_network_%d" % thread_id
    with tf.variable_scope(scope_name):
      # network variables
      self.conv1_filter_size = 8
      self.conv1_filter_num = 16
      self.conv1_stride = 4
      self.conv1_output_size = (image_width - self.conv1_filter_size) / self.conv1_stride + 1  # 20 with 84 * 84 image and no padding
      assert self.conv1_output_size == 20
      self.conv1_weights, self.conv1_biases = \
          self.create_conv_net([self.conv1_filter_size, self.conv1_filter_size, num_channels, self.conv1_filter_num], "common_conv")

      self.conv2_filter_size = 4
      self.conv2_filter_num = 32
      self.conv2_stride = 2
      self.conv2_output_size = (self.conv1_output_size - self.conv2_filter_size) / self.conv2_stride + 1  # 9 starting from 84 * 84 image and no padding
      assert self.conv2_output_size == 9
      self.conv2_weights, self.conv2_biases = \
          self.create_conv_net([self.conv2_filter_size, self.conv2_filter_size, self.conv1_filter_num, self.conv2_filter_num], "common_conv2")

      self.inner1_inputs = self.conv2_output_size * self.conv2_output_size * self.conv2_filter_num # should be 2592 for default
      assert self.inner1_inputs == 2592
      self.inner1_outputs = 256
      self.inner1_weights, self.inner1_biases = self.create_inner_net([self.inner1_inputs, self.inner1_outputs], "common_inner")

      self.actor_inputs = self.inner1_outputs
      self.actor_outputs = num_actions
      self.actor_weights, self.actor_biases = self.create_inner_net([self.actor_inputs, self.actor_outputs], "actor_inner")

      self.critic_inputs = self.inner1_outputs
      self.critic_output = 1
      self.critic_weights, self.critic_biases = self.create_inner_net([self.critic_inputs, self.critic_output], "critic_inner")

    self.device = device


  def common_layer(self, data):
    with tf.device(self.device):
      conv1 = tf.nn.conv2d(data, self.conv1_weights, [1, self.conv1_stride, self.conv1_stride, 1], padding='VALID')
      conv1 = tf.nn.relu(conv1 + self.conv1_biases)
      conv2 = tf.nn.conv2d(data, self.conv2_weights, [1, self.conv2_stride, self.conv2_stride, 1], padding='VALID')
      conv2 = tf.nn.relu(conv1 + self.conv2_biases)
      conv2_shape = conv2.get_shape().as_list()
      inner1 = tf.reshape(conv3, [conv2_shape[0], conv2_shape[1] * conv2_shape[2] * conv2_shape[3]])
      inner1 = tf.matmul(inner1, self.inner1_weights) + self.inner1_biases
      return tf.nn.relu(inner1)


  def pi(self, data):
    with tf.device(self.device):
      common_layer_outputs = common_layer(data, device)
      return tf.nn.softmax(tf.matmul(common_layer_outputs, self.actor_weights) + self.actor_biases)


  def value(self, data):
    with tf.device(self.device):
      common_layer_outputs = common_layer(data, device)
      value = tf.matmul(common_layer_outputs, self.critic_weights) + self.critic_biases
      return tf.reshape(value)


  def pi_and_value(self, data):
      return pi(data, device), value(data, device)


  def weights_and_biases(self):
    return [self.conv1_weights, self.conv1_biases,
            self.conv2_weights, self.conv2_biases,
            self.inner1_weights, self.inner1_biases,
            self.actor_weights, self.actor_biases,
            self.critic_weights, self.critic_biases]


  def create_conv_net(self, shape, name):
    w = shape[0]
    h = shape[1]
    input_channels  = shape[2]
    output_channels = shape[3]
    d = 1.0 / np.sqrt(input_channels * w * h)
    initializer = tf.random_uniform_initializer(minval=-d, maxval=d)

    weights = tf.get_variable(name + '_weights', shape, initializer=initializer)
    biases = tf.get_variable(name + '_biases', shape[3], initializer=initializer)
    return weights, biases


  def create_inner_net(self, shape, name):
    input_channels  = shape[0]
    output_channels = shape[1]
    d = 1.0 / np.sqrt(input_channels)

    initializer = tf.random_uniform_initializer(minval=-d, maxval=d)
    weights = tf.get_variable(name + '_weights', shape, initializer=initializer)
    biases = tf.get_variable(name + '_biases', output_channels, initializer=initializer)
    return weights, biases
