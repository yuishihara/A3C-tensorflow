from a3c_network import A3CNetwork

import tensorflow as tf

class SharedNetwork(A3CNetwork):
  def __init__(self, image_height, image_width, num_channels, num_actions, thread_id, device='/cpu:0'):
    super(SharedNetwork, self).__init__(image_height, image_width, num_channels, num_actions, thread_id, device)
    scope_name = "a3c_network_%d" % thread_id
    with tf.variable_scope(scope_name):
      self.eta = 7e-4
      self.alpha = 0.99
      self.epsilon = 0.1
      self.optimizer = self.prepare_optimizer()
      self.shared_counter = self.prepare_shared_counter()


  def prepare_optimizer(self):
    with tf.device(self.device):
      return tf.train.RMSPropOptimizer(self.eta, self.alpha, self.epsilon, name="shared_optimizer")


  def prepare_shared_counter(self):
    return tf.get_variable('counter', [], initializer=tf.constant_initializer(0), trainable=False)
