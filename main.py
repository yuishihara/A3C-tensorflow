from constants import IMAGE_WIDTH
from constants import IMAGE_HEIGHT
from constants import NUM_CHANNELS
from constants import NUM_ACTIONS

import numpy as np
import tensorflow as tf
import a3c_network as a3c
import shared_network as shared
import actor_learner_thread as actor_thread
import ale_environment as ale

graph = tf.Graph()
with graph.as_default():
  shared_network = shared.SharedNetwork(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS, NUM_ACTIONS, -1)
  network0 = a3c.A3CNetwork(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS, NUM_ACTIONS, 0)
  network1 = a3c.A3CNetwork(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS, NUM_ACTIONS, 1)
  networks = [network0, network1]

if __name__ == '__main__':
  config = tf.ConfigProto()
  with tf.Session(graph=graph, config=config) as session:
    tf.initialize_all_variables().run()
