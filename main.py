import tensorflow as tf
import a3c_network as a3c

# Network parameters
BATCH_SIZE = 32
IMAGE_WIDTH = 84
IMAGE_HEIGHT = 84
NUM_CHANNELS = 4  # dqn inputs 4 image at same time as state

NUM_ACTIONS = 4

graph = tf.Graph()
with graph.as_default():
  network0 = a3c.A3CNetwork(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS, NUM_ACTIONS, 0)
  network1 = a3c.A3CNetwork(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS, NUM_ACTIONS, 1)
  networks = [network0, network1]

if __name__ == '__main__':
  config = tf.ConfigProto()
  with tf.Session(graph=graph, config=config) as session:
    tf.initialize_all_variables().run()
    for network in networks:
      weights_and_biases = network.weights_and_biases()
      for variable in weights_and_biases:
        print(variable.name)
