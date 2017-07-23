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

import unittest


class A3CTest(tf.test.TestCase):
  def setUp(self):
       pass


  def test_select_action(self):
    environment = ale.AleEnvironment('breakout.bin', record_display=False)
    graph = tf.Graph()
    with graph.as_default():
      shared_network = shared.SharedNetwork(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS, NUM_ACTIONS, 100)
      network0 = a3c.A3CNetwork(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS, NUM_ACTIONS, 0)
    with self.test_session(graph = graph) as session:
      test_thread = actor_thread.ActorLearnerThread(session, environment, shared_network, network0, 10, 0)
      probabilities = [0.1, 0.2, 0.3, 0.4]
      selected_num = np.zeros(4)
      available_actions = environment.available_actions()

      samples = 10000
      for i in range(samples):
        action = test_thread.select_action_with(available_actions, probabilities)
        selected_num[action] += 1

      allowed_diff = 500
      for i in range(len(selected_num)):
        mean = probabilities[i] * samples
        print 'mean:%d selected_num:%d for action:%d' % (mean, selected_num[i], i)
        self.assertTrue((mean - allowed_diff) < selected_num[i] and selected_num[i] < (mean + allowed_diff))


  def test_local_gradients(self):
     environment = ale.AleEnvironment('breakout.bin', record_display=False)
     graph = tf.Graph()
     with graph.as_default():
       shared_network = shared.SharedNetwork(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS, NUM_ACTIONS, 100)
       network0 = a3c.A3CNetwork(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS, NUM_ACTIONS, 0)
     with self.test_session(graph = graph) as session:
       test_thread = actor_thread.ActorLearnerThread(session, environment, shared_network, network0, 10, 0)
       test_thread.reset_gradients()
       self.assertEquals(len(test_thread.local_grads), 10)
       for local_grad in test_thread.local_grads:
         session.run(tf.Print(local_grad, [local_grad]))


  def test_pi(self):
     environment = ale.AleEnvironment('breakout.bin', record_display=False)
     graph = tf.Graph()
     with graph.as_default():
       shared_network = shared.SharedNetwork(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS, NUM_ACTIONS, 100)
       network0 = a3c.A3CNetwork(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS, NUM_ACTIONS, 0)
     with self.test_session(graph = graph) as session:
       tf.initialize_all_variables().run()
       test_thread = actor_thread.ActorLearnerThread(session, environment, shared_network, network0, 10, 0)
       test_thread.reset_gradients()
       initial_state = test_thread.get_initial_state(environment)
       state = np.stack(initial_state, axis=-1)
       probabilities = session.run(test_thread.pi, feed_dict={test_thread.state_input : [state]})
       self.assertTrue(len(probabilities[0]) == test_thread.local_network.actor_outputs)
       self.assertTrue(0.99 < np.sum(probabilities) and np.sum(probabilities) < 1.01)


  def test_shape(self):
     environment = ale.AleEnvironment('breakout.bin', record_display=False)
     graph = tf.Graph()
     with graph.as_default():
       shared_network = shared.SharedNetwork(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS, NUM_ACTIONS, 100)
       network0 = a3c.A3CNetwork(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS, NUM_ACTIONS, 0)
     with self.test_session(graph = graph) as session:
       tf.initialize_all_variables().run()
       test_thread = actor_thread.ActorLearnerThread(session, environment, shared_network, network0, 10, 0)
       initial_state = test_thread.get_initial_state(environment)
       state = np.stack(initial_state, axis=-1)

       pi = session.run(test_thread.pi, feed_dict={test_thread.state_input : [state, state], test_thread.action_input: [[0, 0, 1, 0], [0, 1, 0, 0]]})
       print 'pi' + str(np.shape(pi))
       self.assertTrue(np.shape(pi) == (2, 4))

       value = session.run(test_thread.value, feed_dict={test_thread.state_input : [state, state], test_thread.action_input: [[0, 0, 1, 0], [0, 1, 0, 0]], test_thread.reward_input: [[0], [0]]})
       print 'value shape: ' + str(np.shape(value)) + ' value: ' + str(value)
       self.assertTrue(np.shape(value) == (2, 1))


  def test_play_game(self):
     environment = ale.AleEnvironment('breakout.bin', record_display=False)
     graph = tf.Graph()
     with graph.as_default():
       shared_network = shared.SharedNetwork(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS, NUM_ACTIONS, 100)
       network0 = a3c.A3CNetwork(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS, NUM_ACTIONS, 0)
     with self.test_session(graph = graph) as session:
       tf.initialize_all_variables().run()
       test_thread = actor_thread.ActorLearnerThread(session, environment, shared_network, network0, 10, 0)
       test_thread.reset_gradients()
       initial_state = test_thread.get_initial_state(environment)
       self.assertEquals(len(initial_state), NUM_CHANNELS)

       history, last_state = test_thread.play_game(initial_state)
       if last_state is not None:
         self.assertEquals(len(history), test_thread.local_t_max)
       self.assertEquals(len(history[0]), 3)


  def test_accumulate_gradients(self):
     environment = ale.AleEnvironment('breakout.bin', record_display=False)
     graph = tf.Graph()
     local_t_max = 20
     global_t_max = 100
     with graph.as_default():
       shared_network = shared.SharedNetwork(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS, NUM_ACTIONS, 100, local_t_max, global_t_max)
       network0 = a3c.A3CNetwork(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS, NUM_ACTIONS, 0)
     with self.test_session(graph = graph) as session:
       tf.initialize_all_variables().run()
       test_thread = actor_thread.ActorLearnerThread(session, environment, shared_network, network0, local_t_max, global_t_max, 0)
       test_thread.reset_gradients()
       initial_state = test_thread.get_initial_state(environment)
       self.assertEquals(np.shape(initial_state), (IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS))

       history, last_state = test_thread.play_game(initial_state)

       session.run(test_thread.reset_local_grads_ops)
       local_grad = test_thread.local_grads[0]

       r = 0.0
       gamma = 0.99
       for i in range((test_thread.t - 1) - test_thread.t_start, -1, -1):
         state = history[i]['state']
         action = np.zeros(test_thread.local_network.actor_outputs)
         action[history[i]['action']] = 1
         reward = history[i]['reward']
         value = history[i]['value']
         r = reward + gamma * r
         test_thread.accumulate_gradients([state], [action], [[r]], [[value]])
         local_grad_step = local_grad.eval()
         tf.Print(local_grad_step, [local_grad_step]).eval()

       session.run(test_thread.reset_local_grads_ops)

       r = 0
       states_batch = []
       action_batch = []
       r_batch = []
       value_batch = []
       for i in range((test_thread.t - 1) - test_thread.t_start, -1, -1):
         state = history[i]['state']
         action = np.zeros(test_thread.local_network.actor_outputs)
         action[history[i]['action']] = 1
         reward = history[i]['reward']
         value = history[i]['value']

         r = reward + gamma * r
         states_batch.append(state)
         action_batch.append(action)
         r_batch.append([r])
         value_batch.append([value])

       test_thread.accumulate_gradients(states_batch, action_batch, r_batch, value_batch)
       local_grad_batch = local_grad.eval()
       tf.Print(local_grad_batch, [local_grad_batch]).eval()

       frobenius_norm = np.linalg.norm(local_grad_step - local_grad_batch)
       print 'Frobenius norm between batch and step: ' + str(frobenius_norm)
       self.assertTrue(frobenius_norm < 1e-2)
       # No exceptions


  def test_run(self):
     environment = ale.AleEnvironment('breakout.bin', record_display=False)
     graph = tf.Graph()
     with graph.as_default():
       shared_network = shared.SharedNetwork(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS, NUM_ACTIONS, 100)
       network0 = a3c.A3CNetwork(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS, NUM_ACTIONS, 0)
       network1 = a3c.A3CNetwork(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS, NUM_ACTIONS, 1)
     with self.test_session(graph = graph) as session:
       test_thread = actor_thread.ActorLearnerThread(session, environment, shared_network, network0, 10, 0)
       test_thread2 = actor_thread.ActorLearnerThread(session, environment, shared_network, network1, 20, 1)
       session.run(tf.initialize_all_variables())
       shared_weight = test_thread.shared_network.weights_and_biases()[0]
       local_weight = test_thread.local_network.weights_and_biases()[0]
       test_thread.synchronize_network()
       session.run(tf.Print(shared_weight, [shared_weight]))
       session.run(tf.Print(local_weight, [local_weight]))
       test_thread.run()
       session.run(tf.Print(shared_weight, [shared_weight]))
       session.run(tf.Print(local_weight, [local_weight]))
       assert shared_network.shared_counter.eval() == 10

       test_thread2.run()
       assert shared_network.shared_counter.eval() == 20


  def test_test_run(self):
     environment = ale.AleEnvironment('breakout.bin', record_display=False)
     graph = tf.Graph()
     with graph.as_default():
       shared_network = shared.SharedNetwork(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS, NUM_ACTIONS, 100)
       network0 = a3c.A3CNetwork(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS, NUM_ACTIONS, 0)
     with self.test_session(graph = graph) as session:
       test_thread = actor_thread.ActorLearnerThread(session, environment, shared_network, network0, 10, 0)
       session.run(tf.initialize_all_variables())
       trials = 3
       rewards = test_thread.test_run(environment, trials)
       assert len(rewards) == trials


  def loop_listener(self, thread, counter):
    assert thread is not None
    assert 0 <= counter and counter <= 10
    print 'counter: %d' % counter


  def test_loop_listener(self):
     environment = ale.AleEnvironment('breakout.bin', record_display=False)
     graph = tf.Graph()
     with graph.as_default():
       shared_network = shared.SharedNetwork(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS, NUM_ACTIONS, 100)
       network0 = a3c.A3CNetwork(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS, NUM_ACTIONS, 0)
     with self.test_session(graph = graph) as session:
       test_thread = actor_thread.ActorLearnerThread(session, environment, shared_network, network0, 10, 0)
       test_thread.set_loop_listener(self.loop_listener)
       session.run(tf.initialize_all_variables())
       test_thread.run()


if __name__ == '__main__':
  unittest.main()
