import numpy as np
import tensorflow as tf
import threading

class ActorLearnerThread(threading.Thread):
  def __init__(self, session, environment, shared_network, local_network, local_t_max, global_t_max, thread_id, device='/cpu:0'):
    super(ActorLearnerThread, self).__init__()
    self.session = session
    self.local_t_max = local_t_max
    self.global_t_max = global_t_max
    self.shared_network = shared_network
    self.local_network = local_network
    self.t = 1
    self.t_start = self.t
    self.image_width = local_network.input_shape()[0]
    self.image_height = local_network.input_shape()[1]
    self.num_channels = local_network.input_shape()[2]
    self.loop_listener = None
    self.skip_num = 4
    self.eps = 1e-10
    self.beta = 0.01
    self.gamma = 0.99
    self.grad_clip = 40
    self.device = device
    self.thread_id = thread_id

    self.environment = environment

    self.state_input, self.action_input, self.reward_input, self.value_input = self.prepare_placeholders(thread_id)
    self.pi = self.local_network.pi(self.state_input)
    self.value = self.local_network.value(self.state_input)
    self.policy_loss, self.value_loss = self.prepare_loss_operations(thread_id)
    self.local_grads = self.prepare_local_gradients(thread_id)
    self.reset_local_grads_ops = self.prepare_reset_local_gradients_ops(self.local_grads, thread_id)

    total_loss = self.policy_loss + self.value_loss
    self.accum_local_grads_ops = self.prepare_accum_local_gradients_ops(self.local_grads, total_loss, thread_id)
    self.apply_grads = self.prepare_apply_gradients(self.local_grads)
    self.sync_operations = self.prepare_sync_ops(self.shared_network, self.local_network)


  def prepare_placeholders(self, thread_id):
    scope_name = "thread_%d_placeholder" % thread_id
    with tf.variable_scope(scope_name):
      state_shape=[None] + list(self.local_network.input_shape())
      assert state_shape == [None, self.image_width, self.image_height, self.num_channels]
      state_input = tf.placeholder(tf.float32, shape=state_shape, name="state_input")

      action_shape=[None] + list(self.local_network.actor_output_shape())
      assert action_shape == [None, 4]
      action_input = tf.placeholder(tf.float32, shape=action_shape, name="action_input")

      reward_shape=[None] + list(self.local_network.critic_output_shape())
      assert reward_shape == [None, 1]
      reward_input = tf.placeholder(tf.float32, shape=reward_shape, name="reward_input")

      value_shape=[None, 1]
      value_input = tf.placeholder(tf.float32, shape=value_shape, name="value_input")

      return state_input, action_input, reward_input, value_input


  def prepare_loss_operations(self, thread_id):
    with tf.device(self.device):
      scope_name = "thread_%d_operations" % thread_id
      with tf.name_scope(scope_name):
        log_pi = tf.log(tf.clip_by_value(self.pi, self.eps, 1.0))
        entropy = - tf.reduce_sum(tf.mul(self.pi, log_pi))

        log_pi_a_s = tf.reduce_sum(tf.mul(log_pi, self.action_input), reduction_indices=1, keep_dims=True)

        # log_pi_a_s * advantage. This multiplication is bigger then better
        # append minus to use gradient descent as gradient ascent
        advantage = self.reward_input - self.value_input
        policy_loss = - tf.reduce_sum(log_pi_a_s * advantage) - entropy * self.beta
        value_loss = tf.nn.l2_loss(self.reward_input - self.value)

        return policy_loss, value_loss


  def prepare_local_gradients(self, thread_id):
    scope_name = "thread_%d_grads" % thread_id
    local_grads = []
    with tf.name_scope(scope_name):
      for variable in self.local_network.weights_and_biases():
        name = variable.name.replace(":", "_") + "_local_grad"
        shape = variable.get_shape().as_list()
        local_grad = tf.Variable(tf.zeros(shape, dtype=variable.dtype), name=name, trainable=False)
        local_grads.append(local_grad.ref())
    assert len(local_grads) == 10
    return local_grads


  def prepare_accum_local_gradients_ops(self, local_grads, dy, thread_id):
    scope_name = "thread_%d_accum_ops" % thread_id
    accum_grad_ops = []
    with tf.device(self.device):
      with tf.name_scope(scope_name):
        dxs = [v.ref() for v in self.local_network.weights_and_biases()]
        grads = tf.gradients(dy, dxs,
            gate_gradients=False,
            aggregation_method=None,
            colocate_gradients_with_ops=False)
        for (grad, var, local_grad) in zip(grads, self.local_network.weights_and_biases(), local_grads):
          name = var.name.replace(":", "_") + "_accum_grad_ops"
          accum_ops = tf.assign_add(local_grad, grad, name=name)
          accum_grad_ops.append(accum_ops)
    assert len(accum_grad_ops) == 10
    return tf.group(*accum_grad_ops, name="accum_ops_group_%d" % thread_id)


  def prepare_reset_local_gradients_ops(self, local_grads, thread_id):
    scope_name = "thread_%d_reset_ops" % thread_id
    reset_grad_ops = []
    with tf.device(self.device):
      scope_name = "thread_%d_reset_operations" % thread_id
      with tf.name_scope(scope_name):
        for (var, local_grad) in zip(self.local_network.weights_and_biases(), local_grads):
          zero = tf.zeros(var.get_shape().as_list(), dtype=var.dtype)
          name = var.name.replace(":", "_") + "_reset_grad_ops"
          reset_ops = tf.assign(local_grad, zero, name=name)
          reset_grad_ops.append(reset_ops)
    assert len(reset_grad_ops) == 10
    return tf.group(*reset_grad_ops, name="reset_grad_ops_group_%d" % thread_id)


  def prepare_apply_gradients(self, local_grads):
    with tf.device(self.device):
      clipped_grads = [tf.clip_by_value(grad, -self.grad_clip, self.grad_clip) for grad in local_grads]
      apply_grads = self.shared_network.optimizer.apply_gradients(
          zip(clipped_grads, self.shared_network.weights_and_biases()),
          global_step=self.shared_network.shared_counter)
      return apply_grads


  def prepare_sync_ops(self, origin, target):
    with tf.device(self.device):
      copy_operations = [target.assign(origin)
          for origin, target in zip(origin.weights_and_biases(), target.weights_and_biases())]
      return tf.group(*copy_operations)


  def run(self):
    available_actions = self.environment.available_actions()
    self.environment.reset()
    update_times = 1
    initial_state = self.get_initial_state(self.environment)
    last_state = initial_state
    while self.get_global_step() < self.global_t_max:
      if self.thread_id == 0:
        print 'thread_id: %d, learning_rate: %f' % (self.thread_id, self.session.run(self.shared_network.learning_rate))
        print 'global_step %d' % self.session.run(self.shared_network.shared_counter.ref())

      self.reset_gradients()
      self.synchronize_network()
      self.t_start = self.t

      if self.loop_listener is not None:
        self.loop_listener(self, update_times)

      if self.environment.is_end_state():
        # print 'thread_id: %d is now resetting' % self.thread_id
        self.environment.reset()
        initial_state = self.get_initial_state(self.environment)
      else:
        initial_state = last_state
      assert np.shape(initial_state) == (84, 84, 4)

      history, last_state = self.play_game(initial_state)

      if last_state is None:
        r = 0
      else:
        r = self.session.run(self.value, feed_dict={self.state_input : [last_state]})[0][0]

      states_batch = []
      action_batch = []
      reward_batch = []
      value_batch = []
      for i in range((self.t - 1) - self.t_start, -1, -1):
        snapshot = history[i]
        state, action, reward, value = self.extract_history(snapshot)

        r = reward + self.gamma * r
        states_batch.append(state)
        action_batch.append(action)
        reward_batch.append([r])
        value_batch.append([value])

      if len(history) is not 0:
        self.accumulate_gradients(states_batch, action_batch, reward_batch, value_batch)
        self.update_shared_gradients()
        update_times += 1


  def test_run(self, environment, trials):
    rewards = []
    for i in range(trials):
      initial_state = self.get_initial_state(environment)
      assert np.shape(initial_state) == (84, 84, 4)
      reward = self.test_play_game(environment, initial_state)
      print 'test play trial: %d finished. total reward: %d' % (i, reward)
      rewards.append(reward)
      environment.reset()
    return rewards


  def extract_history(self, history):
    state = history['state']
    action = np.zeros(self.local_network.actor_outputs)
    action[history['action']] = 1
    reward = history['reward']
    value = history['value']
    return state, action, reward, value


  def play_game(self, initial_state):
    history = []
    state = initial_state
    next_state = state
    next_screen = None
    available_actions = self.environment.available_actions()
    probabilities = None
    value = None
    action = None
    while self.environment.is_end_state() == False and (self.t - self.t_start) != self.local_t_max:
      state = next_state
      probabilities, value = self.session.run([self.pi, self.value], feed_dict={self.state_input : [state]})
      action = self.select_action_with(available_actions, probabilities[0])

      reward = 0.0
      for i in range(self.skip_num):
        intermediate_reward, next_screen = self.environment.act(action)
        reward += intermediate_reward
        if self.environment.is_end_state():
          break
      reward = np.clip([reward], -1.0, 1.0)[0]

      data = {'state':state, 'action':action, 'reward':reward, 'value':value[0][0]}
      history.append(data)
      next_screen = np.reshape(next_screen, (self.image_width, self.image_height, 1))
      next_state = np.append(state[:, :, 1:], next_screen, axis=-1)

      self.t += 1

    if self.environment.is_end_state():
      last_state = None
    else:
      last_state = next_state

    if self.thread_id == 0:
      moves = ['no-op', 'start', 'right', 'left']
      moves = [(move, prob) for move, prob in zip(moves, probabilities[0])]
      print "probabilities: %s, action: %s, value: %s" % (str(moves), str(moves[action]), str(value))
      # print 'self.t_start: %d, self.t: %d' % (self.t_start, self.t)
    return history, last_state


  def test_play_game(self, environment, initial_state):
    total_reward = 0
    state = initial_state
    next_state = state
    next_screen = None
    available_actions = environment.available_actions()
    random_action_probability = 0.01
    action_num = 0
    random_action_num = 0
    while environment.is_end_state() == False:
      state = next_state
      action = None
      if random_action_probability < np.random.rand():
        probabilities = self.session.run(self.pi, feed_dict={self.state_input : [state]})
        action = self.select_action_with(available_actions, probabilities[0])
      else:
        action = self.select_random_action_from(available_actions)
        random_action_num += 1
      action_num += 1

      for i in range(self.skip_num):
        reward, next_screen = environment.act(action)
        total_reward += reward

      next_screen = np.reshape(next_screen, (self.image_width, self.image_height, 1))
      next_state = np.append(state[:, :, 1:], next_screen, axis=-1)
    print 'random action probability %f' % (float(random_action_num) / float(action_num))

    return total_reward


  def get_initial_state(self, environment):
    available_actions = environment.available_actions()

    random_action_num = np.random.randint(0, 30)
    black_screen = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
    last_screen = black_screen
    for i in range(random_action_num):
      environment.act(0)
    reward, last_screen = environment.act(1)

    initial_state = [black_screen, black_screen, black_screen, last_screen]

    assert np.shape(initial_state) == (self.num_channels, self.image_height, self.image_width)
    return np.stack(initial_state, axis=-1)


  def synchronize_network(self):
    self.session.run(self.sync_operations)


  def accumulate_gradients(self, state, action, r, value):
    self.session.run(self.accum_local_grads_ops,
        feed_dict={self.state_input: state,
          self.action_input: action,
          self.reward_input: r,
          self.value_input: value})


  def reset_gradients(self):
    self.session.run(self.reset_local_grads_ops)


  def update_shared_gradients(self):
    self.session.run(self.apply_grads)


  def save_parameters(self, file_name, global_step):
    if self.saver is None:
      return
    self.saver.save(self.session, save_path=file_name, global_step=global_step)


  def set_saver(self, saver):
    self.saver = saver


  def get_global_step(self):
    return tf.train.global_step(self.session, self.shared_network.shared_counter) * self.local_t_max


  def set_loop_listener(self, listener):
    self.loop_listener = listener;


  def select_action_with(self, available_actions, probabilities):
    return np.random.choice(available_actions, p=probabilities)


  def select_random_action_from(self, available_actions):
    return np.random.choice(available_actions)
