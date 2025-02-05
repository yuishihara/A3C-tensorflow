from constants import IMAGE_WIDTH
from constants import IMAGE_HEIGHT
from constants import NUM_CHANNELS

import gflags
import sys
import os
import time
import re
import numpy as np
import tensorflow as tf
import a3c_network as a3c
import shared_network as shared
import actor_learner_thread as actor_thread
import ale_environment as ale

FLAGS = gflags.FLAGS
gflags.DEFINE_string('summary_dir', 'results/summary', 'Target summary directory')
gflags.DEFINE_string('checkpoint_dir', 'results/checkpoint', 'Target checkpoint directory')
gflags.DEFINE_string('rom', 'breakout.bin', 'Rom name to play')
gflags.DEFINE_string('trained_file', '', 'File to restore the network')
gflags.DEFINE_integer('threads_num', 8, 'Threads to create')
gflags.DEFINE_integer('local_t_max', 5, 'batch size to use for training')
gflags.DEFINE_integer('global_t_max', 8e7, 'Max steps')
gflags.DEFINE_boolean('use_gpu', True, 'True to use gpu, False to use cpu')
gflags.DEFINE_boolean('shrink_image', False, 'Just shrink image for preprocessing')
gflags.DEFINE_boolean('life_lost_as_end', True, 'Treat live lost as end of episode')
gflags.DEFINE_boolean('evaluate', False, 'Evaluate trained network')
gflags.DEFINE_boolean('take_video', False, 'Take video for during evaluation')


def merged_summaries(maximum, median, average):
  max_summary = tf.scalar_summary('rewards max', maximum)
  med_summary = tf.scalar_summary('rewards med', median)
  avg_summary = tf.scalar_summary('rewards avg', average)
  return tf.merge_summary([max_summary, med_summary, avg_summary])


previous_time = time.time()
previous_step = 0
previous_evaluation_step = 0
evaluation_environment = None
summary_writer = None
summary_op = None
def loop_listener(thread, iteration):
  global previous_time
  global previous_step
  global previous_evaluation_step
  global maximum_input
  global median_input
  global average_input
  global summary_writer
  global summary_op
  global evaluation_environment
  checkpoint_dir = FLAGS.checkpoint_dir
  STEPS_PER_EPOCH = 100
  current_time = time.time()
  current_step = thread.get_global_step()
  elapsed_time = current_time - previous_time
  steps = (current_step - previous_step)
  print("itearation: %d, previous step: %d" % (iteration, previous_step))
  print("### Performance: {} steps in {:.5f} seconds. {:.0f} STEPS/s. {:.2f}M STEPS/hour".format(
    steps, elapsed_time, steps / elapsed_time, steps / elapsed_time * 3000 / 1000000.))
  previous_time = current_time
  previous_step = current_step
  if STEPS_PER_EPOCH < (current_step - previous_evaluation_step):
    step = thread.get_global_step()
    print 'Save network parameters! step: %d' % step
    thread.save_parameters(checkpoint_dir + '/network_parameters', step)

    previous_evaluation_step = current_step
    trials = 10
    rewards = []
    for i in range(trials):
      reward = thread.test_run(evaluation_environment)
      rewards.append(reward)
    maximum = np.max(rewards)
    median = np.median(rewards)
    average = np.average(rewards)
    epoch = current_step / STEPS_PER_EPOCH
    summary_writer.add_summary(thread.session.run(summary_op,
      feed_dict={maximum_input: maximum, median_input: median, average_input: average}),
      epoch)
    print 'test run for epoch: %d. max: %d, med: %d, avg: %f' % (epoch, maximum, median, average)


def create_dir_if_not_exist(directory):
  if not os.path.exists(directory):
    os.makedirs(directory)


def remove_old_files(directory):
  for file in os.listdir(directory):
    os.remove(os.path.join(directory, file))


def write_training_settings(directory):
  file_path = os.path.join(directory, "settings.log")
  with open(file_path, "w") as f:
    flags = ['rom: ' + str(FLAGS.rom),
        'threads_num: ' + str(FLAGS.threads_num),
        'local_t_max: ' + str(FLAGS.local_t_max),
        'global_t_max: ' + str(FLAGS.global_t_max),
        'shrink_image: ' + str(FLAGS.shrink_image),
        'use_gpu: ' + str(FLAGS.use_gpu),
        'life_lost_as_end: ' + str(FLAGS.life_lost_as_end)]
    for flag in flags:
      line = f.write(flag + '\n')


def start_training():
  global previous_evaluation_step
  global previous_step
  global summary_writer
  global summary_op
  global maximum_input
  global median_input
  global average_input
  global evaluation_environment
  checkpoint_dir = FLAGS.checkpoint_dir
  summary_dir = FLAGS.summary_dir
  graph = tf.Graph()
  config = tf.ConfigProto()

  # Output to tensorboard
  create_dir_if_not_exist(summary_dir)
  # Model parameter saving
  create_dir_if_not_exist(checkpoint_dir)
  if FLAGS.trained_file is '':
    remove_old_files(summary_dir)
    remove_old_files(checkpoint_dir)

  summary_writer = tf.train.SummaryWriter(summary_dir, graph=graph)
  write_training_settings('results')

  networks = []
  shared_network = None
  summary_op = None

  evaluation_environment = ale.AleEnvironment(FLAGS.rom, record_display=False,
    show_display=True, id=100, shrink=FLAGS.shrink_image, life_lost_as_end=False)

  with graph.as_default():
    num_actions = len(evaluation_environment.available_actions())
    maximum_input = tf.placeholder(tf.int32)
    median_input = tf.placeholder(tf.int32)
    average_input = tf.placeholder(tf.int32)
    summary_op = merged_summaries(maximum_input, median_input, average_input)
    device = '/gpu:0' if FLAGS.use_gpu else '/cpu:0'
    shared_network = shared.SharedNetwork(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS, num_actions, 100,
        FLAGS.local_t_max, FLAGS.global_t_max, device)
    for i in range(FLAGS.threads_num):
      network = a3c.A3CNetwork(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS, num_actions, i, device)
      networks.append(network)
    saver = tf.train.Saver(max_to_keep=None)


  with tf.Session(graph=graph, config=config) as session:
    threads = []
    for thread_num in range(FLAGS.threads_num):
      show_display = True if (thread_num == 0) else False
      environment = ale.AleEnvironment(FLAGS.rom, record_display=False, show_display=show_display, id=thread_num, shrink=FLAGS.shrink_image)
      thread = actor_thread.ActorLearnerThread(session, environment, shared_network,
          networks[thread_num], FLAGS.local_t_max, FLAGS.global_t_max, thread_num)
      thread.set_saver(saver)
      thread.daemon = True
      if thread_num == 0:
        thread.set_loop_listener(loop_listener)
      threads.append(thread)

    session.run(tf.initialize_all_variables())
    if FLAGS.trained_file is not '':
      saver.restore(session, checkpoint_dir + '/' + FLAGS.trained_file)
      previous_step = threads[0].get_global_step()
      previous_evaluation_step = previous_step
    else:
      print 'No trained file specified. Use default parameter'

    for i in range(FLAGS.threads_num):
      threads[i].start()

    while True:
      try:
        ts = [thread.join(10) for thread in threads if thread is not None and thread.isAlive()]
        if len(ts) == 0:
          break
      except KeyboardInterrupt:
        print 'Ctrl-c received! Sending kill to threads...'
        for thread in threads:
          thread.kill_received = True
        break

    print 'Training finished!!'


def start_evaluation():
  checkpoint_dir = FLAGS.checkpoint_dir
  graph = tf.Graph()
  config = tf.ConfigProto()

  shared_network = None
  evaluation_network = None
  environment = ale.AleEnvironment(FLAGS.rom, record_display=FLAGS.take_video, show_display=show_display, id=0, shrink=FLAGS.shrink_image, life_lost_as_end=False)

  with graph.as_default():
    num_actions = len(environment.available_actions())
    device = '/gpu:0' if FLAGS.use_gpu else '/cpu:0'
    shared_network = shared.SharedNetwork(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS, num_actions, 100,
        FLAGS.local_t_max, FLAGS.global_t_max, device)
    evaluation_network = a3c.A3CNetwork(IMAGE_WIDTH, IMAGE_HEIGHT, NUM_CHANNELS, num_actions, 0, device)
    saver = tf.train.Saver(max_to_keep=None)

  with tf.Session(graph=graph, config=config) as session:
    show_display = True
    thread_num = 0
    thread = actor_thread.ActorLearnerThread(session, environment, shared_network, evaluation_network, FLAGS.local_t_max, FLAGS.global_t_max, thread_num)
    thread.set_saver(saver)
    thread.daemon = True

    session.run(tf.initialize_all_variables())

    if FLAGS.trained_file is not '':
      saver.restore(session, checkpoint_dir + '/' + FLAGS.trained_file)
    else:
      print 'No trained file specified. Abort evaluation'
      return

    trials = 10
    rewards = []
    for i in range(trials):
      reward = thread.test_run(environment)
      rewards.append(reward)

    maximum = np.max(rewards)
    median = np.median(rewards)
    average = np.average(rewards)

    print 'Evaluation finished. max: %d, med: %d, avg: %f' % (maximum, median, average)


if __name__ == '__main__':
  try:
    argv = FLAGS(sys.argv)
  except gflags.FlagsError:
    print 'Incompatible flags were specified'

  os.environ['OMP_NUM_THREADS'] = '1'

  if FLAGS.evaluate == True:
    start_evaluation()
  else:
    start_training()
