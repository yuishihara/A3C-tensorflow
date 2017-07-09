#  The MIT License (MIT)
#
#  Copyright (c) 2016 Yu Ishihara
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

from ale_python_interface import ALEInterface
from environment import Environment
import numpy as np
import cv2


class AleEnvironment(Environment):
  def __init__(self, rom_name, record_display=True, show_display=False, id = 0, shrink=False, life_lost_as_end=True, use_grayscale=False):
    super(AleEnvironment, self).__init__()
    self.ale = ALEInterface()
    self.ale.setInt('random_seed', int(np.random.rand() * 100))
    self.ale.setFloat('repeat_action_probability', 0.0)
    self.ale.setBool('color_averaging', False)
    self.record_display = record_display
    self.show_display = show_display

    if self.record_display:
      self.ale.setBool('display_screen', True)
      self.ale.setString('record_screen_dir', 'movie')
    elif self.show_display:
      self.display_name = rom_name + '_' + str(id)
      cv2.startWindowThread()
      cv2.namedWindow(self.display_name)

    self.ale.loadROM(rom_name)
    self.actions = self.ale.getMinimalActionSet()
    self.screen_width, self.screen_height = self.ale.getScreenDims()
    self.use_grayscale = use_grayscale
    if self.use_grayscale:
      self.screen = np.zeros((self.screen_height, self.screen_width, 1), dtype=np.uint8)
    else:
      self.screen = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
      self.prev_screen = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
    self.shrink = shrink
    self.life_lost_as_end = life_lost_as_end
    self.lives_lost = False
    self.lives = self.ale.lives()


  def __enter__(self):
    return self


  def __exit__(self, exc_type, exc_value, traceback):
    cv2.destroyWindow(self.display_name)


  def act(self, action):
    reward = self.ale.act(self.actions[action])
    if self.use_grayscale:
      screen = self.ale.getScreenGrayscale(self.screen)
    else:
      current_screen = self.ale.getScreenRGB(self.screen)
      screen = np.maximum(current_screen, self.prev_screen)
      self.prev_screen = current_screen
      screen = screen[:, :, 0] * 0.2126 + screen[:, :, 1] * 0.0722 + screen[:, :, 2] * 0.7152
      screen = screen.astype(np.uint8)
    screen = np.reshape(screen, (self.screen_height, self.screen_width, 1))
    state = self.preprocess(screen)
    self.lives_lost = True if self.lives > self.ale.lives() else False
    self.lives = self.ale.lives()
    return reward, state


  def is_end_state(self):
    if self.life_lost_as_end:
      return self.ale.game_over() or self.lives_lost
    else:
      return self.ale.game_over()


  def reset(self):
    if self.ale.game_over():
      self.ale.reset_game()
    self.lives = self.ale.lives()
    self.lives_lost = False


  def available_actions(self):
    # return available indexes instead of actual action value
    return range(0, len(self.actions))


  def preprocess(self, screen):
    if self.show_display:
      cv2.imshow(self.display_name, screen)

    if self.shrink:
      resized = cv2.resize(screen, (84, 84))
    else:
      resized = cv2.resize(screen, (84, 110))
      resized = resized[18:102, :]

    scaled = resized.astype(np.float32) / 255.0
    return scaled
