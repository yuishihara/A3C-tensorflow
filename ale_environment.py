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
  def __init__(self, rom_name, record_display=True, show_display=False, id = 0):
    super(AleEnvironment, self).__init__()
    self.ale = ALEInterface()
    self.ale.setInt('frame_skip', 3)
    self.ale.setInt('random_seed', int(np.random.rand() * 100))
    self.ale.setFloat('repeat_action_probability', 0.0)
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
    self.screen = np.empty((self.screen_height, self.screen_width, 1), dtype=np.uint8)


  def __enter__(self):
    return self


  def __exit__(self, exc_type, exc_value, traceback):
    cv2.destroyWindow(self.display_name)


  def act(self, action):
    reward = self.ale.act(self.actions[action])
    screen = self.ale.getScreenGrayscale(self.screen)
    screen = np.reshape(screen, (self.screen_height, self.screen_width, 1))
    state = self.preprocess(screen)
    return reward, state


  def is_end_state(self):
    return self.ale.game_over()


  def reset(self):
    self.ale.reset_game()


  def available_actions(self):
    # return available indexes instead of actual action value
    return range(0, len(self.actions))


  def preprocess(self, screen):
    if self.show_display:
      cv2.imshow(self.display_name, screen)
    resized = cv2.resize(screen, (84, 84))
    return resized
