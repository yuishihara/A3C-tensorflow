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


class AleInterface(Environment):
  def __init__(self, rom_name, record_display=True):
    super(AleInterface, self).__init__()
    self.ale = ALEInterface()
    self.record_display = record_display

    if self.record_display:
      self.ale.setBool('display_screen', True)
      self.ale.setString('record_screen_dir', 'movie')
    else:
      self.display_name = rom_name
      cv2.startWindowThread()
      cv2.namedWindow(self.display_name)

    self.ale.loadROM(rom_name)
    self.actions = self.ale.getMinimalActionSet()
    self.screen_width, self.screen_height = self.ale.getScreenDims()


  def act(self, action):
    reward = self.ale.act(self.actions[action])
    screen = self.ale.getScreenGrayscale()
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
    if not self.record_display:
      cv2.imshow(self.display_name, screen)
    resized = cv2.resize(screen, (84, 84))
    return resized
