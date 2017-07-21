# DQN-tensorflow
Implementation of [A3C](https://arxiv.org/pdf/1602.01783.pdf) using TensorFlow v0.9(But it is easy to modify and run it on higher versions)

## Prerequisitss
From [Here](https://github.com/yuishihara/Arcade-Learning-Environment/tree/multi_thread), clone multi thread supported arcade learning environment.
make and install it. Modifications to ale is necessary to avoid multi thread problems

## Usage

```sh
$ python main.py
```

There are several options to change learning parameters and behaviors.

- rom: Atari rom file to play. Defaults to breakout.bin.
- threads_num: Number of learner threads to run in parallel. Defaults to 8.
- local_t_max: Number of steps to look ahead. Defaults to 5.
- global_t_max: Number of iterations to train. Defaults to 1e-8. Learning rate will decrease propotional to this value.
- use_gpu: Whether to use gpu or cpu. Defaults to True. To use cpu set it to False.
- shrink_image: Whether to just shrink or trim and shrink state image. Defaults to False.
- life_lost_as_end: Treat life lost in the game as end of state. Defaults to True.
- evaluate: Evaluate trained network. Defaults to False.

Options can be used like follows

```sh
$ python main.py --rom="pong.bin" --threads_num=4
```

## Results

TBD

### To load trained network

```sh
$ python main.py --evaluate=True --checkpoint_dir=trained_results/breakout/ --trained_file=network_parameters-80002500
```

## License
MIT
