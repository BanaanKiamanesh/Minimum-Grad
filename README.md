# Minimum Grad

This is a simple implementation of a auto grad system implemented in MATLAB only for educational purposes. 

The project is inspired by:

  1. [micrograd](https://github.com/karpathy/micrograd/) by Andrej Karpathy.
  2. [teenygrad](https://github.com/tinygrad/teenygrad) by George Hotz.

The backend is the Tensor class which supports basic operations and automatic differentiation. Plus there are implementations of a activation functions and numerical optimizers for training simple neural networks. It is not intended for production use, although I tried to optimize it at some level. It is mainly for learning and understanding the concepts behind automatic differentiation and neural networks.

**Couple notes to consider:**

  1. The clarity and readability of the code is prioritized over performance.
  2. There are multiple examples if you are interested in seeing the way this operates and how to use get started with it.
  3. I wanted to implement more Layers like Conv2D, LSTM, etc. but I didn't have the time yet. Feel free to contribute if you want to see more layers.
  4. If you wanna run it, make sure to add the src folder and the subfolders to your MATLAB path.
