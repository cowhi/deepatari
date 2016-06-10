.. -*- mode: rst -*-

=========
deepatari
=========

|License|_ |Docs|_

.. |License| image:: https://img.shields.io/badge/license-MIT-blue.svg
.. _License: https://github.com/cowhi/deepatari/blob/master/LICENSE.txt

.. |Docs| image:: https://readthedocs.org/projects/deepatari/badge/?version=latest
.. _Docs: http://deepatari.readthedocs.io/en/latest/?badge=latest

**Framework for developing and comparing reinforcement learning algorithms in
the Atari game playing domain**

This is a framework to support the development of algorithms for intelligent
agents in the Atari game playing domain. The idea is to let researcher or
programmer focus on the development of algorithms without having to think about
the whole environment implementation and to facilitate and generalize the
evaluation of these algorithms to make results comparable.

A big advantage is that you can simply write your own implementation using
whatever library you prefer and it should seamlessly work with the framework.
The following image shows how the framework is organized. So far you should
be able to add another learner class by copying the skeleton for a new learner
class from './deepatari/learner/skeleton.py' and run an experiment using the
'--with name_of_your_learner' argument.

.. image:: https://github.com/cowhi/deepatari/blob/master/experiment_setup.png
  :alt: Experiment setup
  :width: 740
  :height: 435
  :align: center

Based on:

* Python

* ALE (http://www.arcadelearningenvironment.org/)

* OpenAI gym (https://openai.com/blog/openai-gym-beta/)

Tested on CentOS release 6.6 with Python 2.7.11

.. contents:: **README contents**
  :depth: 2

Installation
============

This guide will help install all the necessary software in a virtual
environment generated with Anaconda. It is a step by step guide which
might not be necessary for everyone. If you want to integrate the
dependencies into your existing installation please do so.

Please be aware that I do not cover the installation of the Nvidia CUDA and
CuDNN driver, please follow these instructions:

http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/

Create a working directory
--------------------------

Create a folder, where you download all necessary stuff and keep the
rest of your system clean:

.. code:: shell

  mkdir deepatari_stuff && cd deepatari_stuff

Prepare conda environment
-------------------------

If you haven't done it already, install anaconda with instructions from here:

http://conda.pydata.org/docs/installation.html

Create a conda environment with the basic packages and activate it:

.. code:: shell

  conda create --name deepatariEnv python=2.7 pip numpy opencv matplotlib
  source activate deepatariEnv

Add non standard packages
-------------------------

The rest of the packages should be installed in this order, because they
downgrade certain packages which will be upgraded later on again.

Download and install Neon [3]:

.. code:: shell

  git clone https://github.com/NervanaSystems/neon.git
  cd neon && make sysinstall && cd ..

Download and install OpenAI gym [4]:

.. code:: shell

  git clone https://github.com/openai/gym.git
  cd gym && pip install -e '.[atari]' && cd ..

Install the latest Theano [6] version :

.. code:: shell

  pip install --upgrade https://github.com/Theano/Theano/archive/master.zip


Install the latest Lasagne [2] for Theano :

.. code:: shell

  pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip

Install Tensorflow (Ubuntu/Linux 64-bit, GPU enabled. Requires CUDA toolkit 7.5
and CuDNN v4. Other options on library page.) [5]:

.. code:: shell

  pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.8.0-cp27-none-linux_x86_64.whl


Install Keras [1]:

.. code:: shell

  pip install keras

Install deepatari
-----------------

The setup routine installs some runnable scripts to use directly from the
command line:

.. code:: shell

  git clone git@github.com:cowhi/deepatari.git
  cd deepatari && python setup.py install && cd ..

Now you should be able to test the installation.

Running the software
====================

After the installation you are ready to run your experiments.

One way to test the software with the minimal settings is to run the following
command in any directory. The program will run and create an individual folder
for each experiment as a subfolder of the 'results' directory, which will be
made if it does not exist. This example uses the Neon implementation of the
original DQN because it supports running on CPU in contrast to the Lasagne
implementation. If you have a compatible Nvidia GPU, you can ran any
implementation directly on the GPU.

.. code:: shell

  learn_to_play --with DQNNeon --fill_mem_size 128 --train_steps 128 --test_steps 64 --epochs 2 --log_type stdout --log_stats False --backend cpu

Sources & Inspirations
======================

I want to thank the authors of the following packages, framework and algorithms,
which served as an inspiration and were the basis for some of the algorithms
implementations.

Original code
-------------

Provided by Google Deep Mind under

  https://sites.google.com/a/deepmind.com/dqn/

based on their paper:

  Mnih, Volodymyr, Koray Kavukcuoglu, David Silver, Andrei A. Rusu, Joel Veness,
  Marc G. Bellemare, Alex Graves et al. "Human-level control through deep
  reinforcement learning." Nature 518, no. 7540 (2015): 529-533.
  http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html

Other implementations
---------------------

The following packages where heavily used for the respective learners:

* **deep_q_rl:** DQN implementation based on Theano and Lasagne

  https://github.com/spragunr/deep_q_rl

* **simple_dqn:** DQN implementation based on Neon

  https://github.com/tambetm/simple_dqn

The following packages are also interesting:

* **AgentNet:** A lightweight library to build and train deep reinforcement
  learning

  https://github.com/yandexdataschool/AgentNet

* **sherjilozair/dqn:** Basic DQN implementation, which uses OpenAI's gym
  environment and Keras/Theano

  https://github.com/sherjilozair/dqn

* **DEER:** DEEp Reinforcement learning framework (very similar to this package)

  https://github.com/VinF/deer

Community
---------

* **Deep Q-Learning:** Discussion group for DQN (mostly **deep_q_rl**)

  https://groups.google.com/forum/#!forum/deep-q-learning

Others
------

* **RLPy:** Framework for conducting sequential decision making experiments

   https://github.com/rlpy/rlpy

* **PyBrain:** Python Machine Learning Library with Reinforcement Learning

  https://github.com/pybrain/pybrain/tree/master/pybrain/rl

References
==========
[1] Keras: http://keras.io/

[2] Lasagne: http://lasagne.readthedocs.io/en/latest/

[3] Nervana Neon: http://www.nervanasys.com/technology/neon/

[4] OpenAI: https://openai.com/blog/openai-gym-beta/

[5] Tensorflow: https://www.tensorflow.org/

[6] Theano: http://deeplearning.net/software/theano/


Todo's
======

* Video playback (see: https://github.com/tambetm/simple_dqn)
* Record videos (see: https://github.com/openai/gym/blob/master/README.rst#id11)
* Prioritized Replay Memory (see: https://github.com/VinF/deer/blob/master/deer/helper/tree.py)
* Multi Agent support (see: https://github.com/yandexdataschool/AgentNet)
* Double DQN (see https://github.com/VinF/deer/blob/master/deer/q_networks/q_net_keras.py)

Known problems
==============

* Program does not run because of a problem with h5py

Error message:

.. code:: shell

  ...
  File "h5py/h5p.pyx", line 72, in h5py.h5p.propwrap (/tmp/pip-build-5MMDj7/h5py/h5py/h5p.c:2407)
  ValueError: Not a property list class (Not a property list class)

Possible fix (install more recent version of h5py from non-standard repo):

.. code:: shell

  conda install -c conda-forge h5py=2.6.0
