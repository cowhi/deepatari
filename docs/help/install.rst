.. -*- mode: rst -*-

Installation guide
==================

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
