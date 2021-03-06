# MIT Deep Learning

<a href="https://deeplearning.mit.edu/"><img src="https://deeplearning.mit.edu/files/images/mit_deep_learning.png"></a>

This repository is a collection of tutorials for [MIT Deep Learning](https://deeplearning.mit.edu/) courses. 

Expanded by loopuleasa for learning purposes.

## Setup

Install [Anaconda](https://www.anaconda.com/download/), create a fresh conda environment and install the following packages:

````
conda create mit-env
conda activate mit-env
````

For the basic tutorial:

````
conda install tensorflow
conda install matplotlib
conda install seaborn
conda install opencv-python
conda install ipython
````

For driving segmentation tutorial:

````
conda install pillow
conda install tqdm
conda install scikit-learn
conda install tabulate
````

Install Jupyter notebooks

````
conda config --add channels conda-forge
conda install jupyter
conda install sympy
conda install jupytext
````

Setup [Jupytext](https://github.com/mwouts/jupytext)
````
jupyter notebook --generate-config
````
then open up the generated ``jupyter_notebook_config.py`` and append the following
````
c.NotebookApp.contents_manager_class = "jupytext.TextFileContentsManager"
c.ContentsManager.default_jupytext_formats = "ipynb,py"
c.ContentsManager.preferred_jupytext_formats_save = "py:percent"
````

Now all jupyter notebooks are jupytext compatible and can be changed directly from the ``.py`` files, while the ``.ipynb`` files are synchronised on the fly when page is refreshed.

To run jupyter notebooks simply use the following command to launch the server:

````
jupyter notebook
````


If you want to create new jupytext notebooks, use the commands:

````
#convert a jupyter notebook to python
jupytext --to py:percent <notebook.ipynb>

#convert any jupytext notebook (.py, .md, etc.) back to a .ipynb  
jupytext --to notebook <jupytext_notebook>  
````


## Tutorial: Deep Learning Basics

<a href="https://github.com/lexfridman/mit-deep-learning/blob/master/tutorial_deep_learning_basics/deep_learning_basics.ipynb"><img src="https://i.imgur.com/j4FqBuR.gif"></a>

This tutorial accompanies the [lecture on Deep Learning Basics](https://www.youtube.com/watch?list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf&v=O5xeyoRL95U). It presents several concepts in deep learning, demonstrating the first two (feed forward and convolutional neural networks) and providing pointers to tutorials on the others. This is a good place to start.

Links: \[ [Jupyter Notebook](https://github.com/lexfridman/mit-deep-learning/blob/master/tutorial_deep_learning_basics/deep_learning_basics.ipynb) \]
\[ [Google Colab](https://colab.research.google.com/github/lexfridman/mit-deep-learning/blob/master/tutorial_deep_learning_basics/deep_learning_basics.ipynb) \]
\[ [Lecture Video](https://www.youtube.com/watch?list=PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf&v=O5xeyoRL95U) \]





## Tutorial: Driving Scene Segmentation

<a href="https://github.com/lexfridman/mit-deep-learning/blob/master/tutorial_driving_scene_segmentation/tutorial_driving_scene_segmentation.ipynb"><img src="images/thumb_driving_scene_segmentation.gif"></a>

This tutorial demostrates semantic segmentation with a state-of-the-art model (DeepLab) on a sample video from the MIT Driving Scene Segmentation Dataset.

Links: \[ [Jupyter Notebook](https://github.com/lexfridman/mit-deep-learning/blob/master/tutorial_driving_scene_segmentation/tutorial_driving_scene_segmentation.ipynb) \]
\[ [Google Colab](https://colab.research.google.com/github/lexfridman/mit-deep-learning/blob/master/tutorial_driving_scene_segmentation/tutorial_driving_scene_segmentation.ipynb) \]


## DeepTraffic Deep Reinforcement Learning Competition

<a href="https://selfdrivingcars.mit.edu/deeptraffic"><img src="images/thumb_deeptraffic.gif"></a>

DeepTraffic is a deep reinforcement learning competition. The goal is to create a neural network that drives a vehicle (or multiple vehicles) as fast as possible through dense highway traffic.

Links: \[ [GitHub](https://github.com/lexfridman/deeptraffic) \] \[ [Website](https://selfdrivingcars.mit.edu/deeptraffic) \] \[ [Paper](https://arxiv.org/abs/1801.02805) \]

## Team

- [Lex Fridman](https://lexfridman.com)
- [Li Ding](https://www.mit.edu/~liding/)
- [Jack Terwilliger](https://www.mit.edu/~jterwill/)
- [Michael Glazer](https://www.mit.edu/~glazermi/)
- [Aleksandr Patsekin](https://www.mit.edu/~patsekin/)
- [Aishni Parab](https://www.mit.edu/~aishni/)
- [Dina AlAdawy](https://www.mit.edu/~aladawy/)
- [Henri Schmidt](https://www.mit.edu/~henris/)
