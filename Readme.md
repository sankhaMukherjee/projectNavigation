# projectNavigation

One Paragraph of project description goes here

Watch an agent that has been trained for approximately 20 iterations 650 iterations [here](https://youtu.be/khzMY8EACpQ).

## Prerequisites

You will need to have a valid Python installation on your system. This has been tested with Python 3.6. It does not assume a particulay version of python, however, it makes no assertions of proper working, either on this version of Python, or on another. 

Since this work was initially done on a Mac, the `./p1_navigation` folder contains a binary for mac. You will not need this for running the code. Just for running the solution. There are several environments available:

 - [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
 - [Mac](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
 - [Win32](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
 - [Win64](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

Install these in a convinient location within your folder structure, and use this file for the training etc. A configuration file `src/config.json` has a parameter `bananaFile` which you must change to point to the location of this file that you just downloaded.

## Installing

1. Clone this repository to your computer, and create a virtual environment. Note that there is a `Makefile` at the base folder. You will want to run the command `make env` to generate the environment. Do this at the first time you are cloning the repo. The python environment is already copied from the deep reinforced learning repository, and is installed within your environment, when you generate the environment.

2. The next thing that you want to do is to activate your virtual environment. For most cases, it will look like the following: `source env/bin/activate`. If you are using the [`fish`](https://fishshell.com) shell, you will use the command `source env/bin/activate.fish`.

3. Change to the folder `src` for all other operations. There is a `Malefile` available within this folder as well. You can use the Makefile as is, or simply run the python files yourself.

## Operation

After you have generated the environment, activated it, and switched to the `src` folder, you will want to execute the program. The folder structure is shown below:

```bash
.
├── config.json             # <- file containing configuration information
├── dqn_agent.py            # <- file containint the agent
├── model.py                # <- file containing a simple sequential neural network
├── models                  # <- a folder containing saved models
└── projectNavigation.py    # <- main program that runs everything
```

You will typically run the program by the following command `python projectNavigation.py`. Most of the essential parameters can be modified by changing the information within the `config.json` file. The configuration file has several sections: 

 - `bananaFile`: This is the location of the Unity Environment that will allow you to explore the environment 
 - `no_graphics`: Set this to `true` if you dont want to view the environment 
 - `trainingParams`: hyperparameters that will be used for training 
 - `training`: You will be able to configuration so that the program will train an agent 
 - `run`: A trained agent will have the opportunity of running within this environment. 

Several different types of operations that you can do is described below:

### 1. Training the model

For training the model, you would want to change the `["training"]["todo"]` parameter to `true`. It is possible that you may want to start from a pre-trained model. In that case, you should provide a path to an earlier model in the parameter `["training"]["startModel"]`.

After the training is complete, this will generate a checkpoint file `src/models/checkpoint-[<date time string>].pth`. Thus, every time you train the model, a new file will be generated, and wont destroy models that had been trained earlier. 

Before training, you may want to change the hyperparameters within the section `["trainingParams"]` in the configuration file. Available parameters, and their current values are shown:

```python
{
    "n_episodes"  :  2000,   # <- number of episodes to run
    "max_t"       :  1000,   # <- num of maximum iterations to go through in a single iteration
    "eps_start"   :  1.0,    # <- starting value of epsilon (for epsilon greedy parameter)
    "eps_end"     :  0.01,   # <- the minimum epsilon value to use
    "eps_decay"   :  0.995   # <- The rate at which the epsilon will be multiplied to decrease the value of the current epsilon
}
```

An example of training the model is shown below:

```bash

+------------------------------
| Training the agent ...
+------------------------------
Episode 100 Average Score: 1.09
Episode 200 Average Score: 3.94
Episode 300 Average Score: 7.88
Episode 400 Average Score: 10.43
Episode 500 Average Score: 12.77
Episode 600 Average Score: 13.57
Episode 700 Average Score: 14.59
Episode 726 Average Score: 15.00
Environment solved in 626 episodes! Average Score: 15.00
```

It takes approximately 650 steps to reach a moving-average score of 15.

![training image](https://photos.google.com/album/AF1QipPyOVcmRPqi7P5GuooFigqpv5yHnJSWhwWXel27/photo/AF1QipNZf-T4gNFIznQIDX54wdHSVojkZxs8GFZcEi3N)


## Authors

Sankha S. Mukherjee - Initial work (2018)

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details

## Acknowledgments

 - This repo contains a copy of the python environment available [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/python). 
 - Two files [dqn_agent.py](https://raw.githubusercontent.com/udacity/deep-reinforcement-learning/master/dqn/exercise/dqn_agent.py) and [model.py](https://raw.githubusercontent.com/udacity/deep-reinforcement-learning/master/dqn/exercise/model.py) have been used with practically no modification.
 - The solution follows very closely to the solution present for the dqn challenge present [here](https://github.com/udacity/deep-reinforcement-learning/blob/master/dqn/solution/Deep_Q_Network_Solution.ipynb). 

 