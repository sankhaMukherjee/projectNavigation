# projectNavigation

One Paragraph of project description goes here

## Prerequisites

You will need to have a valid Python installation on your system. This has been tested with Python 3.6. It does not assume a particulay version of python, however, it makes no assertions of proper working, either on this version of Python, or on another. 

Since this work was initially done on a Mac, the `./p1_navigation` folder contains a binary for mac. You will not need this for running the code. Just for running the solution. There are several environments available:

 - [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
 - [Mac](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
 - [Win32](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
 - [Win64](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

Install these in a convinient location within your folder structure, and use this file for the training etc.

## Installing

1. Clone this repository to your computer, and create a virtual environment. Note that there is a `Makefile` at the base folder. You will want to run the command `make env` to generate the environment. Do this at the first time you are cloning the repo. The python environment is already copied from the deep reinforced learning repository, and is installed within your environment, when you generate the environment.

2. The next thing that you want to do is to activate your virtual environment. For most cases, it will look like the following: `source env/bin/activate`. If you are using the [`fish`](https://fishshell.com) shell, you will use the command `source env/bin/activate.fish`.

3. Change to the folder `src` for all other operations. There is a `Malefile` available within this folder as well. You can use the Makefile as is, or simply run the python files yourself.

## Operation

After you have generated the environment, activated it, and switched to the `src` folder, you will want to execute the program. The folder structure is shown below:

```
.
├── config.json             # <- file containing configuration information
├── dqn_agent.py            # <- file containint the agent
├── model.py                # <- file containing a simple sequential neural network
├── models                  # <- a folder containing saved models
└── projectNavigation.py    # <- main program that runs everything
```

You will typically run the program by the following command `python projectNavigation.py`.

Most of the essential parameters can be modified by changing the information within the `config.json` file. You can either train the model, or run the model. After the training is complete, this will generate a checkpoint file `src/models/checkpoint.pth`.

## Authors

Sankha S. Mukherjee - Initial work (2018)

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details

## Acknowledgments

 - This repo contains a copy of the python environment available [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/python). 
 - Two files [dqn_agent.py](https://raw.githubusercontent.com/udacity/deep-reinforcement-learning/master/dqn/exercise/dqn_agent.py) and [model.py](https://raw.githubusercontent.com/udacity/deep-reinforcement-learning/master/dqn/exercise/model.py) have been used with practically no modification.
 - The solution follows very closely to the solution present for the dqn challenge present [here](https://github.com/udacity/deep-reinforcement-learning/blob/master/dqn/solution/Deep_Q_Network_Solution.ipynb). 

 