# projectNavigation

One Paragraph of project description goes here

Watch an agent that has been trained for approximately 20 iterations 650 iterations [here](https://youtu.be/khzMY8EACpQ).

## 1. Prerequisites

You will need to have a valid Python installation on your system. This has been tested with Python 3.6. It does not assume a particulay version of python, however, it makes no assertions of proper working, either on this version of Python, or on another. 

Since this work was initially done on a Mac, the `./p1_navigation` folder contains a binary for mac. You will not need this for running the code. Just for running the solution. There are several environments available:

 - [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
 - [Mac](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
 - [Win32](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
 - [Win64](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

Install these in a convinient location within your folder structure, and use this file for the training etc. A configuration file `src/config.json` has a parameter `bananaFile` which you must change to point to the location of this file that you just downloaded.

## 2. Installing

1. Clone this repository to your computer, and create a virtual environment. Note that there is a `Makefile` at the base folder. You will want to run the command `make env` to generate the environment. Do this at the first time you are cloning the repo. The python environment is already copied from the deep reinforced learning repository, and is installed within your environment, when you generate the environment.

2. The next thing that you want to do is to activate your virtual environment. For most cases, it will look like the following: `source env/bin/activate`. If you are using the [`fish`](https://fishshell.com) shell, you will use the command `source env/bin/activate.fish`.

3. Change to the folder `src` for all other operations. There is a `Malefile` available within this folder as well. You can use the Makefile as is, or simply run the python files yourself.

## 3. Operation

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

### 3.1. Training the model

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

![Imgur](https://i.imgur.com/gAq79Mc.png)

### 3.2. Run the model 

To run the model, you will need to change the  `["training"]["todo"]` parameter to `true` within the `src/config.json` file. For running the model, it is imperative that a model file be provided. This repo comes with several pre-trained model files in the `src/models` folder. If you want to see the program being evaluated, you should use the model `checkpoint-2018-10-13--15-18-43.pth` directly by setting the parameter `["run"]["startModel"]` in the configuration file. 

## 4. Model Description

A short description of the model will be provided here. The learning is done by an `Agent` (defined in `src/dqn_agent.py`). This agent uses a simple 3-layer neural network which acts as the `QNetwork` (defined in `src\model`). The agent employs a replay buffer `ReplayBuffer` (defined in `src/dqn_agent.py`). The function of each of these will be described in the following subsections, and finally, the learning algorithm will be described. 

### 4.1. The `ReplayBuffer`

This maintains a `deque` that continuously adds new experiences (i.e. a tuple containing the current state, the current action, the next state, the next action, the next reward, and whether we are done with the current episode). It has two methods. The first allows one to add experiences from this deque, and another that allows it to sample from it. 

### 4.2. The `QNetwork`

This is a simple 3-layer fully connected network. The input is the current state, and the output is a vector of the same size as the action space, and represents a Q value for each action. The first two layers have relu activation, while the last one is unactivated, which allows the last layer to have any real value. 

### 4.3. The `Agent`

The agent is composed of two Qnetworks (for a double DQN architecture), a replay buffer and an optimizer. We shall call the two QNetworks the local (dynamic Q Network) and the target (ideal Q network).

An `act` method allows one to generate an action using an epsilon-greedy policy from the local network. 

A `learn` method calculates the MSE error between the expected Q value `(reward + gamma * Q(next state, next action))` vs. the current calculated Q value, and updates the optimizer.

A `soft_update` step updates the target network with the current dynamic weights.

A `step` method. At each step, this function adds relevant data into the replay buffer. After every `UPDATE_EVERY` step, this method will also call its own `learn` method to update the local Q-Network.


### 4.4. The learning algorithm

This is present within the `src/projectNavigation.py` file within the function `trainAgent` The algorithm may be briefly described as follows:

1. Do the following for many episodes 
2. restart the environment 
3. Get the current state
3. At each step:
    3.1. get an action using an $\epsilon$-greedy policy
    3.2. Use that action to go to the next state (and get the reward at the same time) 
    3.3. Update the agent with this information (the agent adds this to its replay memory and may wish to update the local network/target network, depending upon the batch size etc.)
    3.4. If it is done, break out of the current task
4. If the current average score is greater than 15 stop training 
5. Save the model.

## 5. Future Work

The following might be added to improve the learning algorithm:

1. priotirized experience replay
2. Double DQN

## 6. Authors

Sankha S. Mukherjee - Initial work (2018)

## 7. License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details

## 8. Acknowledgments

 - This repo contains a copy of the python environment available [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/python). 
 - Two files [dqn_agent.py](https://raw.githubusercontent.com/udacity/deep-reinforcement-learning/master/dqn/exercise/dqn_agent.py) and [model.py](https://raw.githubusercontent.com/udacity/deep-reinforcement-learning/master/dqn/exercise/model.py) have been used with practically no modification.
 - The solution follows very closely to the solution present for the dqn challenge present [here](https://github.com/udacity/deep-reinforcement-learning/blob/master/dqn/solution/Deep_Q_Network_Solution.ipynb). 

 