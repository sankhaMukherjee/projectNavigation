import matplotlib.pyplot as plt
import torch, json
import numpy as np

from dqn_agent   import Agent
from unityagents import UnityEnvironment
from collections import deque

def trainAgent(env, agent, trainingParams):

    n_episodes  =  trainingParams['n_episodes']
    max_t       =  trainingParams['max_t']
    eps_start   =  trainingParams['eps_start']
    eps_end     =  trainingParams['eps_end']
    eps_decay   =  trainingParams['eps_decay']

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            
            agent.step(state, action, reward, next_state, done)
            
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(
            i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=15.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                i_episode-100, np.mean(scores_window)))
            break
        
        torch.save(agent.qnetwork_local.state_dict(), 'models/checkpoint.pth')

    return scores, scores_window

def main():


    print('\n+'+'-'*100)
    print('| Generating the environment and the agent ...')
    print('+'+'-'*100)
    config = json.load(open('config.json'))
    env    = UnityEnvironment( 
                file_name=config['bananaFile'], 
                no_graphics=config['no_graphics'] )

    agent = Agent(state_size=37, action_size=4, seed=0)

    print('\n+'+'-'*30)
    print('| Training the agent ...')
    print('+'+'-'*30)
    trainAgent(env, agent, config['trainingParams'])


    env.close()
    
    return

if __name__ == '__main__':
    main()