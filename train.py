import torch
from unityagents import UnityEnvironment

import numpy as np
from collections import deque

from maddpg import MADDPG
import matplotlib.pyplot as plt

def train_maddpg(env, max_episode=1000, max_t=1000, print_every=5, check_history=100,
                 sigma_start=0.2, sigma_end=0.01, sigma_decay=0.995):
    # reset
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]  
    env_info = env.reset(train_mode=True)[brain_name]
    
    # action and state size
    action_size = brain.vector_action_space_size 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('State size:', state_size)
    print('Action size: ', action_size)
      
    # initialize agent
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)
    maddpg = MADDPG(state_size, action_size, random_seed=123)
    
    scores_deque = deque(maxlen=check_history)
    scores = []
   
    # learning multiple episodes
    sigma = sigma_start
    for episode in range(max_episode):
        # prepare for training in the current epoc
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        score = 0
        maddpg.reset(sigma=sigma)
        
        # play and learn in current episode
        for t in range(max_t):
            actions = maddpg.act(states)

            env_info = env.step(actions)[brain_name]   
            next_states = env_info.vector_observations       # get next state (for each agent)
            rewards = env_info.rewards                       # get reward (for each agent)
            dones = env_info.local_done                      # see if episode finished

            maddpg.step(t, states, actions, rewards, next_states, dones)
            states = next_states
            
            reward = np.max(rewards) # get max score of two agents as current score
            score += reward
            if np.any(dones):
                break 
        
        # update sigma for exlporation
        sigma = max(sigma_end, sigma*sigma_decay)
        
        # record score
        epoc_score = score
        scores_deque.append(epoc_score)
        scores.append(epoc_score)
        
        if episode % print_every == 0:
            print('Episode {}\tscore: {:.4f}\tAverage Score: {:.4f}'.format(episode, epoc_score, np.mean(scores_deque)))
        
        if np.mean(scores_deque) >= 0.5 and episode >= check_history:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.4f}'.format(episode - check_history, np.mean(scores_deque)))
            for agent in maddpg.ddpg_agents:
                torch.save(agent.actor_local.state_dict(), 'actor_agent_'+str(agent.id)+'.pth')
                torch.save(agent.actor_local.state_dict(), 'critic_agent_'+str(agent.id)+'.pth')
            break
        
     
    return scores

if __name__ == '__main__':
    env = UnityEnvironment(file_name='Tennis_Linux/Tennis.x86_64')
    scores = train_maddpg(env, max_episode=1000)
    env.close()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Scores')
    plt.xlabel('Episode')
    plt.show()