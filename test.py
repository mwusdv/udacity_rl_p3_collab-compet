import torch
from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt

from maddpg import MADDPG

def test_ddpg(env, episodes=10):
    # reset
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]  
    env_info = env.reset(train_mode=False)[brain_name]
    
    # action and state size
    action_size = brain.vector_action_space_size 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('State size:', state_size)
    print('Action size: ', action_size)
    
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)
    
    # load MADDPG agent
    maddpg = MADDPG(state_size, action_size, random_seed=0)
    for agent in maddpg.ddpg_agents:
       agent.actor_local.load_state_dict(torch.load('actor_agent_'+str(agent.id)+'.pth'))
       agent.critic_local.load_state_dict(torch.load('critic_agent_'+str(agent.id)+'.pth'))

    
    scores = []
    for n in range(episodes):
        # prepare for training in the current epoc
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        score = 0
        
        dones = [False] * num_agents
        states = env_info.vector_observations
        score = 0
        while not np.any(dones):
            actions = maddpg.act(states, add_noise=False)
             
            env_info = env.step(actions)[brain_name]   
            next_states = env_info.vector_observations      # get next state (for each agent)
            rewards = env_info.rewards                       # get reward (for each agent)
            dones = env_info.local_done                      # see if episode finished
            states = next_states
            
            score += np.max(rewards) 
            
        scores.append(score)
            
    print('Average score over {} episodes: {:.4f}'.format(episodes, np.mean(scores)))
    return scores
             
if __name__ == '__main__':
    env = UnityEnvironment(file_name='Tennis_Linux/Tennis.x86_64')
    scores = test_ddpg(env, episodes=50)
    env.close()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Scores')
    plt.xlabel('Episode')
    plt.show()