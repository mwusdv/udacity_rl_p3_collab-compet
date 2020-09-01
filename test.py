import torch
from unityagents import UnityEnvironment
from maddpg import MADDPG

def test_ddpg(env):
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
      
    # initialize agent
    agent = Agent(state_size, action_size, random_seed=0)
    agent.actor_local.load_state_dict(torch.load('checkpoint_actor.data'))
    agent.critic_local.load_state_dict(torch.load('checkpoint_critic.data'))
    
    done = False
    state = env_info.vector_observations[0]  
    score = 0
    while not done:
        action = agent.act(state, add_noise=False)
         
        env_info = env.step(action)[brain_name]   
        next_state = env_info.vector_observations[0]      # get next state (for each agent)
        reward = env_info.rewards[0]                       # get reward (for each agent)
        done = env_info.local_done[0]                      # see if episode finished

        state = next_state
        score += reward
    
    print('Score: {:.2f}'.format(score))
             
if __name__ == '__main__':
    env = UnityEnvironment(file_name='Reacher_Linux/Reacher.x86_64')
    test_ddpg(env)
    env.close()