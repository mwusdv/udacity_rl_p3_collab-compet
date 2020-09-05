# Reinforcement Learning: Collaboration and Competition

## Project Details
For this project, we will train an agent to work with the Tennis environment.

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The task is episodic. After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores. This yields a single score for each episode.

**The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.**



## Getting Started
In the project, we need the Unity environment. We can download it that has already been built.

* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

Then, place the file in the udacity_rl_p3_collab-compet/ folder and unzip (or decompress) the file.

## Instructions
Firt download the necessary zip file according to the above section. Then please modify the 80-th line in the `train.py` and the 58-th line in `test.py` accordingly:

        env = UnityEnvironment(file_name="Reacher_Linux/Reacher.x86_64")
with the correct file name. Currently it is set as `Reacher_Linux/Tennis.x86_64` since I worked on this project in a ubuntu system. 

To launch the training code:

            python train.py
    
At the end of training, 4 data files `actor_agent_1.pth`, `actor_agnet_2.pth`, `critic_agent_1.pth' and 'critic_agent_2.pth` will be saved in the same folder. They are for actor and critic models respectively. 

To launch the test code:

            python test.py

