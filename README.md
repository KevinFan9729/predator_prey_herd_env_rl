# predator_prey_herd_env_rl
An environment of predator and prey herd and reinforcement learning 

## What this is 
2D environments for artificial predator and prey, with simple collision physics.

The custom_env contains a standalone environment and a custom model training script.

The openAI_gym_env contains the same environment wrapped in openAI gym and the corresponding stable baseline model training script.

Multiagent reinforcement learning exists, but the state space is large, and training is time and resources intensive.

Here we want to see whether we can train only one prey leader to guide a whole herd of prey agents to survive longer.

Only one prey leader is the RL agent, and the rest of the prey agents form a herd by the BOID algorithm. 

If the predator kills the prey leader, the predator will stop momentarily (about 1.5 seconds) to simulate the action of "eating the prey", and this pause also gives the prey herd to manoeuvre away from the predator.

## How to use
Run agent.py under custom_env to train a custom model in the standalone environment.

Run train.py under openAI_gym_env to train a stable baseline model in the openAI gym environment.

## Example
The followings are some examples after a limited iteration of training:

### single agent
https://user-images.githubusercontent.com/60720175/214194278-74d26400-f03a-483f-8add-4072b3b5e345.mp4

### prey herd

https://user-images.githubusercontent.com/60720175/214194332-0413c938-0f97-42b4-9063-306d8b41abd7.mp4


