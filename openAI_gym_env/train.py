from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common import results_plotter
import numpy as np
import matplotlib.pyplot as plt
from zmq import device
from gymEnv import predatorPreyCustomEnv
import os
import csv

fields = ['Num timestep', 'mean reward'] 
data=[]

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, checkFreq: int, logDir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.checkFreq = checkFreq
        self.logDir = logDir
        self.savePath = os.path.join(logDir, 'best_model')
        self.bestMeanReward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.savePath is not None:
            os.makedirs(self.savePath, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.checkFreq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.logDir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 50 episodes
              meanReward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.bestMeanReward, meanReward))
                data.append([self.num_timesteps,meanReward])
              
              if self.num_timesteps>15000:#save models and record rewards after learning has began  
                # New best model, you could save the agent here
                if meanReward > self.bestMeanReward:
                    self.bestMeanReward = meanReward
                    # Example for saving best model
                    if self.verbose > 0:
                      print("Saving new best model to {}".format(self.savePath))
                    self.model.save(self.savePath)

        return True

# Create log dir
logDir = "35statesRewardModReduced/"
os.makedirs(logDir, exist_ok=True)

# Create environment
env = predatorPreyCustomEnv(mode="MLP")
env = Monitor(env, logDir)
# Instantiate the agent
#MlpPolicy or CnnPolicy
model = DQN('MlpPolicy', env, buffer_size=5000000,batch_size=32, learning_rate=0.0001, learning_starts=50000, 
target_update_interval=1000, train_freq=4, exploration_final_eps=0.01, exploration_fraction=0.1,device='cuda')
# model=DQN('CnnPolicy', env, buffer_size=5000,learning_starts=100)
callback = SaveOnBestTrainingRewardCallback(checkFreq=1000, logDir=logDir)
# Train the agent

# Train the agent
# model.learn(total_timesteps=500000, callback=callback)
model.learn(total_timesteps=5000000, callback=callback)

# Save the agent
model.save("35statesRewardModReduced")
with open('rewardLog35statesRewardModReduced.csv', 'w') as f:
  write = csv.writer(f)
  write.writerow(fields)
  write.writerows(data)


results_plotter.plot_results([logDir], 2500000, results_plotter.X_TIMESTEPS, "Predator and Prey")
plt.show()


#LargeScreenAllPreyPos best mean reward: 119.49
#LargerScreenCenterPos best mean reward: 163.34
#LargerScreenCenterPosLargerBuffer best mean reward: 213.38
# LargerScreenAllPreyPosLargerBuffer best mean reward: 198.03

# 293.95 16 states
# 227.15 35 states