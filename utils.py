from collections import UserList
import numpy as np
from tf_agents.trajectories import Trajectory
# import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from pandas import read_csv
from tf_agents.trajectories import time_step as ts


def evaluate_policy(agent, test_buffer, num_test_steps=1000):
    difference = 0
    policy_state = agent.policy.get_initial_state(batch_size=1)
    for i in range(min(len(test_buffer), num_test_steps)):
        experience = test_buffer[i]
        time_step = ts.TimeStep(
                step_type=experience.step_type,
                reward=experience.reward,
                discount=experience.discount,
                observation= tf.reshape(experience.observation, (1, 2))
                )
        action_step = agent.policy.action(time_step, policy_state)
        policy_state = action_step.state

        difference += abs(float((experience.action - action_step.action).numpy()[0]))
    
    return difference


def discretize(action, num_actions=5):
    return tf.constant((max(min(action, 100), 0)+(50/(num_actions-1)) // (100/(num_actions-1))), dtype=tf.float32)
    
def undiscretize(action, num_actions=5):
    return tf.constant(100/(num_actions-1) * action)    

def plot_trajs(trajs):
    states = np.squeeze(np.array([traj.observation for traj in trajs]))
    actions = np.squeeze(np.array([traj.action for traj in trajs]))
    plt.figure()
    plt.plot(range(len(states)), states[:, 0:2])
    plt.title('losses')
    plt.savefig('./plot/states_trajs.png')
    plt.figure()
    plt.plot(range(len(actions)), actions)
    plt.title('trajectory actions')
    plt.savefig('./plot/actions_trajs.png')

def plot_loss(losses, num_episodes=0):
    plt.figure()
    # plt.plot(range(len(losses)), losses)
    if num_episodes > 0:
        n = len(losses)//num_episodes
        mean_loss_for_episode = [np.mean(losses[i:i+n]) for i in range(0, len(losses), n)]
        plt.plot(range(0, len(losses), n), mean_loss_for_episode)
    plt.title('losses')
    plt.savefig('./plot/losses.png')

def get_trajectory_from_csv(path, state_dim, train_buffer, test_buffer, TRAIN_TEST_RATIO):
    df = read_csv(path, index_col=0)

    trajs = []
    train_end = len(df) * TRAIN_TEST_RATIO
    for idx, record in df.iterrows():

        state = record.iloc[:state_dim].values  # Convert to numpy array
        action = tf.constant(record["Akcje"], dtype=tf.float32)
        reward = record["Nagrody"]
        continous_action = tf.expand_dims(tf.clip_by_value(action, 0, 100), axis=-1)

        traj = Trajectory(tf.constant(1, dtype=tf.int32, shape=(1,)), 
                        tf.expand_dims(tf.constant(state, dtype=tf.float32), axis=0),
                        continous_action, 
                        (), 
                        tf.constant(1, dtype=tf.int32, shape=(1,)),
                        tf.constant(reward, dtype=tf.float32, shape=(1,)), 
                        tf.constant(0, dtype=tf.float32, shape=(1,)))
        trajs.append(traj)

        if  idx > train_end:
            test_buffer.add_batch(traj)
        else:
            train_buffer.add_batch(traj)

    plot_trajs(traj)

class CustomReplayBuffer(UserList):
    def __init__(self, batch_size=32, num_steps=2):
        super().__init__([])
        self._batch_size=batch_size 
        self._num_steps=num_steps

    def get_iterator(self):
        return self.dataset_gen()  

    def dataset_gen(self):
        indexes = np.arange(len(self.data)-self._num_steps+1)
        idx = 0

        while True:
            # po przejściu wszystkich przestaw kolejność pobierania z listy
            np.random.shuffle(indexes)

            while idx < indexes.shape[0]: # po kolei po przetasowanych indeksach

                

                # jeśli koniec batcha nie przekracza zakresu indeksów
                if idx + self._batch_size < indexes.shape[0]:
                    batch = []
                    for i in range(idx, idx+self._batch_size):
                        shuffled_idx = indexes[i]
                        # slice wycina _num_steps kolejnych doświadczeń z listy
                        batch.append(self.data[shuffled_idx:shuffled_idx+self._num_steps]) 

                    idx += self._batch_size

                # jeśli koniec batcha  przekracza zakresu indeksów
                else:
                    batch = []
                    # przejście do końca listy
                    for i in range(idx, indexes.shape[0]):
                        shuffled_idx = indexes[i]
                        batch.append(self.data[shuffled_idx:shuffled_idx+self._num_steps] ) 

                    # dopełnienie początkiem listy
                    for i in range(idx+self._batch_size - indexes.shape[0]):
                        shuffled_idx = indexes[i]
                        batch.append(self.data[shuffled_idx:shuffled_idx+self._num_steps] ) 

                    idx = idx+self._batch_size - indexes.shape[0] 
                    break
                
                step_type       = np.ndarray((len(batch), len(batch[0])))
                observation     = np.ndarray((len(batch), len(batch[0]), batch[0][0][1].numpy().shape[1]))
                actions         = np.ndarray((len(batch), len(batch[0])))
                info            = np.ndarray((len(batch), len(batch[0])))
                next_step_type  = np.ndarray((len(batch), len(batch[0])))
                reward          = np.ndarray((len(batch), len(batch[0])))
                discount        = np.ndarray((len(batch), len(batch[0])))
                for b in range(len(batch)):
                    for s in range(len(batch[0])):
                        step_type     [b, s] = batch[b][s].step_type       
                        observation   [b, s] = batch[b][s].observation.numpy()[0]     
                        actions       [b, s] = batch[b][s].action         
                        #info          [b, s] = batch[b][s].policy_info            
                        next_step_type[b, s] = batch[b][s].next_step_type  
                        reward        [b, s] = batch[b][s].reward          
                        discount      [b, s] = batch[b][s].discount        

                traj = Trajectory( tf.constant(step_type, dtype=tf.int32),
                                   tf.constant(observation, dtype=tf.float32),
                                   tf.constant(actions, dtype=tf.float32),
                                   (),
                                   tf.constant(next_step_type, dtype=tf.int32),
                                   tf.constant(reward, dtype=tf.float32),
                                   tf.constant(discount, dtype=tf.float32))

                yield traj #Trajectory()









