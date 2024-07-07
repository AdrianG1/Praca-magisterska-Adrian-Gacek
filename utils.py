from collections import UserList
import numpy as np
from tf_agents.trajectories import Trajectory
# import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from pandas import read_csv
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import Trajectory
from tf_agents.trajectories.time_step import tensor_spec
import logging
import os
import warnings


def evaluate_policy(agent, test_buffer, num_test_steps=1000):
    """Evaluacja polityki poprzez porównanie ze sterowaniem PID"""
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


def abnormalize_state(state):
    """ 
        nienormalizacja stanu (odwrócenie normalizacji)

        Zwraca temperaturę i setpoint [T, T_sp]
    """
    # T_diff = (state[..., 1]-0.5)*100
    # T_sp = state[..., 0]*100
    T_diff = state[..., 1]
    T_sp = state[..., 0]
    T = T_sp - T_diff
    return np.reshape(np.stack([T, T_sp], axis=-1), state.shape) 


def normalize_state(state):
    """ normalizacja stanu """
    T_diff = (state[..., 1] / 200 + 0.5)
    T = state[..., 0] / 100
    return np.reshape(np.stack([T,T_diff], axis=-1), state.shape) 


def discretize(action, num_actions=5):
    """dyskretyzacja dla DQN"""

    a = (50/(num_actions-1))
    b = (100/(num_actions-1))
    clipped_action = max(min(action, 100), 0)
    return tf.constant((clipped_action + a) // b, dtype=tf.float32)
    
def undiscretize(action, num_actions=5):
    """odwrócenie dyskretyzacji dla DQN"""
    return tf.constant(100/(num_actions-1) * action)    

def plot_trajs(trajs):
    """ Plot kontrolny wczytywanej trajektorii"""
    states = np.squeeze(np.array([traj.observation for traj in trajs]))
    actions = np.squeeze(np.array([traj.action for traj in trajs]))
    rewards = np.squeeze(np.array([traj.reward for traj in trajs]))

    plt.figure()
    plt.plot(range(len(states)), states[:, 0:2])
    plt.title('trajectory states')
    plt.savefig('./plot/states_trajs.png')    
    plt.figure()
    plt.plot(range(len(states)), rewards)
    plt.title('trajectory reward')
    plt.savefig('./plot/reward_trajs.png')
    plt.figure()
    plt.plot(range(len(actions)), actions)
    plt.title('trajectory actions')
    plt.savefig('./plot/actions_trajs.png')


def get_data_spec():
    """ generacja specyfikacji trajektorii z pominięciem agenta """
    return Trajectory(  tensor_spec.TensorSpec(shape=(), dtype=np.int32, name='step_type'), 
                        tensor_spec.TensorSpec(shape=(2,), dtype=np.float32, name='observation'),
                        tensor_spec.BoundedTensorSpec(shape=(), dtype=np.float32, minimum=0, maximum=100.0, name='action'), 
                        (), 
                        tensor_spec.TensorSpec(shape=(), dtype=np.int32, name='next_step_type'),
                        tensor_spec.TensorSpec(shape=(), dtype=np.float32, name='reward'), 
                        tensor_spec.BoundedTensorSpec(shape=(), dtype=np.float32, name='discount', minimum=0.0, maximum=1.0)
                    )


def plot_loss(losses, num_episodes=0):
    plt.figure()
    plt.plot(losses, "b")

    if num_episodes > 0:
        n = len(losses)//num_episodes
        mean_loss_for_episode = [np.mean(losses[i:i+n]) for i in range(0, len(losses), n)]

        plt.plot(range(0, len(losses), n), mean_loss_for_episode, "r", linewidth=2)
        plt.axis((0, len(losses), min(mean_loss_for_episode), max(mean_loss_for_episode)))

    plt.title('losses')
    plt.savefig('./plot/losses.png')


def get_trajectory_from_csv(path, state_dim, replay_buffer, test_buffer, train_test_ratio, discount=0.75):
    """ Wczytywanie danych z csv """
    df = read_csv(path, index_col=0)

    trajs = []

    train_end = len(df) * train_test_ratio
    for idx, record in df.iterrows():

        state = record.iloc[:state_dim].values  # Convert to numpy array
        action = tf.constant(record["Akcje"]/100, dtype=tf.float32)
        reward = record["Nagrody"]
        continous_action = tf.expand_dims(tf.clip_by_value(action, 0, 100), axis=-1)

        traj = Trajectory(tf.constant(1, dtype=tf.int32, shape=(1,)), 
                        tf.expand_dims(tf.constant(state, dtype=tf.float32), axis=0),
                        continous_action, 
                        (), 
                        tf.constant(1, dtype=tf.int32, shape=(1,)),
                        tf.constant(reward, dtype=tf.float32, shape=(1,)), 
                        tf.constant(discount, dtype=tf.float32, shape=(1,)))
        trajs.append(traj)


        if  idx < train_end: 
            replay_buffer.add_batch(traj)
        else:
            test_buffer.append(traj)

    return trajs


class CustomReplayBuffer(UserList):
    """
    Alternatywny replay buffer, który ma za zadanie bardziej równomiernie wykorzystywać dane uczące
    wykorzystując w losowej kolejności wszystkie dane tyle samo razy.
    """
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



def configure_tensorflow_logging():

    open('tensorflow_info.log', 'w').close()

    # Ustaw poziom logowania TensorFlow na INFO, aby przechwytywać wszystkie logi
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    tf.get_logger().setLevel('INFO')

    # Skonfiguruj własny logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Utwórz handler dla logów poniżej poziomu ERROR, który zapisuje do pliku
    file_handler = logging.FileHandler('tensorflow_info.log')
    file_handler.setLevel(logging.INFO)
    file_handler.addFilter(lambda record: record.levelno < logging.ERROR)

    # Utwórz handler dla logów ERROR i wyższych, które są wyświetlane normalnie
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)

    # Dodaj formatery do handlerów (opcjonalne, ale zalecane)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Usuń domyślne handlery, aby uniknąć podwójnych logów
    for handler in logger.handlers:
        logger.removeHandler(handler)

    # Dodaj handlery do loggera
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    tf_logger = tf.get_logger()
    for handler in tf_logger.handlers:
        tf_logger.removeHandler(handler)
    tf_logger.addHandler(file_handler)
    tf_logger.addHandler(console_handler)

    def custom_warning(message, category, filename, lineno, file=None, line=None):
        log_message = f'{filename}:{lineno}: {category.__name__}: {message}\n'
        logger.warning(log_message)

    warnings.showwarning = custom_warning

