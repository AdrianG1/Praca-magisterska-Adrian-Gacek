import tensorflow as tf
from environmentv3 import Environment
import os
import sys

from utils import plot_trajs
import optuna
import DQN_copy_2
from pandas import read_csv
from copy import deepcopy

from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_rnn_network

from utils import CustomReplayBuffer
from tf_agents.utils import common
import tensorflow as tf
import numpy as np
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
import multiprocessing
import functools
from tf_agents.system import multiprocessing
import warnings
warnings.filterwarnings('ignore')
from tf_agents.environments import ParallelPyEnvironment
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts
from tf_agents.train.utils import strategy_utils
from tf_agents.trajectories import Trajectory

TRAINING_STEPS = 10
BATCH_SIZE = 32
TRAIN_TEST_RATIO = 0.75



def evaluate_policy(agent, num_test_steps=1000):
    global test_buffer
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

        difference += abs(float((experience.action - action_step.action*25).numpy()[0]))
    return difference 



def training_agent(agent, train_iterator, num_episodes, steps_per_episode):
    min_diff = np.inf
    min_diff_ep = -1

    for episode in range(num_episodes):
        for _ in range(steps_per_episode):
            experience = next(train_iterator)
            train_loss = agent.train(experience).loss

        rating = evaluate_policy(agent)
        if rating < min_diff:
            min_diff = rating
            min_diff_ep = episode
     
    # print("\n\n\n min diff (ep): ", min_diff,  min_diff_ep, "\n")
    return min_diff


def objective(trial):
    global env, train_buffer, net_structure
    

    num_actor_input_layers = 2
    input_fc_layer_params = tuple([trial.suggest_int(f'input_layer{i}', 64, 256) for i in range(num_actor_input_layers)])
    
    num_lstm = 1
    lstm_size = (trial.suggest_int(f'lstm_layer', 16, 128),)
    
    num_actor_output_layers = 2
    output_fc_layer_params = tuple([trial.suggest_int(f'output_layer{i}', 64, 256) for i in range(num_actor_output_layers)])
    

    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)

    target_update_tau = trial.suggest_uniform('target_update_tau', 0.001, 0.02)
    target_update_period = trial.suggest_int('target_update_period', 1, 10)

    gamma = trial.suggest_uniform('gamma', 0.9, 1)    
    epsilon_greedy = trial.suggest_uniform('target_update_tau', 0.03, 0.2)

    n_step_update = 1
    activation_fn = trial.suggest_categorical('activation_fn', [tf.keras.activations.selu,
                                                                tf.keras.activations.relu,
                                                                tf.keras.activations.elu,
                                                                tf.keras.activations.tanh])


    agent = configure_agent(env, input_fc_layer_params, lstm_size, 
                            output_fc_layer_params, activation_fn,
                            learning_rate, epsilon_greedy, n_step_update, 
                            target_update_tau, target_update_period, 
                            gamma)
 
    train_iterator = train_buffer.get_iterator()
    max_rating = training_agent(agent, train_iterator, TRAINING_STEPS)
    del agent,
    return max_rating

def configure_agent(env, input_fc_layer_params, lstm_size, output_fc_layer_params, activation_fn,
                    learning_rate, epsilon_greedy, n_step_update, target_update_tau, target_update_period, 
                    gamma):

    q_net = q_rnn_network.QRnnNetwork(
        env.observation_spec(),
        env.action_spec(),
        input_fc_layer_params=input_fc_layer_params,
        lstm_size=lstm_size,
        output_fc_layer_params=output_fc_layer_params,
        activation_fn=activation_fn)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    agent = dqn_agent.DqnAgent(
        env.time_step_spec(),
        env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        epsilon_greedy = epsilon_greedy,
        n_step_update = n_step_update,
        target_update_tau = target_update_tau,
        target_update_period = target_update_period,
        gamma = gamma)

    
    agent.initialize()
    return agent
        

def discretize(action):
    return tf.constant((max(min(action, 100), 0)+12) // 25, dtype=tf.float32)
    

def create_environment():
    #return wrappers.ActionDiscretizeWrapper(Environment(), num_actions=5)
    return Environment(discret=True)



def get_trajectory_from_csv(path, state_dim, train_buffer, test_buffer, TRAIN_TEST_RATIO):
    df = read_csv(path, index_col=0)
    trajs = []
    train_end = len(df) * TRAIN_TEST_RATIO
    for idx, record in df.iterrows():
        state = record.iloc[:state_dim].values*100  # Convert to numpy array
        action = tf.constant(record["Akcje"], dtype=tf.float32)
        reward = record["Nagrody"]
        continous_action = tf.expand_dims(discretize(action), axis=-1)

        traj = Trajectory(tf.constant(1, dtype=tf.int32, shape=(1,)), 
                        tf.expand_dims(tf.constant(state, dtype=tf.float32), axis=0),
                        continous_action, 
                        (), 
                        tf.constant(1, dtype=tf.int32, shape=(1,)),
                        tf.constant(reward, dtype=tf.float32, shape=(1,)), 
                        tf.constant(0, dtype=tf.float32, shape=(1,)))
        trajs.append(traj)

        if  idx > train_end:
            test_buffer.append(traj)
        else:
            train_buffer.append(traj)






def main(argv=None):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Is GPU build:", tf.test.is_built_with_cuda())
    if argv is None:
        argv = []

    global env
    env = ParallelPyEnvironment([create_environment] * 1)
    env = tf_py_environment.TFPyEnvironment(env)

    agent = DQN_copy_2.configure_agent(env)

    agent.train = common.function(agent.train)

    global test_buffer, train_buffer, strategy
    train_buffer = CustomReplayBuffer(num_steps=12)

    test_buffer = []
    del agent
    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=True)

    print("================================== collecting data ===============================================")
    get_trajectory_from_csv("./csv_data/trajectory.csv", 2, train_buffer, test_buffer, TRAIN_TEST_RATIO)

    # collected_data_checkpoint = tf.train.Checkpoint(replay_buffer)
    # collected_data_checkpoint.save("./replay_buffers/replay_buffer")
    # collected_data_checkpoint.restore("./replay_buffers/replay_buffer-1")


    print("================================== optimizing ======================================================")
    original_stdout = sys.stdout
    with open('optuna_output.log', 'w') as f:
        # Redirect stdout to the file
        sys.stdout = f

        try:
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=100, catch=(ValueError,), n_jobs=1)
        except:
            pass   
        print("Best trial:")
        best_trial = study.best_trial
        print("  Value: {}".format(best_trial.value))
        print("  Params: ")
        for key, value in best_trial.params.items():
            print("    {}: {}".format(key, value))
        
        sys.stdout = original_stdout
        
    print("done")



if __name__ == '__main__':
    multiprocessing.handle_main(functools.partial(main))
    