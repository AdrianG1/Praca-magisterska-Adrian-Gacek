import tensorflow as tf
from environmentv3 import Environment
import os
import sys

from utils import plot_trajs, get_data_spec
import optuna
import SAC
from pandas import read_csv
from copy import deepcopy

from tf_agents.agents.ddpg.actor_rnn_network import ActorRnnNetwork
from tf_agents.agents.ddpg import critic_rnn_network

from tf_agents.agents.td3 import td3_agent
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

# params
TRAINING_STEPS = 15
TRAIN_TEST_RATIO = 0.75
num_steps_dataset = 2

def evaluate_policy2(agent):
    """
    Evaluates policy based off cumulative rewards

    :param agent: evaluated agent
    :return: cumulative reward 
    """
    
    reward = 0
    env = tf_py_environment.TFPyEnvironment(Environment(discret=False, episode_time=30, seed=5132))

    policy = deepcopy(agent.policy)
    policy_state = policy.get_initial_state(batch_size=1)
    time_step = env.reset()
    time_step = ts.TimeStep(
            step_type=tf.reshape(time_step.step_type, (1, )),
            reward=tf.reshape(time_step.reward, (1, )),
            discount=time_step.discount,
            observation= tf.reshape(time_step.observation, (1, 2))
            )
    
    while not time_step.is_last():
        action_step = policy.action(time_step, policy_state)
        policy_state = action_step.state
        time_step = env.step(action_step.action)
        time_step = ts.TimeStep(
                step_type=tf.reshape(time_step.step_type, (1, )),
                reward=tf.reshape(time_step.reward, (1, )),
                discount=time_step.discount,
                observation= tf.reshape(time_step.observation, (1, 2))
                )
        reward += time_step.reward

    return reward


def evaluate_policy(agent_original, num_test_steps=1000):
    """
    Evaluates policy based on difference between experienced PID response
    to given observations and agent response for the same observations.

    :param agent: evaluated agent
    :param num_test_steps: number of steps 
    :return: cumulative difference 
    """
    
    global test_buffer, net_structure
    difference = 0

    agent = deepcopy(agent_original)
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



def training_agent(agent, train_iterator, num_episodes):
    """
    Training agent with tested configuration

    :param train_iterator: generator returning experience (train data)
    :param num_episodes: number of episodes per training
    
    :return: max rating from evaluation 
    """
    
    steps_per_episode = 110
    max_rating = -np.inf
    max_rating_ep = -1

    for episode in range(num_episodes):
        for _ in range(steps_per_episode):
            experience, _ = next(train_iterator)
            train_loss = agent.train(experience).loss


        if episode in [9 , 14]:
            rating = evaluate_policy2(agent)
            if rating > max_rating:
                max_rating = rating
                max_rating_ep = episode
     
    print("\n\n\n min diff (ep): ", max_rating,  max_rating_ep, "\n")
    return max_rating


def objective(trial):
    """
    Optimized objective function necessary for optuna. Configures agent based on constraints, trains
    it and evaluate
    
    :param trial: trial optuna object
    :return: max rating from trial 
    """
    
    global env, train_buffer, net_structure
    
    try:
        batch_size_pow = trial.suggest_int('batch_size 2^', 3, 9)
        batch_size = 2**batch_size_pow

        # net structure
        num_actor_fc_input = trial.suggest_int('num_actor_fc_input', 0, 2)
        if  num_actor_fc_input == 3:
            actor_fc_input = (
                trial.suggest_int('actor_fc_input0', 64, 256),
                trial.suggest_int('actor_fc_input1', 64, 256),
                trial.suggest_int('actor_fc_input2', 32, 256)
            )
        if  num_actor_fc_input == 2:
            actor_fc_input = (
                trial.suggest_int('actor_fc_input0', 64, 256),
                trial.suggest_int('actor_fc_input1', 32, 256)
            )
        elif  num_actor_fc_input == 1:
            actor_fc_input = (
                trial.suggest_int('actor_fc_input0', 64, 256),
            )
        else:
            actor_fc_input = None


        actor_lstm = (
            trial.suggest_int('actor_lstm_size', 32, 128),
        )

        num_actor_fc_output = trial.suggest_int('num_actor_fc_output', 1, 2)
        if num_actor_fc_output == 3:
            actor_fc_output = (
                trial.suggest_int('actor_fc_output0', 64, 256),
                trial.suggest_int('actor_fc_output1', 64, 256),
                100 #trial.suggest_int('actor_fc_output1', 64, 256)
            )
        if num_actor_fc_output == 2:
            actor_fc_output = (
                trial.suggest_int('actor_fc_output0', 64, 256),
                100 #trial.suggest_int('actor_fc_output1', 64, 256)
            )
        else:
            actor_fc_output = (100,)

        num_critic_fc_input = trial.suggest_int('num_critic_fc_input', 0, 2)
        if num_critic_fc_input == 2:
            critic_fc_input = (
                trial.suggest_int('critic_fc_input0', 64, 256),
                trial.suggest_int('critic_fc_input1', 64, 256)
            )        
        elif num_critic_fc_input == 1:
            critic_fc_input = (trial.suggest_int('critic_fc_input0', 64, 256),)
        else:
            critic_fc_input = None

        critic_lstm = (
            trial.suggest_int('critic_lstm_size', 32, 128),
        )
        num_critic_fc_output = trial.suggest_int('num_critic_fc_output', 1, 2)
        if num_critic_fc_output == 3:
            critic_fc_output = (
                trial.suggest_int('critic_fc_output0', 64, 256),
                trial.suggest_int('critic_fc_output1', 64, 256),
                100 #trial.suggest_int('critic_fc_output1', 64, 256),
            )      
        elif num_critic_fc_output == 2:
            critic_fc_output = (
                trial.suggest_int('critic_fc_output0', 64, 256),
                100 #trial.suggest_int('critic_fc_output1', 64, 256),
            )
        else:
            critic_fc_output = (100,)

        activation_idx = trial.suggest_int('activation_fn', 0, 3)
        activation_fns = [tf.keras.activations.selu,tf.keras.activations.relu,
                        tf.keras.activations.elu,tf.keras.activations.tanh]
        activation_fn = activation_fns[activation_idx]


        actor_learning_rate = trial.suggest_loguniform('actor_learning_rate', 1e-6, 3e-3)
        critic_learning_rate_ratio = trial.suggest_loguniform('critic_learning_rate_ratio', 1e0, 1e2)
        critic_learning_rate = critic_learning_rate_ratio * actor_learning_rate

        target_update_tau = trial.suggest_uniform('target_update_tau', 0.001, 0.02)

        target_update_period = 1
        target_actor_period_difference = trial.suggest_int('target_actor_period_difference', 1, 6)
        actor_update_period = target_update_period + target_actor_period_difference

        gamma = trial.suggest_uniform('gamma', 0.88, 0.99)

        exploration_noise_std = trial.suggest_uniform('exploration_noise_std', 0.01, 0.3)

        num_steps_dataset = trial.suggest_int('num_steps_dataset', 2, 20)
        reward_scale_factor = trial.suggest_loguniform('reward_scale_factor', 1e-2, 1e2)
        




        agent = configure_agent(env, 
                            critic_learning_rate, actor_learning_rate,
                            target_update_tau, target_update_period, actor_update_period,
                            gamma, exploration_noise_std,
                            actor_fc_input, actor_lstm, actor_fc_output,
                            critic_lstm, critic_fc_output, critic_fc_input,
                            activation_fn, reward_scale_factor
                            )
    

        train_dataset = train_buffer.as_dataset(
            num_parallel_calls=3, 
            sample_batch_size=batch_size, 
            num_steps=num_steps_dataset).prefetch(3)

        train_iterator = iter(train_dataset)

        max_rating = training_agent(agent, train_iterator, TRAINING_STEPS)
        del agent, train_dataset, train_iterator

    except KeyboardInterrupt:
        raise ValueError("Trial stopped, rating undefined")

    return max_rating

def configure_agent(env, 
                        critic_learning_rate,actor_learning_rate,
                        target_update_tau, target_update_period, actor_update_period,
                        gamma, exploration_noise_std,
                        actor_fc_input, actor_lstm, actor_fc_output,
                        critic_lstm, critic_fc_output, critic_fc_input, 
                        activation_fn, reward_scale_factor
                        ):

    global strategy
    # Actor network
    with strategy.scope():
        actor_net = ActorRnnNetwork(
            env.observation_spec(),
            env.action_spec(),

            conv_layer_params=None,
            input_fc_layer_params=actor_fc_input, #actor_fc_input,#actor_fc_input,
            lstm_size=actor_lstm,
            output_fc_layer_params=actor_fc_output, #actor_fc_output #actor_fc_output
            activation_fn=activation_fn
        )

        critic_net = critic_rnn_network.CriticRnnNetwork(
            (env.observation_spec(), env.action_spec()),
            observation_conv_layer_params=None,
            observation_fc_layer_params=None,
            action_fc_layer_params=None,
            joint_fc_layer_params=critic_fc_input,
            lstm_size=critic_lstm,
            output_fc_layer_params=critic_fc_output, 
            activation_fn=activation_fn
        )

        actor_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=actor_learning_rate)
        critic_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=critic_learning_rate)

        agent = td3_agent.Td3Agent(
            env.time_step_spec(),
            env.action_spec(),
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            exploration_noise_std=exploration_noise_std,
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            actor_update_period=actor_update_period,
            td_errors_loss_fn=tf.math.squared_difference,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor
        )

        agent.initialize()
    return agent


def create_environment():
    return Environment(discret=False)


def get_trajectory_from_csv(path, state_dim, train_buffer, test_buffer, TRAIN_TEST_RATIO):
    """
    Fills replay buffer with Trajectories created from experience stored in csv file  
    
    :param path:            path to csv file
    :param state_dim:       number of dimensions of observation
    :param replay_buffer:   filled replay buffer 
    
    :return: list of trajectories useful for debugging               
    """
 
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
            test_buffer.append(traj)
        else:
            train_buffer.add_batch(traj)


def main(argv=None):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Is GPU build:", tf.test.is_built_with_cuda())
    if argv is None:
        argv = []

    global env
    env = ParallelPyEnvironment([create_environment] * 1)
    env = tf_py_environment.TFPyEnvironment(env)



    global test_buffer, train_buffer, strategy

    train_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                        data_spec=get_data_spec(),
                        batch_size=1,
                        max_length=20000)
    test_buffer = []

    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=True)

    print("================================== collecting data ===============================================")
    get_trajectory_from_csv("./csv_data/trajectory.csv", 2, train_buffer, test_buffer, TRAIN_TEST_RATIO)

    print("================================== optimizing ======================================================")
    original_stdout = sys.stdout
    with open('optuna_output.log', 'w') as f:
        # Redirect stdout to the file
        sys.stdout = f

        try:
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=100, catch=(ValueError,), n_jobs=1)
        except KeyboardInterrupt:
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
    