import tensorflow as tf
from environmentv3 import Environment
import os
import sys

import optuna
from copy import deepcopy

from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import q_rnn_network
from tf_agents.policies import policy_saver, policy_loader
from tf_agents.drivers import py_driver
from tf_agents.policies import py_tf_eager_policy
import reverb
from tf_agents.specs import tensor_spec
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.utils import common
import tensorflow as tf
import numpy as np
from tf_agents.environments import tf_py_environment
import multiprocessing
import functools
from tf_agents.system import multiprocessing
import warnings
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.environments import ParallelPyEnvironment

from utils import get_data_spec
warnings.filterwarnings('ignore')
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts
from tf_agents.train.utils import strategy_utils

# params
TRAINING_STEPS = 20
BATCH_SIZE = 32
TRAIN_TEST_RATIO = 0.75
POLICY_LOAD_PATH = "DQN_2_30"
buffer_size = 1000
CONNECTED = True

def evaluate_policy(agent):
    """
    Evaluates policy based on difference between experienced PID response
    to given observations and agent response for the same observations.

    :param agent: evaluated agent
    :param num_test_steps: number of steps 
    :return: cumulative difference 
    """

    env = tf_py_environment.TFPyEnvironment(Environment(discret=True, connected=CONNECTED, 
                                                        episode_time=60, seed=5132))

    policy = deepcopy(agent.policy)

    diff = 0
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
        diff += abs(time_step.observation[0, 1])
        if diff > 5000:
            return 6000

    return diff



def training_agent(agent, train_sequence_length):
    """
    Training agent with tested configuration

    :param train_iterator: generator returning experience (train data)
    :param num_episodes: number of episodes per training
    
    :return: max rating from evaluation 
    """
    
    train_env = create_environment()
    train_py_env = tf_py_environment.TFPyEnvironment(train_env)
    
    table_name = 'uniform_table'
    replay_buffer_signature = tensor_spec.from_spec(
        agent.collect_data_spec)
    replay_buffer_signature = tensor_spec.add_outer_dim(
        replay_buffer_signature)

    table = reverb.Table(
        table_name,
        max_size=buffer_size,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=replay_buffer_signature)

    reverb_server = reverb.Server([table])

    replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
                                agent.collect_data_spec,
                                table_name=table_name,
                                sequence_length=train_sequence_length,
                                local_server=reverb_server)

    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
                                replay_buffer.py_client,
                                table_name,
                                sequence_length=train_sequence_length)
     

    dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=BATCH_SIZE,
            num_steps=train_sequence_length).prefetch(3)
    
    iterator = iter(dataset)

    # Create a driver to collect experience.
    collect_driver = py_driver.PyDriver(
        train_env,
        py_tf_eager_policy.PyTFEagerPolicy(
        agent.collect_policy, use_tf_function=True, batch_time_steps=True),
        [rb_observer],
        max_steps=BATCH_SIZE*4)
    
    time_step = train_env.reset()

    policy_state = agent.policy.get_initial_state(batch_size=1)

    for i in range(TRAINING_STEPS):
        time_step, policy_state = collect_driver.run(time_step, policy_state)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, _ = next(iterator)
        train_loss = agent.train(experience).loss

        if i in [10, 19]:
            saver = policy_saver.PolicySaver(agent.policy)
            os.makedirs(f'./policies/eval{i}', exist_ok=True)
            saver.save(f'./policies/eval{i}')

    del train_env

def objective(trial):
    """
    Optimized objective function necessary for optuna. Configures agent based on constraints, trains
    it and evaluate
    
    :param trial: trial optuna object
    :return: max rating from trial 
    """
    
    global env, train_buffer

    input_fc_layer_params     = (209, 148)
    lstm_size                 = (77,)
    output_fc_layer_params    = (216, 100)
    learning_rate = 0.00033 * trial.suggest_loguniform('learning_rate_fraction', 1e-2, 1e0)

    target_update_tau = 0.005041497519914242
    target_update_period = 1

    gamma = trial.suggest_uniform('gamma', 0.9, 1)    
    epsilon_greedy = trial.suggest_uniform('epsilon_greedy', 0.03, 0.3)

    n_step_update = 1
    activation_fn = tf.keras.activations.selu
    train_sequence_length = trial.suggest_int('train_sequence_length', 2, 10)


    agent = configure_agent(env, input_fc_layer_params, lstm_size, 
                            output_fc_layer_params, activation_fn,
                            learning_rate, epsilon_greedy, n_step_update, 
                            target_update_tau, target_update_period, 
                            gamma)
    
    tf_policy = policy_loader.load(f'./policies/{POLICY_LOAD_PATH}')    
    agent.policy.update(tf_policy)

    training_agent(agent, train_sequence_length)

    min_diff = np.inf

    for i in [10, 19]:
        tf_policy = policy_loader.load(f'./policies/eval{i}')    
        agent.policy.update(tf_policy)
        rating = evaluate_policy(agent)
        if rating < min_diff:
            min_diff = rating

    del agent, 
    return min_diff

def configure_agent(env, input_fc_layer_params, lstm_size, output_fc_layer_params, activation_fn,
                    learning_rate, epsilon_greedy, n_step_update, target_update_tau, target_update_period, 
                    gamma):
    """
    Configures DQN agent based on environment and passed parameters.
    
    :param env:
    :params ...: parameters of DQN agent
    :return: configured DQN agent 
    """

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
    return Environment(discret=True, num_actions=5, episode_time=99999999,
                        connected=CONNECTED, seed=198999)


def main(argv=None):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Is GPU build:", tf.test.is_built_with_cuda())
    if argv is None:
        argv = []

    global env
    env = ParallelPyEnvironment([create_environment] * 1)
    env = tf_py_environment.TFPyEnvironment(env)



    global train_buffer, strategy

    train_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                        data_spec=get_data_spec(),
                        batch_size=1,
                        max_length=8000)

    global strategy
    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=True)


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
    