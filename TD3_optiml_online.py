import tensorflow as tf
from environmentv3 import Environment
import sys

import optuna
from copy import deepcopy
from tf_agents.train.utils import strategy_utils

from tf_agents.agents.ddpg.actor_rnn_network import ActorRnnNetwork
from tf_agents.agents.ddpg import critic_rnn_network

from tf_agents.agents.td3 import td3_agent
from tf_agents.utils import common
import tensorflow as tf
import numpy as np
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.drivers import py_driver
import reverb
from tf_agents.specs import tensor_spec
import multiprocessing
import functools
from tf_agents.system import multiprocessing
import warnings
warnings.filterwarnings('ignore')
from tf_agents.environments import ParallelPyEnvironment
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts
from tf_agents.policies import policy_loader
from tf_agents.policies import py_tf_eager_policy

TRAINING_STEPS = 50
TRAIN_TEST_RATIO = 0.75
POLICY_LOAD_ID = 21

actor_input_fc_layer_params     = (209, 148)
actor_lstm_size                 = (77,)
actor_output_fc_layer_params    = (216, 100)

critic_joint_fc_layer_params    = None
critic_lstm_size                = (124,)
critic_output_fc_layer_params   = (96, 100)

def evaluate_policy2(agent):
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

    return -reward

def evaluate_policy(policy):
    difference = 0
    env = tf_py_environment.TFPyEnvironment(Environment(discret=False, episode_time=30, seed=5132))

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
        difference += np.abs(time_step.observation[0, 1])

    return difference


def training_agent(agent, train_env, batch_size, buffer_size, num_steps_dataset, num_episodes):
    steps_per_episode = 10
    min_diff = np.inf
    min_diff_ep = -1


    table_name = 'uniform_table'
    replay_buffer_signature = tensor_spec.from_spec(
        agent.collect_data_spec)
    replay_buffer_signature = tensor_spec.add_outer_dim(
        replay_buffer_signature)

    table = reverb.Table(
        table_name,
        max_size=int(buffer_size),
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=replay_buffer_signature)

    reverb_server = reverb.Server([table])

    replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
                                agent.collect_data_spec,
                                table_name=table_name,
                                sequence_length=int(num_steps_dataset),
                                local_server=reverb_server)

    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
                                replay_buffer.py_client,
                                table_name,
                                sequence_length=int(num_steps_dataset))
    
    # random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
    #                                             train_env.action_spec())
    
    dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=int(batch_size),
            num_steps=int(num_steps_dataset)).prefetch(3)
    
    iterator = iter(dataset)
    # Create a driver to collect experience.
    collect_driver = py_driver.PyDriver(
        train_env,
        py_tf_eager_policy.PyTFEagerPolicy(
        agent.collect_policy, use_tf_function=True, batch_time_steps=True),
        [rb_observer],
        max_steps=int(batch_size))
    
    time_step = train_env.reset()
    policy_state = agent.policy.get_initial_state(batch_size=1)

    for episode in range(num_episodes):
        for _ in range(steps_per_episode):
            # Collect a few steps and save to the replay buffer.
            time_step, policy_state = collect_driver.run(time_step, policy_state)

            # Sample a batch of data from the buffer and update the agent's network.
            experience, _ = next(iterator)
            train_loss = agent.train(experience).loss


        if episode in [30, 40, 49]:
            rating = evaluate_policy(agent.policy)
            if rating < min_diff:
                min_diff = rating
                min_diff_ep = episode

    del train_env, reverb_server, replay_buffer, collect_driver
     
    print("\n\n\n min diff (ep): ", min_diff,  min_diff_ep, "\n")
    return min_diff


def objective(trial):

    try:
        c_coef = 1
        e_coef = trial.suggest_int('e_coef', 0, 200)
        normalization_flag = trial.suggest_int('normalization_flag', 0, 1)
        scaler_path = 'scaler.pkl' if normalization_flag else None

        batch_size_pow = trial.suggest_int('batch_size 2^', 2, 5)
        batch_size = 2**batch_size_pow
        num_steps_dataset = trial.suggest_int('num_steps_dataset', 2, 20)

        activation_idx = trial.suggest_int('activation_fn', 0, 3)
        activation_fns = [tf.keras.activations.selu,tf.keras.activations.relu,
                        tf.keras.activations.elu,tf.keras.activations.tanh]
        activation_fn = activation_fns[activation_idx]


        actor_learning_rate = trial.suggest_loguniform('actor_learning_rate', 1e-5, 3e-3)
        critic_learning_rate_ratio = trial.suggest_loguniform('critic_learning_rate_ratio', 1e0, 1e2)
        critic_learning_rate = critic_learning_rate_ratio * actor_learning_rate

        target_update_tau               = 0.011791682151010022
        actor_update_period             = 5
        target_update_period            = 1
        gamma                           =  0.962232033742456
        reward_scale_factor             = 0.9874855517385459

        exploration_noise_std = trial.suggest_uniform('exploration_noise_std', 0.001, 0.3)

        buffer_size = trial.suggest_loguniform('buffer_size', batch_size, 500)


        train_env = Environment(discret=False, episode_time=999999, 
                                seed=667, scaler_path=scaler_path, c_coef=c_coef, e_coef=e_coef)
        train_py_env = tf_py_environment.TFPyEnvironment(train_env)


        agent = configure_agent(train_py_env, 
                            critic_learning_rate, actor_learning_rate,
                            target_update_tau, target_update_period, actor_update_period,
                            gamma, exploration_noise_std,
                            actor_input_fc_layer_params, actor_lstm_size, actor_output_fc_layer_params, 
                            
                            critic_joint_fc_layer_params, critic_lstm_size, critic_output_fc_layer_params,
                            activation_fn, reward_scale_factor
                            )
    

        agent.initialize()
        tf_policy = policy_loader.load(f'./policies/td3{POLICY_LOAD_ID}')    
        agent.policy.update(tf_policy)

        agent.train = common.function(agent.train)
        max_rating = training_agent(agent, train_env, batch_size, buffer_size, num_steps_dataset, TRAINING_STEPS)
        del agent

    except KeyboardInterrupt:
        raise ValueError("Trial stopped, rating undefined")

    return max_rating

def configure_agent(env, 
                        critic_learning_rate,actor_learning_rate,
                        target_update_tau, target_update_period, actor_update_period,
                        gamma, exploration_noise_std,
                        actor_fc_input, actor_lstm, actor_fc_output,
                        critic_fc_input, critic_lstm, critic_fc_output, 
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


def main(argv=None):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Is GPU build:", tf.test.is_built_with_cuda())
    if argv is None:
        argv = []

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
    