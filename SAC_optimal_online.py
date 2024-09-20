from tf_agents.environments import ParallelPyEnvironment
import tensorflow as tf
from environmentv3 import Environment
from utils import *
from copy import deepcopy
import sys
import optuna

from tf_agents.agents.ddpg import critic_rnn_network
from tf_agents.agents.sac import sac_agent
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.utils import common
import tensorflow as tf
import numpy as np
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
import warnings
warnings.filterwarnings('ignore')
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts
from tf_agents.train.utils import strategy_utils
import functools
from tf_agents.system import multiprocessing
from time import time


TRAIN_TEST_RATIO = 0.75
TRAINING_STEPS = 20


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

    return reward


def training_agent(agent, train_iterator, num_episodes):
    steps_per_episode = 110
    max_rating = -np.inf
    max_rating_ep = -1

    for episode in range(num_episodes):
        for _ in range(steps_per_episode):
            experience, _ = next(train_iterator)
            train_loss = agent.train(experience).loss


        if episode in [14, 19]:
            rating = evaluate_policy2(agent)
            if rating > max_rating:
                max_rating = rating
                max_rating_ep = episode
     
    print("\n\n\n min diff (ep): ", max_rating,  max_rating_ep, "\n")
    return max_rating


def objective(trial):
    global env, train_buffer
    try:
        # num_of_layers = trial.suggest_int('num_of_layers', 2, 8)
        batch_size_pow = trial.suggest_int('batch_size 2^', 3, 9)
        batch_size = 2**batch_size_pow
        num_steps_dataset = trial.suggest_int('num_steps_dataset', 2, 20)

        # net structure
        actor_fc_input                  = (195,)
        actor_lstm                      = (101,)
        actor_fc_output                 = (100,)

        critic_fc_input                 = None
        critic_lstm                     = (35,)
        critic_fc_output                = (105, 100)

        target_update_tau               =  0.031101832198767103
        target_update_period            = 1
        gamma                           = 0.9674273939790276
        reward_scale_factor             = 0.23014718662243797

        actor_learning_rate = trial.suggest_loguniform('actor_learning_rate', 1e-6, 3e-3)
        critic_learning_rate_ratio = trial.suggest_loguniform('critic_learning_rate_ratio', 1e0, 1e2)
        critic_learning_rate = critic_learning_rate_ratio * actor_learning_rate
        alpha_learning_rate = trial.suggest_loguniform('alpha_learning_rate', 1e-6, 3e-3)

        td_errors_loss_fn = tf.math.squared_difference
        
        agent = configure_agent(env, 
                                critic_learning_rate,actor_learning_rate, alpha_learning_rate,
                                target_update_tau, target_update_period,
                                gamma, reward_scale_factor,
                                actor_fc_input, actor_lstm, actor_fc_output,
                                critic_lstm, critic_fc_output, critic_fc_input,
                                td_errors_loss_fn)
    
        agent.train = common.function(agent.train)

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
                    actor_fc_input, actor_lstm, actor_fc_output,
                    critic_lstm, critic_fc_output, critic_fc_input,
                    activation_fn,
                    actor_learning_rate, critic_learning_rate, alpha_learning_rate,
                    cql_alpha, include_critic_entropy_term, num_cql_samples, use_lagrange_cql_alpha,
                    target_update_tau, target_update_period,gamma
                    ):

    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=True)
    
    # Actor network
    with strategy.scope():

        actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
                                env.observation_spec(),
                                env.action_spec(),
                                input_fc_layer_params=actor_fc_input,
                                input_dropout_layer_params=None,
                                lstm_size=actor_lstm,
                                output_fc_layer_params=actor_fc_output,
                                activation_fn=activation_fn)

        
        # Critic network
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

        agent = sac_agent.SacAgent(
                env.time_step_spec(),
                env.action_spec(),
                actor_network=actor_net,
                critic_network=critic_net,
                actor_optimizer=tf.keras.optimizers.Adam(
                    learning_rate=actor_learning_rate),
                critic_optimizer=tf.keras.optimizers.Adam(
                    learning_rate=critic_learning_rate),
                alpha_optimizer=tf.keras.optimizers.Adam(
                    learning_rate=alpha_learning_rate),
                cql_alpha=cql_alpha,
                include_critic_entropy_term=include_critic_entropy_term,
                num_cql_samples=num_cql_samples,
                use_lagrange_cql_alpha=use_lagrange_cql_alpha,
                target_update_tau=target_update_tau,
                target_update_period=target_update_period,
                td_errors_loss_fn=tf.math.squared_difference,
                gamma=gamma)
            
    
        agent.initialize()

    return agent



def create_environment():
    return Environment(discret=False)


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
    
    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=True)


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
    # main()
    multiprocessing.handle_main(functools.partial(main))