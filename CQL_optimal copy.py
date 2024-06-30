

import tensorflow as tf
from environmentv3 import Environment
from utils import *
from copy import deepcopy
import sys
import optuna

from tf_agents.agents.ddpg import critic_rnn_network
from tf_agents.agents.cql import cql_sac_agent
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

BATCH_SIZE = 32
TRAINING_STEPS = 12
STEPS_PER_EPISODE = 15000 // BATCH_SIZE // 5
TRAIN_TEST_RATIO = 0.75
n_trials = 50
EVALUATE_LIST = [7, 9, 11]


def evaluate_policy(agent, num_test_steps=1000):
    global test_buffer
    difference = 0

    policy_state = agent.policy.get_initial_state(batch_size=1)
    for i in range(0, len(test_buffer), len(test_buffer)//num_test_steps):
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
        if difference > 5000:
            return 5000
    return difference 


def training_agent(agent, train_iterator, num_episodes, steps_per_episode):
    min_diff = np.inf
    min_diff_ep = -1
    start_training_time = time()

    for episode in range(num_episodes):
        for _ in range(steps_per_episode):
            experience, _ = next(train_iterator)
            train_loss = agent.train(experience).loss

        if episode in EVALUATE_LIST:
            rating = evaluate_policy(agent)
            print(f"episode: {episode} difference: {rating}")
            if rating < min_diff:
                min_diff = rating
                min_diff_ep = episode
        
        print(f"training_time: {time() - start_training_time}") 
     
    # print("\n\n\n min diff (ep): ", min_diff,  min_diff_ep, "\n")
    return min_diff


def objective(trial):
    global env, train_buffer
    
    # net structure
    actor_input_fc_layer_params = (
        trial.suggest_int('actor_fc_input0', 128, 256),
        trial.suggest_int('actor_fc_input1', 64, 256)
    )
    
    actor_input_dropout_layer_params=None

    actor_lstm_size = (
        trial.suggest_int('actor_lstm_size', 32, 128),
    )
    actor_output_fc_layer_params = (
        trial.suggest_int('actor_fc_output0', 64, 256),
        100 #trial.suggest_int('actor_fc_output1', 64, 256)
    )

    if trial.suggest_categorical('include_critic_joint_fc_layer_params', [True, False]):
        critic_joint_fc_layer_params = (
            trial.suggest_int('critic_fc_input0', 64, 256),
            trial.suggest_int('critic_fc_input1', 64, 256)
        )
    else:
        critic_joint_fc_layer_params = None

    critic_lstm_size = (
        trial.suggest_int('critic_lstm_size', 32, 128),
    )
    critic_output_fc_layer_params = (
        trial.suggest_int('critic_fc_output0', 64, 256),
        100 #trial.suggest_int('critic_fc_output1', 64, 256),
    )

    activation_idx = trial.suggest_int('activation_fn', 0, 3)
    activation_fns = [tf.keras.activations.selu,tf.keras.activations.relu,
                      tf.keras.activations.elu,tf.keras.activations.tanh]
    activation_fn = activation_fns[activation_idx]

    #learning rates
    actor_learning_rate = trial.suggest_loguniform('actor_learning_rate', 1e-5, 1e-3)
    actor_critic_lr_ratio = trial.suggest_int('actor_critic_lr_ratio', 1, 100) 
    critic_learning_rate = (1e-3 - actor_learning_rate) * actor_critic_lr_ratio / 100 + actor_learning_rate
    alpha_learning_rate = trial.suggest_loguniform('alpha_learning_rate', 1e-5, 1e-3)

    # other params
    target_update_tau = trial.suggest_uniform('target_update_tau', 0.001, 0.02)
    target_update_period = trial.suggest_int('target_update_period', 1, 2)
    gamma = trial.suggest_uniform('gamma', 0.8, 0.99)

    num_cql_samples = trial.suggest_int('num_cql_samples', 1, 50)
    cql_alpha = trial.suggest_uniform('cql_alpha', 0.70, 0.99)
    include_critic_entropy_term = trial.suggest_categorical('include_critic_entropy_term', [True, False])
    use_lagrange_cql_alpha = trial.suggest_categorical('use_lagrange_cql_alpha', [True, False])

    num_steps_dataset = trial.suggest_int('num_steps_dataset', 4, 20)
    




    agent = configure_agent(env,
                    actor_input_fc_layer_params, actor_input_dropout_layer_params, actor_lstm_size,
                    actor_output_fc_layer_params, activation_fn,
                    critic_joint_fc_layer_params, critic_lstm_size, critic_output_fc_layer_params,
                    actor_learning_rate, critic_learning_rate, alpha_learning_rate,
                    cql_alpha, include_critic_entropy_term, num_cql_samples, use_lagrange_cql_alpha,
                    target_update_tau, target_update_period,gamma
                    )
 
    agent.train = common.function(agent.train)

    train_dataset = train_buffer.as_dataset(
        num_parallel_calls=3, 
        sample_batch_size=BATCH_SIZE, 
        num_steps=num_steps_dataset).prefetch(3)

    train_iterator = iter(train_dataset)
    # steps_per_episode = int(train_buffer.num_frames()) // BATCH_SIZE
    min_difference = training_agent(agent, train_iterator, TRAINING_STEPS, STEPS_PER_EPISODE)
    del agent, train_dataset, train_iterator
    return min_difference



def configure_agent(env,
                    actor_input_fc_layer_params, actor_input_dropout_layer_params, actor_lstm_size,
                    actor_output_fc_layer_params, activation_fn,
                    critic_joint_fc_layer_params, critic_lstm_size, critic_output_fc_layer_params,
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
                                input_fc_layer_params=actor_input_fc_layer_params,
                                input_dropout_layer_params=actor_input_dropout_layer_params,
                                lstm_size=actor_lstm_size,
                                output_fc_layer_params=actor_output_fc_layer_params,
                                activation_fn=activation_fn)

        
        # Critic network
        critic_net = critic_rnn_network.CriticRnnNetwork(
            (env.observation_spec(), env.action_spec()),
            observation_conv_layer_params=None,
            observation_fc_layer_params=None,
            action_fc_layer_params=None,
            joint_fc_layer_params=critic_joint_fc_layer_params,
            lstm_size=critic_lstm_size,
            output_fc_layer_params=critic_output_fc_layer_params,
            activation_fn=activation_fn
        )

        agent = cql_sac_agent.CqlSacAgent(
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
    env = tf_py_environment.TFPyEnvironment(create_environment())

    global test_buffer, train_buffer

    train_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                        data_spec=get_data_spec(),
                        batch_size=env.batch_size,
                        max_length=20000)
    test_buffer = []

    print("================================== collecting data ===============================================")
    get_trajectory_from_csv("./csv_data/trajectory.csv", 2, train_buffer, test_buffer, TRAIN_TEST_RATIO)

    print("================================== optimizing ======================================================")

    # study = optuna.load_study(study_name="my_study", storage=storage)


    original_stdout = sys.stdout
    with open('optuna_output.log', 'w') as f:
        # Redirect stdout to the file
        sys.stdout = f
        try:
            study = optuna.create_study(direction='minimize', storage='sqlite:///optuna_study.db')
            study.optimize(objective, n_trials=n_trials, catch=(ValueError,), n_jobs=1, show_progress_bar=True)
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
    main()
    # multiprocessing.handle_main(functools.partial(main))