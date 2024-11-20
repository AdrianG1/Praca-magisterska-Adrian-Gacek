import tensorflow as tf
from environmentv3 import Environment
import os
from utils import evaluate_policy, plot_loss, plot_trajs, get_trajectory_from_csv, configure_tensorflow_logging

from tf_agents.agents.ddpg import critic_rnn_network
from tf_agents.agents.sac import sac_agent
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.utils import common
import tensorflow as tf
import numpy as np
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
import multiprocessing
import functools
from tf_agents.system import multiprocessing
import warnings
warnings.filterwarnings('ignore', )
from tf_agents.environments import ParallelPyEnvironment
from tf_agents.utils import common
from tf_agents.policies import policy_saver, policy_loader
from tqdm import tqdm
from tf_agents.trajectories import Trajectory
from tf_agents.train.utils import strategy_utils
from tf_agents.agents.sac import tanh_normal_projection_network


# data params
BATCH_SIZE          = 64
TRAIN_TEST_RATIO    = 1

# agent params
actor_learning_rate             = 1,88e-4
critic_learning_rate            = actor_learning_rate * 47.615404273158745
alpha_learning_rate             = 8.850215277629775e-05


actor_input_fc_layer_params     = (195,)
actor_lstm_size                 = (101,)
actor_output_fc_layer_params    = (100,)

critic_joint_fc_layer_params    = None
critic_lstm_size                = (35,)
critic_output_fc_layer_params   = (105, 100)

target_update_tau               = 0.031101832198767103
target_update_period            = 1
actor_update_period             = 3
gamma                           = 0.9674273939790276
reward_scale_factor             = 0.23014718662243797

activation_fn                   = tf.keras.activations.relu

train_sequence_length           = 12
num_episodes                    = 25


def configure_agent(env):
    """
    Configures SAC agent based on environment and listed configuration parameters.

    :param env: environment with specification
    :return: configured SAC agent 
    """
    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=True)
    
    # Actor network
    with strategy.scope():
        actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
                                env.observation_spec(),
                                env.action_spec(),
                                input_fc_layer_params=actor_input_fc_layer_params,
                                input_dropout_layer_params=None,
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
                target_update_tau=target_update_tau,
                target_update_period=target_update_period,
                td_errors_loss_fn=tf.math.squared_difference,
                actor_loss_weight=3,
                gamma=gamma,
                reward_scale_factor=reward_scale_factor,
                initial_log_alpha=0.5,

                train_step_counter=tf.Variable(0))

        agent.initialize()

    return agent



def create_environment():
    """
    Creates environment specifing data structure.

    :return: environment with configured spec for SAC 
    """
    
    return Environment(discret=False)


def main(argv=None):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Is GPU build:", tf.test.is_built_with_cuda())
    if argv is None:
        argv = []

    configure_tensorflow_logging()

    train_env = tf_py_environment.TFPyEnvironment(create_environment)

    agent = configure_agent(train_env)

    agent.train = common.function(agent.train)

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                        data_spec=agent.collect_data_spec,
                        batch_size=train_env.batch_size,
                        max_length=20000)
    del train_env
    
    print("================================== collecting data ===============================================")
    test_buffer = []

    trajs = get_trajectory_from_csv("./csv_data/trajectory_real.csv", 2, replay_buffer, test_buffer, TRAIN_TEST_RATIO)
    plot_trajs(trajs)

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, 
        sample_batch_size=BATCH_SIZE, 
        num_steps=train_sequence_length).prefetch(3)

    iterator = iter(dataset)

    print("================================== training ======================================================")
    # Run the training loop
    steps_per_episode = int(replay_buffer.num_frames()) // BATCH_SIZE
    losses = np.full(num_episodes*steps_per_episode+1, -1)
    # agent.policy.update(policy_loader.load("./policies/SAC40"))

    for episode in tqdm(range(num_episodes)):
        try:
            for _ in range(steps_per_episode):
                # Sample a batch of data from the buffer and update the agent's network.
                experience, unused_info = next(iterator)
                train_loss = agent.train(experience).loss
                step = agent.train_step_counter.numpy()
                losses[step] = train_loss
            tqdm.write('step = {0}: loss = {1}'.format(step, train_loss))
          
            saver = policy_saver.PolicySaver(agent.policy)
            os.makedirs(f'./policies/SAC_{episode}', exist_ok=True)
            saver.save(f'./policies/SAC_{episode}')
        except KeyboardInterrupt:
            break

    plot_loss(losses, num_episodes)
    saver = policy_saver.PolicySaver(agent.policy)
    saver.save('./policies/SAC')
    print("done")



if __name__ == '__main__':
    multiprocessing.handle_main(functools.partial(main))
    