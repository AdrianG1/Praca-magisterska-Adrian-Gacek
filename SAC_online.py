import tensorflow as tf
from environmentv3 import Environment
# from environmentv3_transmission import Environment

import os
from utils import plot_loss, configure_tensorflow_logging 
from tf_agents.policies import py_tf_eager_policy

from tf_agents.drivers import py_driver
import reverb
from tf_agents.specs import tensor_spec
from tf_agents.agents.ddpg import critic_rnn_network
from tf_agents.agents.sac import sac_agent
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.utils import common
import tensorflow as tf
import numpy as np
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
import multiprocessing
import functools
from tf_agents.system import multiprocessing
import warnings
warnings.filterwarnings('ignore')
from tf_agents.utils import common
from tf_agents.policies import policy_saver, policy_loader
from tqdm import tqdm
from tf_agents.train.utils import strategy_utils

# data params
POLICY_LOAD_ID      = 8

BATCH_SIZE          = 256
TRAIN_TEST_RATIO    = 0.75

# agent params
num_episodes                    = 60
train_sequence_length           = 4
actor_learning_rate             = 3.76e-05
critic_learning_rate            = actor_learning_rate * 47.6063829787234
alpha_learning_rate             = 2.95e-06


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

e_coef                          = 0
buffer_size                     = 2000


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
                gamma=gamma,
                reward_scale_factor=reward_scale_factor,
                initial_log_alpha=0.5,
                train_step_counter=tf.Variable(0))

        agent.initialize()

    return agent


def create_environment():
    return Environment(discret=False, episode_time=999999, connected=True, env_step_time=1,
                       scaler_path=None, c_coef=1, e_coef=e_coef, log_steps=False, seed=64547778)


def main(argv=None):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Is GPU build:", tf.test.is_built_with_cuda())
    if argv is None:
        argv = []

    train_env = create_environment()
    train_py_env = tf_py_environment.TFPyEnvironment(train_env)
    agent = configure_agent(train_py_env)

    agent.initialize()
    tf_policy = policy_loader.load(f'./policies/SAC-{POLICY_LOAD_ID}')    
    agent.policy.update(tf_policy)

    agent.train = common.function(agent.train)


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
        max_steps=BATCH_SIZE)
    
    time_step = train_env.reset()



    policy_state = agent.policy.get_initial_state(batch_size=1)
    print("================================== training ======================================================")
    configure_tensorflow_logging()
    # Run the training loop
    steps_per_episode = 1
    losses = []

    for episode in tqdm(range(num_episodes)):
        try:
            sum_diff = 0
            sum_reward = 0
            for i in range(steps_per_episode):
                # Collect a few steps and save to the replay buffer.
                time_step, policy_state = collect_driver.run(time_step, policy_state)

                # Sample a batch of data from the buffer and update the agent's network.
                experience, _ = next(iterator)
                train_loss = agent.train(experience).loss

                step = agent.train_step_counter.numpy()
                losses.append(train_loss)

                sum_diff += np.sum(np.abs(experience.observation[:, 1]))/np.sum(experience.observation[:, 1].shape)
                sum_reward += np.sum(np.abs(experience.reward))/np.sum(experience.reward.shape)
                tqdm.write('step = {0}: loss = {1}, env time = {2}'.format(step, train_loss, train_env.time))

            tqdm.write('episode = {0}: sum difference = {1}'.format(episode, sum_diff))
            saver = policy_saver.PolicySaver(agent.policy)
            os.makedirs(f'./policies/SAC_online{episode}', exist_ok=True)
            saver.save(f'./policies/SAC_online{episode}')
        except KeyboardInterrupt:
            break


    plot_loss(losses, num_episodes)
    saver = policy_saver.PolicySaver(agent.policy)
    saver.save('./policies/SAC')
    print("done")


if __name__ == '__main__':
    multiprocessing.handle_main(functools.partial(main))
    