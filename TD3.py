import tensorflow as tf
from environmentv3 import Environment
import os
from utils import plot_loss, plot_trajs, get_trajectory_from_csv, configure_tensorflow_logging
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
from tf_agents.environments import ParallelPyEnvironment
from tf_agents.utils import common
from tf_agents.policies import policy_saver, policy_loader
from tqdm import tqdm
from tf_agents.trajectories import Trajectory
from tf_agents.train.utils import strategy_utils
from tf_agents.agents.sac import tanh_normal_projection_network


BATCH_SIZE = 32
TRAIN_TEST_RATIO = 0.75

critic_learning_rate = 3e-4
actor_learning_rate = 3e-5
alpha_learning_rate = 3e-5



train_sequence_length = 12
num_episodes = 25

def configure_agent(env):

    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=True)

    # Actor network
    with strategy.scope():
        actor_net = ActorRnnNetwork(
            env.observation_spec(),
            env.action_spec(),

            conv_layer_params=None,
            input_fc_layer_params=(200, 100),
            lstm_size=(75,),
            output_fc_layer_params=(200, 100),
            activation_fn=tf.keras.activations.relu
        )

        critic_net = critic_rnn_network.CriticRnnNetwork(
            (env.observation_spec(), env.action_spec()),
            observation_conv_layer_params=None,
            observation_fc_layer_params=None,
            action_fc_layer_params=None,
            joint_fc_layer_params=None,
            lstm_size=(75,),
            output_fc_layer_params=(200, 100),
            activation_fn=tf.keras.activations.relu
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
            exploration_noise_std=0.1,
            target_update_tau=0.005,
            target_update_period=1,
            actor_update_period=2,
            gamma=0.99,
            reward_scale_factor=1,
            td_errors_loss_fn=tf.math.squared_difference
        )

        agent.initialize()
    return agent


def create_environment():
    return Environment(discret=False)


def main(argv=None):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Is GPU build:", tf.test.is_built_with_cuda())
    configure_tensorflow_logging()
    if argv is None:
        argv = []

    env = ParallelPyEnvironment([create_environment] * 1)
    train_env = tf_py_environment.TFPyEnvironment(env)

    agent = configure_agent(train_env)

    agent.train = common.function(agent.train)
    # agent.policy.update(policy_loader.load("./policies/td324"))

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                        data_spec=agent.collect_data_spec,
                        batch_size=train_env.batch_size,
                        max_length=10000)
    env.close()
    del train_env, env

    print("================================== collecting data ===============================================")
    trajs = get_trajectory_from_csv("./csv_data/trajectory.csv", 2, replay_buffer, [], TRAIN_TEST_RATIO, discount=0)
    plot_trajs(trajs)

    # collected_data_checkpoint = tf.train.Checkpoint(replay_buffer)
    # # collected_data_checkpoint.save("./replay_buffers/replay_buffer")
    # collected_data_checkpoint.restore("./replay_buffers/replay_buffer-1")

    # Dataset generates trajectories with shape [Bx2x...]
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, 
        sample_batch_size=BATCH_SIZE, 
        num_steps=train_sequence_length).prefetch(3)

    iterator = iter(dataset)

    print("================================== training ======================================================")
    # Run the training loop
    steps_per_episode = int(replay_buffer.num_frames()) // BATCH_SIZE
    losses = []
    for episode in tqdm(range(num_episodes)):
        try:

            sum_loss = 0

            for _ in range(steps_per_episode):
                # Sample a batch of data from the buffer and update the agent's network.
                experience, unused_info = next(iterator)
                train_loss = agent.train(experience).loss
                step = agent.train_step_counter.numpy()
                losses.append(train_loss)
                sum_loss += train_loss
                
            tqdm.write('step = {0}: mean loss = {1}'.format(step, sum_loss/steps_per_episode))

            saver = policy_saver.PolicySaver(agent.policy)
            os.makedirs(f'./policies/td3{episode}', exist_ok=True)
            saver.save(f'./policies/td3{episode}')
        except KeyboardInterrupt:
            break
            print("next")


    plot_loss(losses, num_episodes)
    saver = policy_saver.PolicySaver(agent.policy)
    saver.save('./policies/td3')
    print("done")



if __name__ == '__main__':
    multiprocessing.handle_main(functools.partial(main))