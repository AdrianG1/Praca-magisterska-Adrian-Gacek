import tensorflow as tf
from environmentv3 import Environment
import os
from utils import evaluate_policy, plot_loss, plot_trajs, get_trajectory_from_csv, configure_tensorflow_logging

from tf_agents.agents.ddpg import critic_rnn_network
from tf_agents.agents.cql import cql_sac_agent
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
warnings.filterwarnings('ignore')
from tf_agents.utils import common
from tf_agents.policies import policy_saver
from tqdm import tqdm
from tf_agents.trajectories import Trajectory
from tf_agents.train.utils import strategy_utils

BATCH_SIZE = 256
DISCOUNT = 0.75
TRAIN_TEST_RATIO = 0.5

num_episodes = 10
train_sequence_length = 10

actor_learning_rate = 4.618152654403827e-05
critic_learning_rate = 25* actor_learning_rate
alpha_learning_rate = 7.400528836439677e-05
cql_alpha_learning_rate = 1e-5

cql_alpha= 0.280423024569609
include_critic_entropy_term=True
num_cql_samples=13
use_lagrange_cql_alpha=True

target_update_tau = 0.003415103446262748 
target_update_period = 1 
gamma = 0.855869890833244 
reward_scale_factor = 1

actor_input_fc_layer_params         =(134,) 
actor_lstm_size                     =(103,)
actor_output_fc_layer_params        =(100,)
actor_activation_fn                 =tf.keras.activations.selu

critic_joint_fc_layer_params=None
critic_lstm_size=                   (40,)
critic_output_fc_layer_params           =(100, 100)
critic_activation_fn=tf.keras.activations.selu



def configure_agent(env):

    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=True)
    
    # Actor network
    with strategy.scope():
        # flatten = tf.keras.layers.Flatten()

        actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
                                env.observation_spec(),
                                env.action_spec(),
                                input_fc_layer_params=actor_input_fc_layer_params,
                                input_dropout_layer_params=None,
                                lstm_size=actor_lstm_size,
                                output_fc_layer_params=actor_output_fc_layer_params,
                                activation_fn=actor_activation_fn)

        # combined_actor = sequential.Sequential([flatten, actor_net])
        
        # Critic network
        critic_net = critic_rnn_network.CriticRnnNetwork(
            (env.observation_spec(), env.action_spec()),
            observation_conv_layer_params=None,
            observation_fc_layer_params=None,
            action_fc_layer_params=None,
            joint_fc_layer_params=critic_joint_fc_layer_params,
            lstm_size=critic_lstm_size,
            output_fc_layer_params=critic_output_fc_layer_params,
            activation_fn=critic_activation_fn
        )
        # combined_critic = sequential.Sequential([flatten, critic_net])


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
                cql_alpha_learning_rate=cql_alpha_learning_rate,
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
    configure_tensorflow_logging()

    train_env = tf_py_environment.TFPyEnvironment(create_environment())

    agent = configure_agent(train_env)

    agent.train = common.function(agent.train)

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                        data_spec=agent.collect_data_spec,
                        batch_size=1,
                        max_length=20000)
    train_env.close()
    del train_env

    print("================================== collecting data ===============================================")
    test_buffer = []

    trajs = get_trajectory_from_csv("./csv_data/trajectory7.csv", 2, replay_buffer, test_buffer, TRAIN_TEST_RATIO)
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
    steps_per_episode = int(replay_buffer.num_frames()) // BATCH_SIZE * 3
    losses = np.full(num_episodes*steps_per_episode+1, -1)

    for episode in tqdm(range(num_episodes)):
        try:
            for _ in range(steps_per_episode):
                # Sample a batch of data from the buffer and update the agent's network.
                experience, unused_info = next(iterator)
                train_loss = agent.train(experience).loss
                step = agent.train_step_counter.numpy()
                losses[step] = train_loss
            tqdm.write('step = {0}: loss = {1}'.format(step, train_loss))
            # if episode % 5 == 0:
            #     tqdm.write('evaluated difference = {0}:\n'.format(evaluate_policy(agent, test_buffer)))


            saver = policy_saver.PolicySaver(agent.policy)
            os.makedirs(f'./policies/CQL{episode}', exist_ok=True)
            saver.save(f'./policies/CQL{episode}')

            collected_data_checkpoint = tf.train.Checkpoint(agent)
            collected_data_checkpoint.save(f"./policies/CCQL{episode}")
        except KeyboardInterrupt:
            break
            print("next")

    plot_loss(losses, num_episodes)
    saver = policy_saver.PolicySaver(agent.policy)
    saver.save('./policies/CQL')
    print("done")



if __name__ == '__main__':
    multiprocessing.handle_main(functools.partial(main))