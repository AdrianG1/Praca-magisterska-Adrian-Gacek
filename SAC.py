import tensorflow as tf
from environmentv3 import Environment
import os
from utils import evaluate_policy, CustomReplayBuffer

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
warnings.filterwarnings('ignore')
from tf_agents.environments import ParallelPyEnvironment
from tf_agents.utils import common
from tf_agents.policies import policy_saver
from tqdm import tqdm
from tf_agents.trajectories import Trajectory
from tf_agents.train.utils import strategy_utils
from tf_agents.agents.sac import tanh_normal_projection_network

BATCH_SIZE = 32
BATCH_SIZE_EVAL = 1
LEARNING_RATE = 2e-4
DISCOUNT = 0.75
TRAIN_TEST_RATIO = 0.75
NUM_STEPS_DATASET = 2

num_episodes = 25
train_sequence_length = 12
batch_size = 256 # @param {type:"integer"}

critic_learning_rate = 3e-4 # @param {type:"number"}
actor_learning_rate = 3e-5 # @param {type:"number"}
alpha_learning_rate = 3e-5 # @param {type:"number"}

target_update_tau = 0.005 # @param {type:"number"}
target_update_period = 10 # @param {type:"number"}
gamma = 0.99 # @param {type:"number"}
reward_scale_factor = 1.0 # @param {type:"number"}

actor_fc_layer_params = (75, 75, 75, 75)
critic_joint_fc_layer_params =(75, 75, 75, 75)


global trajs
trajs = []


def plot_trajs():
    global trajs
    states = np.squeeze(np.array([traj.observation for traj in trajs]))
    actions = np.squeeze(np.array([traj.action for traj in trajs]))
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(range(len(states)), states[:, 0:2])
    plt.title('losses')
    plt.savefig('./plot/states_trajs.png')
    plt.figure()
    plt.plot(range(len(actions)), actions)
    plt.title('trajectory actions')
    plt.savefig('./plot/actions_trajs.png')

def plot_loss(losses, num_episodes=0):
    import matplotlib.pyplot as plt

    plt.figure()
    # plt.plot(range(len(losses)), losses)
    if num_episodes > 0:
        n = len(losses)//num_episodes
        mean_loss_for_episode = [np.mean(losses[i:i+n]) for i in range(0, len(losses), n)]
        plt.plot(range(0, len(losses), n), mean_loss_for_episode)
    plt.title('losses')
    plt.savefig('./plot/losses.png')


def configure_agent(env):

    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=True)
    
    # Actor network
    with strategy.scope():
        actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
                                env.observation_spec(),
                                env.action_spec(),
                                input_fc_layer_params=(200, 100),
                                input_dropout_layer_params=None,
                                lstm_size=(75,),
                                output_fc_layer_params=(200, 100),
                                activation_fn=tf.keras.activations.selu)


    # Critic network
        critic_net = critic_rnn_network.CriticRnnNetwork(
            (env.observation_spec(), env.action_spec()),
            observation_conv_layer_params=None,
            observation_fc_layer_params=None,
            action_fc_layer_params=None,
            joint_fc_layer_params=None,
            lstm_size=(75,),
            output_fc_layer_params=(200, 100)
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
                train_step_counter=tf.Variable(0))

    
        agent.initialize()

    return agent

def get_trajectory_from_csv(path, state_dim, replay_buffer, test_buffer):
    from pandas import read_csv
    df = read_csv(path, index_col=0)

    global trajs
    trajs = []
    
    global actions_representation
    actions_representation = {}

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
                        tf.constant(DISCOUNT, dtype=tf.float32, shape=(1,)))
        trajs.append(traj)


        if  idx < train_end:    # TODO przetestować idx < train_end
            replay_buffer.add_batch(traj)
        else:
            test_buffer.append(traj)



def create_environment():
    return Environment(discret=False)


def main(argv=None):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Is GPU build:", tf.test.is_built_with_cuda())
    if argv is None:
        argv = []

    env = ParallelPyEnvironment([create_environment] * 1)
    train_env = tf_py_environment.TFPyEnvironment(env)

    agent = configure_agent(train_env)

    agent.train = common.function(agent.train)

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                        data_spec=agent.collect_data_spec,
                        batch_size=train_env.batch_size,
                        max_length=20000)
    env.close()
    del train_env, env
    
    print("================================== collecting data ===============================================")
    test_buffer = []

    get_trajectory_from_csv("./csv_data/trajectory1.csv", 2, replay_buffer, test_buffer)
    plot_trajs()

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
    losses = np.full(num_episodes*steps_per_episode+1, -1)

    for episode in tqdm(range(num_episodes)):
        try:
            for _ in range(steps_per_episode):
                # Sample a batch of data from the buffer and update the agent's network.
                experience, unused_info = next(iterator)
                train_loss = agent.train(experience).loss
                step = agent.train_step_counter.numpy()
                losses[step] = train_loss
            tqdm.write('\nstep = {0}: loss = {1}\n'.format(step, train_loss))
            if episode % 5 == 0:
                tqdm.write('evaluated difference = {0}:\n'.format(evaluate_policy(agent, test_buffer)))


            saver = policy_saver.PolicySaver(agent.policy)
            os.makedirs(f'./policies/SAC{episode}', exist_ok=True)
            saver.save(f'./policies/SAC{episode}')
        except KeyboardInterrupt:
            break
            print("next")

    plot_loss(losses, num_episodes)
    saver = policy_saver.PolicySaver(agent.policy)
    saver.save('./policies/SAC')
    print("done")



if __name__ == '__main__':
    multiprocessing.handle_main(functools.partial(main))
    