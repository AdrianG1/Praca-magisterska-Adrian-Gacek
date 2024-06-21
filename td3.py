import tensorflow as tf
from environmentv3 import Environment
import os

from tf_agents.agents.ddpg.actor_network import ActorNetwork
from tf_agents.agents.ddpg import critic_network

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

critic_learning_rate = 1.302372409818221e-05 # @param {type:"number"}
actor_learning_rate = 0.0001189958432635629 # @param {type:"number"}
alpha_learning_rate = 3e-4 # @param {type:"number"}

target_update_tau = 0.02151287295318593 # @param {type:"number"}
target_update_period = 10 # @param {type:"number"}
gamma = 0.9182588601932395 # @param {type:"number"}
reward_scale_factor = 0.3946973854302325 # @param {type:"number"}

actor_fc_layer_params = (148, 180, 196, 202, 193)
critic_joint_fc_layer_params =(108, 132, 238, 105, 141)

def configure_agent(env):


    actor_net = ActorNetwork(
        env.observation_spec(),
        env.action_spec(),
        fc_layer_params=actor_fc_layer_params
    )

    critic_net = critic_network.CriticNetwork(
        (env.observation_spec(), env.action_spec()),
        observation_fc_layer_params=None,
        action_fc_layer_params=None,
        joint_fc_layer_params=critic_joint_fc_layer_params
    )

    actor_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=actor_learning_rate)
    critic_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=critic_learning_rate)

    train_step_counter = tf.Variable(0)

    agent = td3_agent.Td3Agent(
        env.time_step_spec(),
        env.action_spec(),
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        exploration_noise_std=0.1,
        target_update_tau=0.005,
        target_update_period=2,
        actor_update_period=2,
        #td_errors_loss_fn=common.element_wise_squared_loss,
        gamma=0.99,
        reward_scale_factor=1.0,
        train_step_counter=train_step_counter
    )

    agent.initialize()
    return agent

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


def get_trajectory_from_csv(path, state_dim, replay_buffer):
    from pandas import read_csv
    df = read_csv(path, index_col=0)

    global trajs
    trajs = []
    
    global actions_representation
    actions_representation = {}

    train_end = len(df) #* TRAIN_TEST_RATIO
    for idx, record in df.iterrows():

        state = record.iloc[:state_dim].values  # Convert to numpy array
        action = tf.constant(record["Akcje"], dtype=tf.float32)
        reward = record["Nagrody"]
        continous_action = tf.expand_dims(tf.clip_by_value(action, 0, 100), axis=-1)

        traj = Trajectory(tf.constant(1, dtype=tf.int32, shape=(1,)), 
                        tf.expand_dims(tf.constant(state*100, dtype=tf.float32), axis=0),
                        continous_action, 
                        (), 
                        tf.constant(1, dtype=tf.int32, shape=(1,)),
                        tf.constant(reward, dtype=tf.float32, shape=(1,)), 
                        tf.constant(DISCOUNT, dtype=tf.float32, shape=(1,)))
        replay_buffer.add_batch(traj)
        trajs.append(traj)

        if  idx > train_end:
            break

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
                        max_length=6000)
    env.close()
    del train_env, env

    print("================================== collecting data ===============================================")
    get_trajectory_from_csv("./csv_data/trajectory.csv", 2, replay_buffer)
    plot_trajs()

    collected_data_checkpoint = tf.train.Checkpoint(replay_buffer)
    collected_data_checkpoint.save("./replay_buffers/replay_buffer")
    # collected_data_checkpoint.restore("./replay_buffers/replay_buffer-1")

    # Dataset generates trajectories with shape [Bx2x...]
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, 
        sample_batch_size=BATCH_SIZE, 
        num_steps=2).prefetch(3)

    iterator = iter(dataset)

    print("================================== training ======================================================")
    # Run the training loop
    num_episodes = 50
    steps_per_episode = int(replay_buffer.num_frames()) // BATCH_SIZE
    losses = np.full(num_episodes*steps_per_episode+1, -1)

    for episode in range(num_episodes):
        try:
            for _ in range(steps_per_episode):
                # Sample a batch of data from the buffer and update the agent's network.
                experience, unused_info = next(iterator)
                train_loss = agent.train(experience).loss
                step = agent.train_step_counter.numpy()
                losses[step] = train_loss
            tqdm.write('step = {0}: loss = {1}'.format(step, train_loss))

            saver = policy_saver.PolicySaver(agent.policy)
            os.makedirs(f'./policies/td3{episode}', exist_ok=True)
            saver.save(f'./policies/td3{episode}')
        except KeyboardInterrupt:
            print("next")


    plot_loss(losses, num_episodes)
    saver = policy_saver.PolicySaver(agent.policy)
    saver.save('./policies/td3')
    print("done")



if __name__ == '__main__':
    multiprocessing.handle_main(functools.partial(main))