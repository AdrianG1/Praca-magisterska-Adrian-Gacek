from environmentv3 import Environment
from utils import CustomReplayBuffer
from PID import PID
import tensorflow as tf
import numpy as np
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_rnn_network

from tf_agents.replay_buffers import tf_uniform_replay_buffer
import multiprocessing
import functools
from tf_agents.system import multiprocessing
import warnings
warnings.filterwarnings('ignore')
from tf_agents.environments import ParallelPyEnvironment, wrappers
from tf_agents.utils import common
from tf_agents.policies import policy_saver
from tqdm import tqdm
from tf_agents.trajectories import Trajectory
from tf_agents.specs import array_spec
import os

BATCH_SIZE = 32
BATCH_SIZE_EVAL = 1
LEARNING_RATE = 2e-4
DISCOUNT = 0.75
TRAIN_TEST_RATIO = 0.75
NUM_STEPS_DATASET = 2


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


def discretize(action):
    return tf.constant((max(min(action, 100), 0)+12) // 25, dtype=tf.float32)
    

def create_environment():
    #return wrappers.ActionDiscretizeWrapper(Environment(), num_actions=5)
    return Environment(discret=True)



def configure_agent(env):
    q_net = q_rnn_network.QRnnNetwork(
            env.observation_spec(),
            env.action_spec(),
            input_fc_layer_params=(200, 100),
            lstm_size=(40, ),
            output_fc_layer_params=(200, 100),
            activation_fn=tf.keras.activations.selu)

    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-4)

    agent = dqn_agent.DqnAgent(
        env.time_step_spec(),
        env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        epsilon_greedy = 0.1,
        n_step_update = 1,
        target_update_tau = 0.01,
        target_update_period = 2,
        gamma = 0.99)

    return agent

def get_trajectory_from_csv(path, state_dim, replay_buffer):
    from pandas import read_csv
    df = read_csv(path, index_col=0)

    global trajs
    trajs = []
    
    global actions_representation
    actions_representation = {}

    train_end = len(df) * TRAIN_TEST_RATIO
    for idx, record in df.iterrows():

        state = record.iloc[:state_dim].values  # Convert to numpy array
        action = record["Akcje"]
        reward = record["Nagrody"]
        discrete_action = tf.expand_dims(discretize(action), axis=-1)

        traj = Trajectory(tf.constant(1, dtype=tf.int32, shape=(1,)), 
                        tf.expand_dims(tf.constant(state, dtype=tf.float32), axis=0),
                        discrete_action, 
                        (), 
                        tf.constant(1, dtype=tf.int32, shape=(1,)),
                        tf.constant(reward, dtype=tf.float32, shape=(1,)), 
                        tf.constant(DISCOUNT, dtype=tf.float32, shape=(1,)))
        replay_buffer.append(traj)
        trajs.append(traj)

        if  idx > train_end:
            break


def main(argv=None):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Is GPU build:", tf.test.is_built_with_cuda())
    if argv is None:
        argv = []

    env = ParallelPyEnvironment([create_environment] * 1)
    train_env = tf_py_environment.TFPyEnvironment(env)
    agent = configure_agent(train_env)
    del train_env, env
    agent.initialize()
    # (Optional) Reset the agent's policy state
    agent.train = common.function(agent.train)

    replay_buffer = CustomReplayBuffer(num_steps=12)

    print("================================== collecting data ===============================================")
    get_trajectory_from_csv("./csv_data/trajectory.csv", 2, replay_buffer)
    plot_trajs()

    # Dataset generates trajectories with shape [Bx2x...]
    replay_buffer.compile()

    print("================================== training ======================================================")
    # Run the training loop
    num_episodes = 25
    steps_per_episode = int(len(replay_buffer)) // BATCH_SIZE
    losses = np.full(num_episodes*steps_per_episode+1, -1)

    for episode in range(num_episodes):
        try:
            for _ in range(steps_per_episode):
                # Sample a batch of data from the buffer and update the agent's network.
                experience = replay_buffer.get_sample()
                train_loss = agent.train(experience).loss
                step = agent.train_step_counter.numpy()
                losses[step] = train_loss
            tqdm.write('step = {0}: loss = {1}'.format(step, train_loss))

            saver = policy_saver.PolicySaver(agent.policy)
            os.makedirs(f'./policies/DQN{episode}', exist_ok=True)
            saver.save(f'./policies/DQN{episode}')
        except KeyboardInterrupt:
            break

    plot_loss(losses, num_episodes)
    saver = policy_saver.PolicySaver(agent.policy)
    saver.save('./policies/DQN')
    # env.close()
    # eval_py_env.close()
    print("done")



if __name__ == '__main__':
    multiprocessing.handle_main(functools.partial(main))
    