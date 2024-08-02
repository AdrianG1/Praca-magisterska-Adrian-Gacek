from environmentv3 import Environment
from utils import CustomReplayBuffer, discretize, plot_trajs, plot_loss, configure_tensorflow_logging
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
from tf_agents.policies import policy_saver, policy_loader
from tqdm import tqdm
from tf_agents.trajectories import Trajectory
from tf_agents.specs import array_spec
import os
from tqdm import tqdm

BATCH_SIZE = 256
DISCOUNT = 0.75
TRAIN_TEST_RATIO = 1
NUM_ACTIONS = 5

learning_rate             = 0.0006653782419255851/2

input_fc_layer_params     = (209, 148)
lstm_size                 = (77,)
output_fc_layer_params    = (216, 100)


target_update_tau               = 0.005041497519914242
actor_update_period             = 2
target_update_period            = 1
epsilon_greedy                  = 0.1
gamma                           = 0.962232033742456 
reward_scale_factor             = 0.9874855517385459 

activation_fn                   = tf.keras.activations.selu

train_sequence_length = 4
num_episodes = 25

def create_environment():
    #return wrappers.ActionDiscretizeWrapper(Environment(), num_actions=5)
    return Environment(discret=True, num_actions=NUM_ACTIONS)



def configure_agent(env):
    q_net = q_rnn_network.QRnnNetwork(
            env.observation_spec(),
            env.action_spec(),
            input_fc_layer_params=input_fc_layer_params,
            lstm_size=lstm_size,
            output_fc_layer_params=output_fc_layer_params,
            activation_fn=activation_fn)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    agent = dqn_agent.DqnAgent(
        env.time_step_spec(),
        env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        epsilon_greedy = epsilon_greedy,
        n_step_update = 1,
        target_update_tau = target_update_tau,
        target_update_period = target_update_period,
        gamma = gamma)

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
        discrete_action = tf.expand_dims(discretize(action, num_actions=NUM_ACTIONS), axis=-1)
        reward = record["Nagrody"] - action/(state[0]/100)/3.5/3

        traj = Trajectory(tf.constant(1, dtype=tf.int32, shape=(1,)), 
                        tf.expand_dims(tf.constant(state, dtype=tf.float32), axis=0),
                        discrete_action, 
                        (), 
                        tf.constant(1, dtype=tf.int32, shape=(1,)),
                        tf.constant(reward, dtype=tf.float32, shape=(1,)), 
                        tf.constant(DISCOUNT, dtype=tf.float32, shape=(1,)))
        # replay_buffer.append(traj)
        trajs.append(traj)
        replay_buffer.add_batch(traj)

        if  idx > train_end:
            break
    return trajs


def main(argv=None):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Is GPU build:", tf.test.is_built_with_cuda())
    if argv is None:
        argv = []
    configure_tensorflow_logging()

    env = ParallelPyEnvironment([create_environment] * 1)
    train_env = tf_py_environment.TFPyEnvironment(env)
    agent = configure_agent(train_env)
    agent.initialize()
    # (Optional) Reset the agent's policy state
    agent.train = common.function(agent.train)

    # replay_buffer = CustomReplayBuffer(num_steps=train_sequence_length)

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                        data_spec=agent.collect_data_spec,
                        batch_size=train_env.batch_size,
                        max_length=20000)
    del train_env, env

    print("================================== collecting data ===============================================")
    trajs = get_trajectory_from_csv("./csv_data/trajectory7.csv", 2, replay_buffer)
    plot_trajs(trajs)
    # Dataset generates trajectories with shape [Bx2x...]
    # iterator = replay_buffer.get_iterator()
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, 
        sample_batch_size=BATCH_SIZE, 
        num_steps=train_sequence_length).prefetch(3)

    iterator = iter(dataset)

    print("================================== training ======================================================")
    # Run the training loop
    num_episodes = 5
    # steps_per_episode = int(len(replay_buffer)) // BATCH_SIZE *2
    steps_per_episode = int(replay_buffer.num_frames()) // BATCH_SIZE*4

    losses = []
    agent.policy.update(policy_loader.load("./policies/DQN_2_32"))

    for episode in tqdm(range(num_episodes)):
        try:
            for _ in range(steps_per_episode):
                # Sample a batch of data from the buffer and update the agent's network.
                experience, info = next(iterator)
                train_loss = agent.train(experience).loss
                step = agent.train_step_counter.numpy()
                losses.append(train_loss)
            tqdm.write('step = {0}: loss = {1}'.format(step, train_loss))

            policy2 = agent.post_process_policy()
            saver = policy_saver.PolicySaver(policy2)
            os.makedirs(f'./policies/DQN__{episode}', exist_ok=True)
            saver.save(f'./policies/DQN__{episode}')
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
    