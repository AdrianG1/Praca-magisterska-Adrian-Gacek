from environmentv3 import Environment
from PID import PID
import tensorflow as tf
import numpy as np
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
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

BATCH_SIZE = 1
BATCH_SIZE_EVAL = 1
LEARNING_RATE = 2e-4
DISCOUNT = 0.75

global trajs
trajs = []

def plot_trajs():
    global trajs
    states = np.squeeze(np.array([traj.observation for traj in trajs]))
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(range(len(states)), states[:, 0:2])
    plt.title('losses')
    plt.savefig('./plot/states_trajs.png')

def plot_loss(losses):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(range(len(losses)), losses)
    plt.title('losses')
    plt.savefig('./plot/losses.png')


def discretize(action):
    return tf.constant((max(min(action, 100), 0)+12) // 25, dtype=tf.float32)
    

def create_environment():
    #return wrappers.ActionDiscretizeWrapper(Environment(), num_actions=5)
    return Environment(discret=True)


def collect_step(environment, control_function, replay_buffer):

    time_step = environment.current_time_step()
    action = control_function(time_step.observation, environment.envs[0].time) #TODO time
    discrete_action = tf.expand_dims(discretize(action), axis=-1)
    next_time_step = environment.step(discrete_action)

    traj = Trajectory(time_step.step_type, 
                      time_step.observation, 
                      discrete_action, 
                      (), 
                      next_time_step.step_type, 
                      next_time_step.reward, 
                      next_time_step.discount)
    replay_buffer.add_batch(traj)

    global trajs
    trajs.append(traj)

    if next_time_step.is_last():
        environment.reset()
    

def collect_data(env, control_function, replay_buffer, steps):
    for _ in tqdm(range(steps)):
        collect_step(env, control_function, replay_buffer)
        return 0


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for i in range(num_episodes):
        print(f"episode:{i}")
        time_step = environment.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


def configure_agent(env):
    fc_layer_params = (75, 75, 75, 75, 75, 75)
    q_net = q_network.QNetwork(
        env.observation_spec(),
        env.action_spec(),
        fc_layer_params=fc_layer_params
        )

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
        env.time_step_spec(),
        env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)
    
    return agent

def get_trajectory_from_csv(path, state_dim, replay_buffer):
    from pandas import read_csv
    df = read_csv(path, index_col=0)
    global trajs
    trajs = []
    for _, record in df.iterrows():

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
        replay_buffer.add_batch(traj)
        trajs.append(traj) 


def main(argv=None):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Is GPU build:", tf.test.is_built_with_cuda())
    if argv is None:
        argv = []

    
    env = ParallelPyEnvironment([create_environment] * BATCH_SIZE)
    train_env = tf_py_environment.TFPyEnvironment(env)

    eval_py_env = ParallelPyEnvironment([create_environment] * (BATCH_SIZE_EVAL))
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    agent = configure_agent(train_env)
    agent.initialize()
    # (Optional) Reset the agent's policy state
    agent.train = common.function(agent.train)

    pid = PID()
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                        data_spec=agent.collect_data_spec,
                        batch_size=train_env.batch_size,
                        max_length=11000)

    print("================================== collecting data ===============================================")
    #collect_data(train_env, pid.control, replay_buffer, steps=4000)
    get_trajectory_from_csv("./csv_data/trajectory.csv", 6, replay_buffer)
    plot_trajs()

    collected_data_checkpoint = tf.train.Checkpoint(replay_buffer)
    collected_data_checkpoint.save("./replay_buffers/replay_buffer")
    # return 0
    # collected_data_checkpoint.restore("./replay_buffers/replay_buffer-1")

    # Dataset generates trajectories with shape [Bx2x...]
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, 
        sample_batch_size=8, 
        num_steps=2).prefetch(3)

    iterator = iter(dataset)

    print("================================== training ======================================================")
    
    # Run the training loop
    num_iterations = 4000

    losses = np.full(num_iterations, -1)

    for i in (range(num_iterations)):

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss
        losses[i] = train_loss
        step = agent.train_step_counter.numpy()
        if step % 200 == 0:
            #print('step = {0}: loss = {1}'.format(step, train_loss))
            tqdm.write('step = {0}: loss = {1}'.format(step, train_loss))

        if step % 50000 == 0:
            # Evaluate the agent's policy once in a while
            avg_return = compute_avg_return(eval_env, agent.policy, num_episodes=5)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))

    plot_loss(losses)
    saver = policy_saver.PolicySaver(agent.policy)
    saver.save('./policies/DQN')
    env.close()
    eval_py_env.close()
    print("done")



if __name__ == '__main__':
    multiprocessing.handle_main(functools.partial(main))
    