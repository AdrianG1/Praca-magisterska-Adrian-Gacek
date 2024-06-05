from environmentv3 import Environment
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
from tf_agents.trajectories import trajectory
from tf_agents.policies import policy_saver
from tqdm import tqdm

BATCH_SIZE = 1
BATCH_SIZE_EVAL = 1
LEARNING_RATE = 2e-3

def create_environment():
    #return wrappers.ActionDiscretizeWrapper(Environment(), num_actions=5)
    return Environment(discret=True)


def collect_step(environment, policy, replay_buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    replay_buffer.add_batch(traj)

    if next_time_step.is_last():
        environment.reset()
    print()
    

def collect_data(env, policy, replay_buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, replay_buffer)


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

def check_q_values(agent, time_step, action):
    q_values = agent._q_network(time_step.observation, time_step.step_type)
    print(tf.math.is_nan(q_values[action]), tf.math.is_inf(q_values[action]))
    
def main(argv=None):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Is GPU build:", tf.test.is_built_with_cuda())
    if argv is None:
        argv = []

    
    env = ParallelPyEnvironment([create_environment] * BATCH_SIZE)
    train_env = tf_py_environment.TFPyEnvironment(env)

    eval_py_env = ParallelPyEnvironment([create_environment] * (BATCH_SIZE_EVAL))
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    fc_layer_params = (100, 50, 75, 75, 75, 75)
    q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params
        )

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)

    agent.initialize()

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                        data_spec=agent.collect_data_spec,
                        batch_size=train_env.batch_size,
                        max_length=2000)
    collect_data(train_env, agent.collect_policy, replay_buffer, steps=1)
    
    # Dataset generates trajectories with shape [Bx2x...]
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, 
        sample_batch_size=8, 
        num_steps=2).prefetch(3)

    iterator = iter(dataset)

    # (Optional) Reset the agent's policy state
    agent.train = common.function(agent.train)
    print("==================================training======================================================")
    # Run the training loop
    num_iterations = 1000

    try:
        for _ in (range(num_iterations)):
            # Collect a few steps using collect_policy and save to the replay buffer.
            collect_data(train_env, agent.collect_policy, replay_buffer, steps=1)
            check_q_values(agent, train_env.current_time_step(), 0)

            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(iterator)
            train_loss = agent.train(experience).loss

            step = agent.train_step_counter.numpy()
            if step % 200 == 0:
                #print('step = {0}: loss = {1}'.format(step, train_loss))
                tqdm.write('step = {0}: loss = {1}'.format(step, train_loss))

            if step % 25000 == 0:
                # Evaluate the agent's policy once in a while
                avg_return = compute_avg_return(eval_env, agent.policy, num_episodes=5)
                print('step = {0}: Average Return = {1}'.format(step, avg_return))
    except KeyboardInterrupt:
        pass

    saver = policy_saver.PolicySaver(agent.policy)
    saver.save('./policies')
    env.close()
    eval_py_env.close()
    print("done")



if __name__ == '__main__':
    multiprocessing.handle_main(functools.partial(main))
    