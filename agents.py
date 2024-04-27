from environmentv2 import Environment
import multiprocessing
import functools
from tf_agents.system import multiprocessing
# import warnings
# warnings.filterwarnings('ignore')
from tf_agents.environments import ParallelPyEnvironment
from tf_agents.trajectories import trajectory
import tensorflow as tf
from tf_agents.networks import q_network
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common

# Define the training loop
def train(agent, env, num_iterations=20000):
    for _ in range(num_iterations):
        time_step = env.reset()
        while not time_step.is_last().all():
            action_step = agent.collect_policy.action(time_step)
            next_time_step = env.step(action_step.action)
            traj = trajectory.from_transition(time_step, action_step, next_time_step)
            agent.train(traj)
            time_step = next_time_step


def evaluate(agent, env, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = env.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = agent.policy.action(time_step)
            time_step = env.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return
    average_return = total_return / num_episodes
    return average_return


BATCH_SIZE = 4


def create_environment():
    return Environment()


def main(argv=None):  # Accept argv even if you don't use it
    if argv is None:
        argv = []
    env = ParallelPyEnvironment([create_environment] * BATCH_SIZE)


    # Define the Q-network
    fc_layer_params = (100,)
    q_net = q_network.QNetwork(
        env.observation_spec(),
        env.action_spec(),
        fc_layer_params=fc_layer_params)

    # Define the DQN agent
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3)
    train_step_counter = tf.Variable(0)
    agent = dqn_agent.DqnAgent(
        env.time_step_spec(),
        env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)
    agent.initialize()


    # Run training
    train(agent, env)

    # Evaluate the agent

    average_return = evaluate(agent, env)
    print('Average return:', average_return.numpy())
    
if __name__ == '__main__':
    multiprocessing.handle_main(functools.partial(main))

