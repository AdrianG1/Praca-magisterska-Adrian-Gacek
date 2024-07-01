from environmentv3 import Environment
import tensorflow as tf
import numpy as np
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_rnn_network
from tf_agents.drivers import py_driver
from tf_agents.policies import py_tf_eager_policy
import reverb
from tf_agents.specs import tensor_spec
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.replay_buffers import tf_uniform_replay_buffer
import multiprocessing
import functools
from tf_agents.policies import random_tf_policy
from tf_agents.system import multiprocessing
import warnings
warnings.filterwarnings('ignore')
from tf_agents.utils import common
from tf_agents.policies import policy_saver, policy_loader
from tqdm import tqdm
import os
from utils import plot_loss
# from tf_agents.trajectories import Trajectory
from time import time

BATCH_SIZE = 32
BATCH_SIZE_EVAL = 1
LEARNING_RATE = 2e-4
DISCOUNT = 0.75
TRAIN_TEST_RATIO = 0.75
NUM_STEPS_DATASET = 2
POLICY_LOAD_ID = 20

replay_buffer_max_length = 1000


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


def main(argv=None):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Is GPU build:", tf.test.is_built_with_cuda())
    if argv is None:
        argv = []

    train_env = create_environment()
    train_py_env = tf_py_environment.TFPyEnvironment(train_env)
    agent = configure_agent(train_env)

    agent.initialize()
    tf_policy = policy_loader.load(f'./policies/DQN{POLICY_LOAD_ID}')    

    # (Optional) Reset the agent's policy state
    agent.train = common.function(agent.train)
    

    table_name = 'uniform_table'
    replay_buffer_signature = tensor_spec.from_spec(
        agent.collect_data_spec)
    replay_buffer_signature = tensor_spec.add_outer_dim(
        replay_buffer_signature)

    table = reverb.Table(
        table_name,
        max_size=replay_buffer_max_length,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=replay_buffer_signature)

    reverb_server = reverb.Server([table])

    replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
                                agent.collect_data_spec,
                                table_name=table_name,
                                sequence_length=12,
                                local_server=reverb_server)

    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
                                replay_buffer.py_client,
                                table_name,
                                sequence_length=12)

    

    dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=BATCH_SIZE,
            num_steps=12).prefetch(3)
    
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
    # Run the training loop
    num_episodes = 25
    steps_per_episode = 10
    losses = []

    start_time = time()
    for episode in tqdm(range(num_episodes)):
        try:
            for _ in range(steps_per_episode):
                # Collect a few steps and save to the replay buffer.
                time_step, policy_state = collect_driver.run(time_step, policy_state)

                # Sample a batch of data from the buffer and update the agent's network.
                experience, _ = next(iterator)
                print("step")
                train_loss = agent.train(experience).loss

                step = agent.train_step_counter.numpy()
                losses.append(train_loss)

            tqdm.write('step = {0}: loss = {1}'.format(step, train_loss))

            saver = policy_saver.PolicySaver(agent.policy)
            os.makedirs(f'./policies/DQN_online{episode}', exist_ok=True)
            saver.save(f'./policies/DQN_online{episode}')
        except KeyboardInterrupt:
            break

    plot_loss(losses, num_episodes)
    saver = policy_saver.PolicySaver(agent.policy)
    saver.save('./policies/DQN')
    # env.close()
    # eval_py_env.close()
    print("done")



if __name__ == '__main__':
    main()
    # multiprocessing.handle_main(functools.partial(main))
    