# from environmentv3 import Environment
from environmentv3_transmission import Environment
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
from utils import plot_loss, configure_tensorflow_logging
from DQN import configure_agent
# from tf_agents.trajectories import Trajectory


BATCH_SIZE = 32
DISCOUNT = 0.75
TRAIN_TEST_RATIO = 1
NUM_ACTIONS = 5

learning_rate             = 6.25e-5

input_fc_layer_params     = (209, 148)
lstm_size                 = (77,)
output_fc_layer_params    = (216, 100)


target_update_tau               = 0.008304567528251162
actor_update_period             = 1
target_update_period            = 5
epsilon_greedy                  = 0.0561704800919212
gamma                           = 0.73070440103918044
reward_scale_factor             = 0.9874855517385459 

activation_fn                   = tf.keras.activations.selu

train_sequence_length = 4
num_episodes = 13

POLICY_LOAD_PATH = "DQN_2_30"
buffer_size = 2000

def create_environment():
    return Environment(discret=True, episode_time=999999, connected=True, env_step_time=1,
                       scaler_path=None, c_coef=1, e_coef=0, log_steps=False, seed=64547778)

def main(argv=None):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Is GPU build:", tf.test.is_built_with_cuda())
    if argv is None:
        argv = []

    train_env = create_environment()
    train_py_env = tf_py_environment.TFPyEnvironment(train_env)
    agent = configure_agent(train_py_env)

    agent.initialize()
    tf_policy = policy_loader.load(f'./policies/{POLICY_LOAD_PATH}')    
    agent.policy.update(tf_policy)

    # (Optional) Reset the agent's policy state
    agent.train = common.function(agent.train)


    table_name = 'uniform_table'
    replay_buffer_signature = tensor_spec.from_spec(
        agent.collect_data_spec)
    replay_buffer_signature = tensor_spec.add_outer_dim(
        replay_buffer_signature)

    table = reverb.Table(
        table_name,
        max_size=buffer_size,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1),
        signature=replay_buffer_signature)

    reverb_server = reverb.Server([table])

    replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
                                agent.collect_data_spec,
                                table_name=table_name,
                                sequence_length=train_sequence_length,
                                local_server=reverb_server)

    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
                                replay_buffer.py_client,
                                table_name,
                                sequence_length=train_sequence_length)
    
    # random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
    #                                             train_env.action_spec())
    

    dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=BATCH_SIZE,
            num_steps=train_sequence_length).prefetch(3)
    
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
    configure_tensorflow_logging()
    # Run the training loop
    steps_per_episode = 4
    losses = []

    for episode in tqdm(range(num_episodes)):
        try:
            sum_diff = 0
            sum_reward = 0
            for i in range(steps_per_episode):
                # Collect a few steps and save to the replay buffer.
                time_step, policy_state = collect_driver.run(time_step, policy_state)

                # Sample a batch of data from the buffer and update the agent's network.
                experience, _ = next(iterator)
                train_loss = agent.train(experience).loss

                step = agent.train_step_counter.numpy()
                losses.append(train_loss)

                sum_diff += np.sum(np.abs(experience.observation[:, 1]))/np.sum(experience.observation[:, 1].shape)
                sum_reward += np.sum(np.abs(experience.reward))/np.sum(experience.reward.shape)
                tqdm.write('step = {0}: loss = {1}, env time = {2}'.format(step, train_loss, train_env.time))

            tqdm.write('episode = {0}: sum difference = {1}'.format(episode, sum_diff))
            saver = policy_saver.PolicySaver(agent.policy)
            os.makedirs(f'./policies/DQN_real_online{episode}', exist_ok=True)
            saver.save(f'./policies/DQN_real_online{episode}')

            # if train_env.time > 120*60*2:
            #     break

        except KeyboardInterrupt:
            break
            print("next")


    plot_loss(losses, num_episodes)
    saver = policy_saver.PolicySaver(agent.policy)
    saver.save('./policies/DQN')
    print("done")


if __name__ == '__main__':
    multiprocessing.handle_main(functools.partial(main))