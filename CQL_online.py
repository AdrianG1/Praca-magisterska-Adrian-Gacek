import tensorflow as tf
from environmentv3 import Environment
import os
from utils import plot_loss, plot_trajs, configure_tensorflow_logging 
from tf_agents.policies import py_tf_eager_policy

from tf_agents.drivers import py_driver
import reverb
from tf_agents.specs import tensor_spec
from tf_agents.agents.ddpg import critic_rnn_network
from tf_agents.agents.cql import cql_sac_agent
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.utils import common
import tensorflow as tf
import numpy as np
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.replay_buffers import tf_uniform_replay_buffer
import multiprocessing
import functools
from tf_agents.system import multiprocessing
import warnings
warnings.filterwarnings('ignore')
from tf_agents.environments import ParallelPyEnvironment
from tf_agents.utils import common
from tf_agents.policies import policy_saver, policy_loader
from tqdm import tqdm
from tf_agents.trajectories import Trajectory
from tf_agents.train.utils import strategy_utils
from tf_agents.agents.sac import tanh_normal_projection_network
from CQL import configure_agent

# data params
POLICY_LOAD_ID      = 9

BATCH_SIZE          = 254
TRAIN_TEST_RATIO    = 0.75

# agent params
num_episodes                = 50
train_sequence_length       = 10
buffer_size                 = 3000

actor_learning_rate         = 1.53e-06
critic_learning_rate        = 25.16 * actor_learning_rate      
alpha_learning_rate         = 2.47e-06
cql_alpha_learning_rate     = 3.33e-07

cql_alpha                   = 0.280423024569609
include_critic_entropy_term = True
num_cql_samples             = 13
use_lagrange_cql_alpha      = True

target_update_tau           = 0.003415103446262748 
target_update_period        = 1 
gamma                       = 0.855869890833244 
reward_scale_factor         = 1

actor_input_fc_layer_params         = (134,) 
actor_lstm_size                     = (103,)
actor_output_fc_layer_params        = (100,)
actor_activation_fn                 = tf.keras.activations.selu

critic_joint_fc_layer_params=None
critic_lstm_size                    = (40,)
critic_output_fc_layer_params       = (100, 100)
critic_activation_fn                = tf.keras.activations.selu


def create_environment():
    """
    Configures CQL agent based on environment and listed configuration parameters.

    :param env: environment with specification
    :return: configured CQL agent 
    """
    return Environment(discret=False, episode_time=999999, connected=True, env_step_time=1,
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
    # tf_policy = policy_loader.load(f'./policies/CQL--{POLICY_LOAD_ID}')    
    # agent.policy.update(tf_policy)
    collected_data_checkpoint = tf.train.Checkpoint(agent)
    collected_data_checkpoint.restore("./policies/CCQL9-1")
    
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
    # configure_tensorflow_logging()
    # Run the training loop
    steps_per_episode = 1
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
            os.makedirs(f'./policies/CQL_real_online{episode}', exist_ok=True)
            saver.save(f'./policies/CQL_real_online{episode}')
            if train_env.time > 120*60:
                break 
        except KeyboardInterrupt:
            break
            print("next")


    # plot_loss(losses, num_episodes)
    saver = policy_saver.PolicySaver(agent.policy)
    saver.save('./policies/CQL')

    collected_data_checkpoint = tf.train.Checkpoint(agent)
    collected_data_checkpoint.save(f"./policies/CCQL_online/CCQL_online")
    print("done")


if __name__ == '__main__':
    multiprocessing.handle_main(functools.partial(main))
    