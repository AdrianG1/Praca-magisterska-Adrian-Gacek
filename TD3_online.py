import tensorflow as tf
from environmentv3 import Environment
import os
from utils import plot_loss, configure_tensorflow_logging

from tf_agents.agents.ddpg.actor_rnn_network import ActorRnnNetwork
from tf_agents.agents.ddpg import critic_rnn_network

from tf_agents.agents.td3 import td3_agent
from tf_agents.utils import common
import tensorflow as tf
import numpy as np
from tf_agents.policies import py_tf_eager_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.drivers import py_driver
import reverb
from tf_agents.specs import tensor_spec
from tf_agents.environments import tf_py_environment
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

POLICY_LOAD_ID = 12

BATCH_SIZE = 4
TRAIN_TEST_RATIO = 0.75

actor_learning_rate             =1.863793015696307e-04
critic_learning_rate            = actor_learning_rate *156.48294088634958

actor_input_fc_layer_params     = (209, 148)
actor_lstm_size                 = (77,)
actor_output_fc_layer_params    = (216, 100)

critic_joint_fc_layer_params    = None
critic_lstm_size                = (124,)
critic_output_fc_layer_params   = (96, 100)

exploration_noise_std           = 0.1282693
target_update_tau               = 0.0044578903484011854
actor_update_period             = 5
gamma                           = 0.91626
reward_scale_factor             = 1.3973179008462675

activation_fn                   = tf.keras.activations.elu
e_coef                          = 62
buffer_size                     = 28

train_sequence_length = 4
num_episodes = 50

def configure_agent(env):

    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=True)

    # Actor network
    with strategy.scope():
        actor_net = ActorRnnNetwork(
            env.observation_spec(),
            env.action_spec(),

            conv_layer_params=None,
            input_fc_layer_params=actor_input_fc_layer_params,
            lstm_size=actor_lstm_size,
            output_fc_layer_params=actor_output_fc_layer_params,
            activation_fn=activation_fn
        )

        critic_net = critic_rnn_network.CriticRnnNetwork(
            (env.observation_spec(), env.action_spec()),
            observation_conv_layer_params=None,
            observation_fc_layer_params=None,
            action_fc_layer_params=None,
            joint_fc_layer_params=critic_joint_fc_layer_params,
            lstm_size=critic_lstm_size,
            output_fc_layer_params=critic_output_fc_layer_params,
            activation_fn=activation_fn
        )

        actor_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=actor_learning_rate)
        critic_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=critic_learning_rate)


        agent = td3_agent.Td3Agent(
            env.time_step_spec(),
            env.action_spec(),
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            exploration_noise_std=exploration_noise_std,
            target_update_tau=target_update_tau,
            target_update_period=1,
            actor_update_period=actor_update_period,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            td_errors_loss_fn=tf.math.squared_difference
        )
        

        agent.initialize()
    return agent


def create_environment():
    return Environment(discret=False, episode_time=999999, seed=3214152, scaler_path=None, c_coef=1, e_coef=e_coef)


def main(argv=None):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Is GPU build:", tf.test.is_built_with_cuda())
    if argv is None:
        argv = []

    train_env = create_environment()
    train_py_env = tf_py_environment.TFPyEnvironment(train_env)
    agent = configure_agent(train_py_env)

    agent.initialize()
    tf_policy = policy_loader.load(f'./policies/t3d3{POLICY_LOAD_ID}')    
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
    steps_per_episode = 10
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
                tqdm.write('step = {0}: loss = {1}, reward = {2}'.format(step, train_loss, sum_reward))

            tqdm.write('episode = {0}: sum difference = {1}'.format(episode, sum_diff))
            saver = policy_saver.PolicySaver(agent.policy)
            os.makedirs(f'./policies/td3_online{episode}', exist_ok=True)
            saver.save(f'./policies/td3_online{episode}')
        except KeyboardInterrupt:
            break
            print("next")


    plot_loss(losses, num_episodes)
    saver = policy_saver.PolicySaver(agent.policy)
    saver.save('./policies/td3')
    print("done")



if __name__ == '__main__':
    multiprocessing.handle_main(functools.partial(main))