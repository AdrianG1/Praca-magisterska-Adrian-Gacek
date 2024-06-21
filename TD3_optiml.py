import tensorflow as tf
from environmentv3 import Environment
import os
from utils import plot_trajs
import optuna
import SAC
from pandas import read_csv

from tf_agents.agents.ddpg.actor_rnn_network import ActorRnnNetwork
from tf_agents.agents.ddpg import critic_rnn_network

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
from tf_agents.trajectories import time_step as ts
from tf_agents.train.utils import strategy_utils
from tf_agents.trajectories import Trajectory

TRAINING_STEPS = 30
BATCH_SIZE = 32
TRAIN_TEST_RATIO = 0.75
num_steps_dataset = 2

# critic_learning_rate = 3e-4
# actor_learning_rate = 3e-5 
# alpha_learning_rate = 3e-5 

# target_update_tau = 0.005 
# target_update_period = 10 
# gamma = 0.99 
# reward_scale_factor = 1.0 

# actor_fc_layer_params = (75, 75, 75, 75)
# critic_joint_fc_layer_params =(75, 75, 75, 75)
# #struktura sieci
# num_episodes = 25
# 
# td_errors_loss_fn = tf.math.squared_difference


def evaluate_policy(agent, num_test_steps=1000):
    global test_buffer
    difference = 0

    for i in range(min(len(test_buffer), num_test_steps)):
        experience = test_buffer[i]

        time_step = ts.TimeStep(
                step_type=experience.step_type,
                reward=experience.reward,
                discount=experience.discount,
                observation= tf.reshape(experience.observation, (1, 2))
                )
        action_step = agent.policy.action(time_step)
        #print(experience, time_step, "\n", action_step.action)
        difference += abs(float((experience.action - action_step.action).numpy()[0]))
    
    return difference 



def training_agent(agent, train_iterator, num_episodes):
    steps_per_episode = 300 // BATCH_SIZE
    min_diff = np.inf
    min_diff_ep = -1

    for episode in range(num_episodes):
        for _ in range(steps_per_episode):
            experience, _ = next(train_iterator)
            train_loss = agent.train(experience).loss

        rating = evaluate_policy(agent)
        if rating < min_diff:
            min_diff = rating
            min_diff_ep = episode
     
    print("\n\n\n min diff (ep): ", min_diff,  min_diff_ep, "\n")
    return min_diff


def objective(trial):
    global env, train_buffer

    num_of_layers = trial.suggest_int('num_of_layers', 2, 8)
    actor_fc_layer_params = tuple([trial.suggest_int(f'actor_layer{i}', 64, 256) for i in range(num_of_layers)])
    critic_joint_fc_layer_params = tuple([trial.suggest_int(f'critic_layer{i}', 64, 256) for i in range(num_of_layers)])
    
    net_structure = suggest_network_structure(trial)
    actor_fc_input =    net_structure[0]
    actor_lstm =        net_structure[1]
    actor_fc_output =   net_structure[2]

    critic_lstm =       net_structure[3]
    critic_fc_output =  net_structure[4]

    actor_learning_rate = trial.suggest_loguniform('actor_learning_rate', 1e-5, 1e-2)
    critic_learning_rate = trial.suggest_loguniform('critic_learning_rate', 1e-5, 1e-2)

    target_update_tau = trial.suggest_uniform('target_update_tau', 0.001, 0.05)
    target_update_period = trial.suggest_int('target_update_period', 1, 10)
    gamma = trial.suggest_uniform('gamma', 0.9, 0.999)

    reward_scale_factor = trial.suggest_uniform('reward_scale_factor', 0.1, 1.0)
    td_errors_loss_fn = tf.math.squared_difference
    actor_update_period = trial.suggest_int('actor_update_period', 1, 5)
    exploration_noise_std = trial.suggest_uniform('exploration_noise_std', 0.1, 0.3)

    num_steps_dataset = trial.suggest_int('actor_update_period', 4, 60)
    



    agent = configure_agent(env, 
                        critic_learning_rate,actor_learning_rate,
                        target_update_tau, target_update_period, actor_update_period,
                        gamma, reward_scale_factor, exploration_noise_std,
                        actor_fc_layer_params, critic_joint_fc_layer_params,
                        actor_fc_input, actor_lstm, actor_fc_output,
                        critic_lstm, critic_fc_output
                        )
 

    train_dataset = train_buffer.as_dataset(
        num_parallel_calls=3, 
        sample_batch_size=BATCH_SIZE, 
        num_steps=num_steps_dataset).prefetch(3)

    train_iterator = iter(train_dataset)

    max_rating = training_agent(agent, train_iterator, TRAINING_STEPS)
    return max_rating

def configure_agent(env, 
                        critic_learning_rate,actor_learning_rate,
                        target_update_tau, target_update_period, actor_update_period,
                        gamma, reward_scale_factor, exploration_noise_std,
                        actor_fc_input, actor_lstm, actor_fc_output,
                        critic_lstm, critic_fc_output
                        ):


    actor_net = ActorRnnNetwork(
        env.observation_spec(),
        env.action_spec(),

        conv_layer_params=None,
        input_fc_layer_params=actor_fc_input, #(200, 100),
        lstm_size=actor_lstm,                    #(40,),
        output_fc_layer_params=actor_fc_output             # (200, 100)
    )

    critic_net = critic_rnn_network.CriticRnnNetwork(
        (env.observation_spec(), env.action_spec()),
        observation_conv_layer_params=None,
        observation_fc_layer_params=None,
        action_fc_layer_params=None,
        joint_fc_layer_params=None,
        lstm_size=critic_lstm,                          #(40,),
        output_fc_layer_params=critic_fc_output             #(200, 100)
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
        exploration_noise_std=exploration_noise_std,
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        actor_update_period=actor_update_period,
        #td_errors_loss_fn=common.element_wise_squared_loss,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        train_step_counter=train_step_counter
    )
    #agent.train_sequence_length = train_sequence_length

    agent.initialize()
    return agent
        
def suggest_network_structure(trial):
    # Suggest actor network parameters
    num_actor_input_layers = trial.suggest_int('num_actor_input_layers', 1, 4)
    actor_fc_input = tuple([trial.suggest_int(f'actor_input_layer{i}', 64, 256) for i in range(num_actor_input_layers)])
    
    num_actor_lstm = trial.suggest_int('num_actor_input_layers', 1, 3)
    actor_lstm = tuple([trial.suggest_int(f'actor_input_layer{i}', 16, 128) for i in range(num_actor_lstm)])
    
    num_actor_output_layers = trial.suggest_int('num_actor_output_layers', 1, 4)
    actor_fc_output = tuple([trial.suggest_int(f'actor_output_layer{i}', 64, 256) for i in range(num_actor_output_layers)])
    
    # Suggest critic network parameters
    num_critic_lstm = trial.suggest_int('num_critic_input_layers', 1, 3)
    critic_lstm = tuple([trial.suggest_int(f'critic_input_layer{i}', 16, 128) for i in range(num_critic_lstm)])
    
    num_critic_output_layers = trial.suggest_int('num_critic_output_layers', 1, 3)
    critic_fc_output = tuple([trial.suggest_int(f'critic_output_layer{i}', 64, 256) for i in range(num_critic_output_layers)])
    
    return (actor_fc_input, actor_lstm, actor_fc_output, critic_lstm, critic_fc_output)


def create_environment():
    return Environment(discret=False)


def get_trajectory_from_csv(path, state_dim, train_buffer, test_buffer, TRAIN_TEST_RATIO):
    df = read_csv(path, index_col=0)
    trajs = []
    train_end = len(df) * TRAIN_TEST_RATIO
    for idx, record in df.iterrows():

        state = record.iloc[:state_dim].values*100  # Convert to numpy array
        action = tf.constant(record["Akcje"], dtype=tf.float32)
        reward = record["Nagrody"]
        continous_action = tf.expand_dims(tf.clip_by_value(action, 0, 100), axis=-1)

        traj = Trajectory(tf.constant(1, dtype=tf.int32, shape=(1,)), 
                        tf.expand_dims(tf.constant(state, dtype=tf.float32), axis=0),
                        continous_action, 
                        (), 
                        tf.constant(1, dtype=tf.int32, shape=(1,)),
                        tf.constant(reward, dtype=tf.float32, shape=(1,)), 
                        tf.constant(0, dtype=tf.float32, shape=(1,)))
        trajs.append(traj)

        if  idx > train_end:
            test_buffer.append(traj)
        else:
            train_buffer.add_batch(traj)

    plot_trajs(trajs)



def main(argv=None):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Is GPU build:", tf.test.is_built_with_cuda())
    if argv is None:
        argv = []

    global env
    env = ParallelPyEnvironment([create_environment] * 1)
    env = tf_py_environment.TFPyEnvironment(env)

    agent = SAC.configure_agent(env)

    agent.train = common.function(agent.train)

    global test_buffer, train_buffer
    train_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                        data_spec=agent.collect_data_spec,
                        batch_size=env.batch_size,
                        max_length=10000)
    test_buffer = []
    del agent

    print("================================== collecting data ===============================================")
    get_trajectory_from_csv("./csv_data/trajectory.csv", 2, train_buffer, test_buffer, TRAIN_TEST_RATIO)

    # collected_data_checkpoint = tf.train.Checkpoint(replay_buffer)
    # collected_data_checkpoint.save("./replay_buffers/replay_buffer")
    # collected_data_checkpoint.restore("./replay_buffers/replay_buffer-1")


    print("================================== optimizing ======================================================")

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    print("Best trial:")
    best_trial = study.best_trial
    print("  Value: {}".format(best_trial.value))
    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))

        
    print("done")



if __name__ == '__main__':
    multiprocessing.handle_main(functools.partial(main))
    