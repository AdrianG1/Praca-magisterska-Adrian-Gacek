import tensorflow as tf
from environmentv3 import Environment
import os
from utils import plot_trajs
import optuna
import SAC
from pandas import read_csv

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.networks import actor_distribution_network
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
from tf_agents.policies import policy_saver
from tf_agents.trajectories import time_step as ts
from tf_agents.train.utils import strategy_utils
from tf_agents.agents.sac import tanh_normal_projection_network
from tf_agents.trajectories import Trajectory

BATCH_SIZE = 1
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
        difference += abs(float((experience.action - action_step.action).numpy()[0][0]))
    
    return difference 



def training_agent(agent, num_episodes):
    steps_per_episode = 300
    global train_iterator
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
    global env

    num_of_layers = trial.suggest_int('num_of_layers', 2, 8)
    actor_fc_layer_params = tuple([trial.suggest_int(f'actor_layer{i}', 64, 256) for i in range(num_of_layers)])
    critic_joint_fc_layer_params = tuple([trial.suggest_int(f'critic_layer{i}', 64, 256) for i in range(num_of_layers)])
    actor_learning_rate = trial.suggest_loguniform('actor_learning_rate', 1e-5, 1e-3)
    critic_learning_rate = trial.suggest_loguniform('critic_learning_rate', 1e-5, 1e-3)
    alpha_learning_rate = trial.suggest_loguniform('alpha_learning_rate', 1e-5, 1e-3)
    target_update_tau = trial.suggest_uniform('target_update_tau', 0.001, 0.05)
    target_update_period = trial.suggest_int('target_update_period', 1, 10)
    gamma = trial.suggest_uniform('gamma', 0.9, 0.999)
    reward_scale_factor = trial.suggest_uniform('reward_scale_factor', 0.1, 1.0)
    td_errors_loss_fn = tf.math.squared_difference
    
    agent = configure_agent(env, 
                            critic_learning_rate,actor_learning_rate, alpha_learning_rate,
                            target_update_tau, target_update_period,
                            gamma, reward_scale_factor,
                            actor_fc_layer_params, critic_joint_fc_layer_params,
                            td_errors_loss_fn)
 

    max_rating = training_agent(agent, 30)
    return max_rating

def configure_agent(env_spec, 
                    critic_learning_rate,actor_learning_rate, alpha_learning_rate,
                    target_update_tau, target_update_period,
                    gamma, reward_scale_factor,
                    actor_fc_layer_params, critic_joint_fc_layer_params,
                    td_errors_loss_fn
                    ):

    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=True)

    # Actor network
    with strategy.scope():
        actor_net = actor_distribution_network.ActorDistributionNetwork(
                                env_spec.observation_spec(),
                                env_spec.action_spec(),
                                fc_layer_params=actor_fc_layer_params,
                                continuous_projection_net=(
                                                tanh_normal_projection_network.TanhNormalProjectionNetwork))


    # Critic network
        critic_net = critic_network.CriticNetwork(
                (env_spec.observation_spec(), env_spec.action_spec()),
                observation_fc_layer_params=None,
                action_fc_layer_params=None,
                joint_fc_layer_params=critic_joint_fc_layer_params,
                kernel_initializer='glorot_uniform',
                last_kernel_initializer='glorot_uniform')


        agent = sac_agent.SacAgent(
                env_spec.time_step_spec(),
                env_spec.action_spec(),
                actor_network=actor_net,
                critic_network=critic_net,
                actor_optimizer=tf.keras.optimizers.Adam(
                    learning_rate=actor_learning_rate),
                critic_optimizer=tf.keras.optimizers.Adam(
                    learning_rate=critic_learning_rate),
                alpha_optimizer=tf.keras.optimizers.Adam(
                    learning_rate=alpha_learning_rate),
                target_update_tau=target_update_tau,
                target_update_period=target_update_period,
                td_errors_loss_fn=td_errors_loss_fn,
                gamma=gamma,
                reward_scale_factor=reward_scale_factor,
                train_step_counter=tf.Variable(0))

    
        agent.initialize()

    return agent


def create_environment():
    return Environment(discret=False)


def get_trajectory_from_csv(path, state_dim, train_buffer, test_buffer, TRAIN_TEST_RATIO):
    df = read_csv(path, index_col=0)
    trajs = []
    train_end = len(df) * TRAIN_TEST_RATIO
    for idx, record in df.iterrows():

        state = record.iloc[:state_dim].values  # Convert to numpy array
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

    train_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                        data_spec=agent.collect_data_spec,
                        batch_size=env.batch_size,
                        max_length=20000)
    global test_buffer
    test_buffer = []
    del agent

    print("================================== collecting data ===============================================")
    get_trajectory_from_csv("./csv_data/trajectory.csv", 2, train_buffer, test_buffer, TRAIN_TEST_RATIO)

    # collected_data_checkpoint = tf.train.Checkpoint(replay_buffer)
    # collected_data_checkpoint.save("./replay_buffers/replay_buffer")
    # collected_data_checkpoint.restore("./replay_buffers/replay_buffer-1")

    # Dataset generates trajectories with shape [Bx2x...]
    train_dataset = train_buffer.as_dataset(
        num_parallel_calls=3, 
        sample_batch_size=BATCH_SIZE, 
        num_steps=num_steps_dataset).prefetch(3)

    global train_iterator
    train_iterator = iter(train_dataset)

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
    