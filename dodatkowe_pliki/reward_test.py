import tensorflow as tf
from environmentv3 import Environment
import os
from utils import evaluate_policy, plot_loss, plot_trajs, configure_tensorflow_logging
from pandas import read_csv
from tf_agents.agents.ddpg import critic_rnn_network
from tf_agents.agents.sac import sac_agent
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.utils import common
import tensorflow as tf
import numpy as np
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
import multiprocessing
import functools
from tf_agents.system import multiprocessing
import warnings
warnings.filterwarnings('ignore', )
from tf_agents.environments import ParallelPyEnvironment
from tf_agents.utils import common
from tf_agents.policies import policy_saver, policy_loader
from tqdm import tqdm
from tf_agents.trajectories import Trajectory
from tf_agents.train.utils import strategy_utils
from tf_agents.agents.sac import tanh_normal_projection_network
import pickle
from collections import UserList
import random

class CustomReplayBuffer(UserList):
    """
    Alternatywny replay buffer, który ma za zadanie bardziej równomiernie wykorzystywać dane uczące
    wykorzystując w losowej kolejności wszystkie dane tyle samo razy.
    """
    def __init__(self, batch_size=32, num_steps=2):
        super().__init__([])
        self._batch_size=batch_size 
        self._num_steps=num_steps

    def get_iterator(self):
        return self.dataset_gen()  

    def dataset_gen(self):
        indexes = np.arange(len(self.data)-self._num_steps+1)
        idx = 0
        rng = random.Random(3313131)

        while True:
            # po przejściu wszystkich przestaw kolejność pobierania z listy
            rng.shuffle(indexes)

            while idx < indexes.shape[0]: # po kolei po przetasowanych indeksach

                # jeśli koniec batcha nie przekracza zakresu indeksów
                if idx + self._batch_size < indexes.shape[0]:
                    batch = []
                    for i in range(idx, idx+self._batch_size):
                        shuffled_idx = indexes[i]
                        # slice wycina _num_steps kolejnych doświadczeń z listy
                        batch.append(self.data[shuffled_idx:shuffled_idx+self._num_steps]) 

                    idx += self._batch_size

                # jeśli koniec batcha  przekracza zakresu indeksów
                else:
                    batch = []
                    # przejście do końca listy
                    for i in range(idx, indexes.shape[0]):
                        shuffled_idx = indexes[i]
                        batch.append(self.data[shuffled_idx:shuffled_idx+self._num_steps] ) 

                    # dopełnienie początkiem listy
                    for i in range(idx+self._batch_size - indexes.shape[0]):
                        shuffled_idx = indexes[i]
                        batch.append(self.data[shuffled_idx:shuffled_idx+self._num_steps] ) 

                    idx = idx+self._batch_size - indexes.shape[0] 
                    break
                
                step_type       = np.ndarray((len(batch), len(batch[0])))
                observation     = np.ndarray((len(batch), len(batch[0]), batch[0][0][1].numpy().shape[1]))
                actions         = np.ndarray((len(batch), len(batch[0])))
                info            = np.ndarray((len(batch), len(batch[0])))
                next_step_type  = np.ndarray((len(batch), len(batch[0])))
                reward          = np.ndarray((len(batch), len(batch[0])))
                discount        = np.ndarray((len(batch), len(batch[0])))

                for b in range(len(batch)):
                    for s in range(len(batch[0])):
                        step_type     [b, s] = batch[b][s].step_type       
                        observation   [b, s] = batch[b][s].observation.numpy()[0]     
                        actions       [b, s] = batch[b][s].action         
                        #info          [b, s] = batch[b][s].policy_info            
                        next_step_type[b, s] = batch[b][s].next_step_type  
                        reward        [b, s] = batch[b][s].reward          
                        discount      [b, s] = batch[b][s].discount        

                traj = Trajectory( tf.constant(step_type, dtype=tf.int32),
                                   tf.constant(observation, dtype=tf.float32),
                                   tf.constant(actions, dtype=tf.float32),
                                   (),
                                   tf.constant(next_step_type, dtype=tf.int32),
                                   tf.constant(reward, dtype=tf.float32),
                                   tf.constant(discount, dtype=tf.float32))

                yield traj #Trajectory()


BATCH_SIZE = 256
TRAIN_TEST_RATIO = 1

actor_learning_rate             = 9.401499111738006e-04 / 5
critic_learning_rate            = actor_learning_rate * 9.523080854631749 * 5
alpha_learning_rate             = 8.850215277629775e-05 / 5


actor_input_fc_layer_params     = (195,)
actor_lstm_size                 = (101,)
actor_output_fc_layer_params    = (100,)

critic_joint_fc_layer_params    = None
critic_lstm_size                = (35,)
critic_output_fc_layer_params   = (105, 100)

target_update_tau               = 0.031101832198767103
target_update_period            = 1
actor_update_period             = 3
gamma                           = 0.9674273939790276
reward_scale_factor             = 0.23014718662243797

activation_fn                   = tf.keras.activations.relu

train_sequence_length           = 12
num_episodes                    = 25

with open("scaler.pkl", 'rb') as file:
    scaler = pickle.load(file)
    normalize_reward = lambda x: float(scaler.transform([[x]]))

def calculate_reward(observation, action, last_action):
    """ Wyznaczanie nagrody """

    # komfort temperaturowy
    abs_diff = np.abs(observation[1])
    sqrt_diff = np.sqrt(abs_diff)
    comfort =  -sqrt_diff

    energy = -action/(observation[0]/100)/3.5
    return normalize_reward(comfort * 1 + 0.5 * energy) #


def get_trajectory_from_csv(path, state_dim, replay_buffer, test_buffer, train_test_ratio, discount=0.75):
    """ Wczytywanie danych z csv """
    df = read_csv(path, index_col=0)
    trajs = []
    last_action=0
    train_end = len(df) * train_test_ratio

    for idx in range(len(df)):
        if idx == len(df)-1:
            break
        state = df.iloc[idx, :state_dim].values  # Convert to numpy array
        action = tf.constant(df.iloc[idx, 3], dtype=tf.float32)

        reward = calculate_reward(df.iloc[idx+1, :state_dim].values, df.iloc[idx+1, 3], last_action)

        last_action = action
        continous_action = tf.expand_dims(tf.clip_by_value(action, 0, 100), axis=-1)

        traj = Trajectory(tf.constant(1, dtype=tf.int32, shape=(1,)), 
                        tf.expand_dims(tf.constant(state, dtype=tf.float32), axis=0),
                        continous_action, 
                        (), 
                        tf.constant(1, dtype=tf.int32, shape=(1,)),
                        tf.constant(reward, dtype=tf.float32, shape=(1,)), 
                        tf.constant(discount, dtype=tf.float32, shape=(1,)))
        trajs.append(traj)

        if  idx < train_end: 
            replay_buffer.append(traj)
        else:
            test_buffer.append(traj)

    return trajs

def configure_agent(env):

    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=True)
    
    # Actor network
    with strategy.scope():
        actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
                                env.observation_spec(),
                                env.action_spec(),
                                input_fc_layer_params=actor_input_fc_layer_params,
                                input_dropout_layer_params=None,
                                lstm_size=actor_lstm_size,
                                output_fc_layer_params=actor_output_fc_layer_params,
                                activation_fn=activation_fn)


    # Critic network
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


        agent = sac_agent.SacAgent(
                env.time_step_spec(),
                env.action_spec(),
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
                td_errors_loss_fn=tf.math.squared_difference,
                actor_loss_weight=3,
                gamma=gamma,
                reward_scale_factor=reward_scale_factor,
                initial_log_alpha=0.5,

                train_step_counter=tf.Variable(0))

        agent.initialize()

    return agent



def create_environment():
    return Environment(discret=False)


def main(argv=None):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Is GPU build:", tf.test.is_built_with_cuda())
    if argv is None:
        argv = []

    configure_tensorflow_logging()

    train_env = tf_py_environment.TFPyEnvironment(create_environment)

    agent = configure_agent(train_env)

    agent.train = common.function(agent.train)

    replay_buffer = CustomReplayBuffer(BATCH_SIZE, train_sequence_length)
    del train_env
    
    print("================================== collecting data ===============================================")
    test_buffer = []

    trajs = get_trajectory_from_csv("./csv_data/trajectory7.csv", 2, replay_buffer, test_buffer, TRAIN_TEST_RATIO)
    plot_trajs(trajs)

    # collected_data_checkpoint = tf.train.Checkpoint(replay_buffer)
    # # collected_data_checkpoint.save("./replay_buffers/replay_buffer")
    # collected_data_checkpoint.restore("./replay_buffers/replay_buffer-1")

    # Dataset generates trajectories with shape [Bx2x...]

    iterator = replay_buffer.get_iterator()

    print("================================== training ======================================================")
    # Run the training loop
    steps_per_episode = int(len(replay_buffer)) // BATCH_SIZE
    losses = np.full(num_episodes*steps_per_episode+1, -1)
    # agent.policy.update(policy_loader.load("./policies/SAC40"))

    for episode in tqdm(range(num_episodes)):
        try:
            for _ in range(steps_per_episode):
                # Sample a batch of data from the buffer and update the agent's network.
                experience = next(iterator)
                train_loss = agent.train(experience).loss
                step = agent.train_step_counter.numpy()
                losses[step] = train_loss
            tqdm.write('step = {0}: loss = {1}'.format(step, train_loss))
            # if episode % 5 == 4:
            #     tqdm.write('evaluated difference = {0}:\n'.format(evaluate_policy(agent, test_buffer)))
            
            saver = policy_saver.PolicySaver(agent.policy)
            os.makedirs(f'./policies/SAC-r{episode}', exist_ok=True)
            saver.save(f'./policies/SAC-r{episode}')
        except KeyboardInterrupt:
            break
            print("next")

    plot_loss(losses, num_episodes)
    saver = policy_saver.PolicySaver(agent.policy)
    saver.save('./policies/SAC')
    print("done")



if __name__ == '__main__':
    multiprocessing.handle_main(functools.partial(main))
    