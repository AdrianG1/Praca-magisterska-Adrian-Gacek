from environmentv3 import Environment
import tensorflow as tf
from tf_agents.environments import tf_py_environment
import matplotlib.pyplot as plt
import numpy as np
import os
import psutil
from tf_agents.trajectories import time_step as ts
from tqdm import tqdm

def log_memory_usage():
    process = psutil.Process(os.getpid())
    memory_use = process.memory_info().rss / 1024 ** 2  # memory use in MB
    print(f'\n\n\n\n Memory usage: {memory_use:.2f} MB')

def create_environment():
    return Environment(discret=False)


def main():
    log_memory_usage()

    env = tf_py_environment.TFPyEnvironment(create_environment())
    log_memory_usage()

    # Load the saved policy
    loaded_policy = tf.saved_model.load('./policies/td3')

    rewards = []
    T = []
    T_sp = []
    controls = []
    times = []
    
    log_memory_usage()

    policy_state = [tf.constant([[0 for _ in range(40)]], dtype=tf.float32), tf.constant([[0 for _ in range(40)]], dtype=tf.float32)]
    for i in tqdm(range(14, 15, 1)):
        try:
            loaded_policy = tf.saved_model.load(f'./policies/td3{i}')
            policy_state = [tf.constant([[0 for _ in range(40)]], dtype=tf.float32), tf.constant([[0 for _ in range(40)]], dtype=tf.float32)]

            time_step = env.reset()
            time_step = ts.TimeStep(
                    step_type=tf.reshape(time_step.step_type, (1, )),
                    reward=tf.reshape(time_step.reward, (1, )),
                    discount=time_step.discount,
                    observation= tf.reshape(time_step.observation, (1, 2))
                    )
            
            while not time_step.is_last():
                action_step = loaded_policy.action(time_step, policy_state)
                policy_state = action_step.state

                time_step = env.step(action_step.action)

                time_step = ts.TimeStep(
                        step_type=tf.reshape(time_step.step_type, (1, )),
                        reward=tf.reshape(time_step.reward, (1, )),
                        discount=time_step.discount,
                        observation= tf.reshape(time_step.observation, (1, 2))
                        )

                state, reward, done, info = time_step.observation, time_step.reward, time_step.step_type, 0
                rewards.append(reward)
                T.append(state[:, 0])
                T_sp.append(state[:, 1]+state[:, 0])
                controls.append(action_step.action)
                times.append(env.envs[0].time)
        except KeyboardInterrupt:
            tqdm.write(f"next {i+1}")

    T = np.array(T).reshape(-1)
    T_sp = np.array(T_sp).reshape(-1)
    rewards = np.array(rewards)
    times = np.arange(len(rewards))#np.array(times)

    correct = np.where(np.abs(T_sp - T) <= 1)[0]
    incorrect = np.where(np.abs(T_sp - T) > 1)[0]
    T_correct = T[correct]
    T_incorrect = T[incorrect]

    plt.figure()
    plt.plot(times, T)
    plt.scatter(times[correct], T_correct, c='g')
    plt.scatter(times[incorrect], T_incorrect, c='r')
    plt.plot(times, T_sp)
    plt.plot(times, T_sp+1, 'k--')
    plt.plot(times, T_sp-1, 'k--')
    plt.title('temperature')
    plt.savefig('./plot/temperature.png')

    plt.figure()
    plt.plot(times, tf.squeeze(rewards))
    plt.title('rewards')
    plt.savefig('./plot/rewards.png')

    plt.figure()
    plt.plot(times, tf.squeeze(controls))
    plt.title('controls')
    plt.savefig('plot/controls.png')

if __name__ == '__main__':
    main()
