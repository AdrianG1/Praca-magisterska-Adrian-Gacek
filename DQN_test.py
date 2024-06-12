from environmentv3 import Environment
import tensorflow as tf
from tf_agents.environments import tf_py_environment
import matplotlib.pyplot as plt
import numpy as np
import os
import psutil
import random


def log_memory_usage():
    process = psutil.Process(os.getpid())
    memory_use = process.memory_info().rss / 1024 ** 2  # memory use in MB
    print(f'\n\n\n\n Memory usage: {memory_use:.2f} MB')

def create_environment():
    return Environment(discret=True)


def main():
    env = tf_py_environment.TFPyEnvironment(create_environment())

    # Load the saved policy
    loaded_policy = tf.compat.v2.saved_model.load('./policies/DQN')

    rewards = []
    T = []
    T_sp = []
    controls = []
    times = []
    
    log_memory_usage()
 
    for _ in range(1):
        time_step = env.reset()
        while not time_step.is_last():
            action_step = loaded_policy.action(time_step)
            time_step = env.step(action_step.action)
            state, reward, done, info = time_step.observation, time_step.reward, time_step.step_type, 0
            rewards.append(reward)
            T.append(state[:, 0])
            T_sp.append(state[:, 1])
            controls.append(action_step.action)
            times.append(env.envs[0].time)

    T = np.array(T).reshape(-1)
    T_sp = np.array(T_sp).reshape(-1)
    rewards = np.array(rewards)
    times = np.array(times)

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
    plt.plot(times, rewards)
    plt.title('rewards')
    plt.savefig('./plot/rewards.png')

    plt.figure()
    plt.plot(times, controls)
    plt.title('controls')
    plt.savefig('plot/controls.png')

if __name__ == '__main__':
    main()
    #test()