from environmentv3 import Environment
import tensorflow as tf
from tf_agents.environments import tf_py_environment
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import psutil
from tf_agents.trajectories import time_step as ts
from tqdm import tqdm
from utils import abnormalize_state, discretize, undiscretize

def log_memory_usage():
    process = psutil.Process(os.getpid())
    memory_use = process.memory_info().rss / 1024 ** 2  # memory use in MB
    print(f'\n\n\n\n Memory usage: {memory_use:.2f} MB')

def create_environment():
    return Environment(discret=False, episode_time=120, seed=9171518)


def main():
    log_memory_usage()

    env = tf_py_environment.TFPyEnvironment(create_environment())
    log_memory_usage()
    env.reset()

    rewards = []
    T = []
    T_sp = []
    controls = []
    times = []
    
    
    log_memory_usage()
    for i in tqdm(range(3, 50, 910)):
        sum_diff = 0
        try:
            policy_path = (f'./policies/SAC2_online{i}')
            loaded_policy = tf.saved_model.load(policy_path)
            policy_state = loaded_policy.get_initial_state(batch_size=1)
            time_step = env.reset()
            time_step = ts.TimeStep(
                    step_type=tf.reshape(time_step.step_type, (1, )),
                    reward=tf.reshape(time_step.reward, (1, )),
                    discount=time_step.discount,
                    observation= tf.reshape(time_step.observation, (1, 2))
                    )
            while not time_step.is_last():
                dist = loaded_policy.distribution(time_step, policy_state)
                policy_state = dist.state
                time_step2 = env.step(float(dist.action.loc))
                # action_step = loaded_policy.action(time_step, policy_state)
                # policy_state = action_step.state
                # time_step2 = env.step(action_step.action)
                time_step = ts.TimeStep(
                        step_type=tf.reshape(time_step2.step_type, (1, )),
                        reward=tf.reshape(time_step2.reward, (1, )),
                        discount=time_step2.discount,
                        observation= tf.reshape(time_step2.observation, (1, 2))

                        # observation= tf.constant(np.array([to0, to1]), shape=(1, 2))
                        )

                state, reward, done, info = time_step2.observation, time_step.reward, time_step.step_type, 0

                sum_diff += np.abs(time_step.observation[:, 1].numpy())
                rewards.append(reward)
                real_state = abnormalize_state(state)
                T.append(real_state[:, 0])
                T_sp.append(real_state[:, 1])
                controls.append(float(dist.action.loc))
                # controls.append(float(action_step.action))
                times.append(env.envs[0].time)
                if sum_diff > 10000:
                    tqdm.write(f"time: {env.envs[0].time}")
                    break
                    

            tqdm.write(str([policy_path, float(sum_diff)]))
        except KeyboardInterrupt:
            tqdm.write(f"next {i+1}")


    mean_energy = np.sum([controls[i]  * (times[i+1]-times[i]) for i in range(len(times)-1)]) / times[-1] *100
    mean_reward = np.sum([rewards[i]   * (times[i+1]-times[i]) for i in range(len(times)-1)]) / times[-1] 

    error = np.abs(np.array(T_sp)-np.array(T))
    mean_error  = np.sum([error[i]     * (times[i+1]-times[i]) for i in range(len(times)-1)]) / times[-1] 

    comfort = [(times[i+1]-times[i]) if np.abs(T_sp[i] - T[i]) < 2 else 0 for i in range(len(times)-1)]
    comfort_time = np.sum(comfort)

    print(f"średnie zużycie energii:     {mean_energy:.02f} [%]")
    print(f"średnia nagroda:             {mean_reward:.02f} [1]")
    print(f"średni uchyb temperatury:    {mean_error:.02f}  [°C]")
    print(f"czas w komfortowym zakresie: {comfort_time:.02f}[s]")

    T = np.array(T).reshape(-1)
    T_sp = np.array(T_sp).reshape(-1)
    rewards = np.array(rewards).reshape(-1)
    times = np.array(times).reshape(-1)
    # times = range(len(T))

    correct = np.where(np.abs(T_sp - T) <= 1)[0]
    incorrect = np.where(np.abs(T_sp - T) > 1)[0]
    T_correct = T[correct]
    T_incorrect = T[incorrect]

    plt.figure(figsize=(10, 5))
    plt.plot(times, T)
    # plt.scatter(times[correct], T_correct, c='g')
    # plt.scatter(times[incorrect], T_incorrect, c='r')
    plt.plot(times, T_sp)
    # plt.plot(times, T_sp+1, 'k--')
    # plt.plot(times, T_sp-1, 'k--')
    plt.xlabel("Czas [s]")
    plt.ylabel("Temperatura [°C]")
    plt.title('Temperatura')
    plt.legend(["Temperatura", "Wartość zadana"])
    plt.savefig('./plot/temperature.png')

    plt.figure(figsize=(10, 5))
    plt.plot(times, tf.squeeze(rewards))
    plt.xlabel("Czas [s]")
    plt.ylabel("Nagroda")
    plt.title('Nagroda')
    plt.savefig('./plot/rewards.png')

    plt.figure(figsize=(10, 5))
    plt.plot(times, tf.squeeze(controls))
    plt.xlabel("Czas [s]")
    plt.ylabel("Sterowanie")
    plt.title('Sterowanie')
    plt.savefig('plot/controls.png')

    pd.DataFrame({"time":times, "T": T, "T_sp": T_sp, "rewards":rewards, "controls":controls}).to_csv("./csv_data/test_policy.csv")



if __name__ == '__main__':
    main()
