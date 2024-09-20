from environmentv3_double import Environment
import tclab
import tensorflow as tf
from tf_agents.environments import tf_py_environment
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import psutil
from tf_agents.trajectories import time_step as ts
from tqdm import tqdm
from utils import abnormalize_state
from PID import PID

def log_memory_usage():
    process = psutil.Process(os.getpid())
    memory_use = process.memory_info().rss / 1024 ** 2  # memory use in MB
    print(f'\n\n\n\n Memory usage: {memory_use:.2f} MB')

def create_environment(lab, clk, heater_no, seed=9171518):
    return Environment(discret=False, episode_time=120, seed=seed, lab_clk=clk, lab=lab, heater_no=heater_no)


def main():
    log_memory_usage()

    lab = tclab.setup(connected=True)()
    clk = tclab.clock(99999999999, step=0.25)

    env1 = tf_py_environment.TFPyEnvironment(create_environment(lab, clk, 1, seed=9171518))
    env2 = tf_py_environment.TFPyEnvironment(create_environment(lab, clk, 2, seed=3131441))
    log_memory_usage()
    env1.reset()
    env2.reset()

    rewards = [[], []]
    T = [[], []]
    T_sp = [[], []]
    controls = [[], []]
    times = []
    
    
    log_memory_usage()
    for i in tqdm(range(38, 40, 44)):
        sum_diff = 0
        try:
            policy_path = (f'PID')
            pid1 = PID(Kc=0.11, tauI=26, tauD=6.5)
            pid2 = PID(Kc=0.3, tauI=34, tauD=8.5)
            time_step1 = env1.reset()
            time_step1 = ts.TimeStep(
                    step_type=tf.reshape(time_step1.step_type, (1, )),
                    reward=tf.reshape(time_step1.reward, (1, )),
                    discount=time_step1.discount,
                    observation= tf.reshape(time_step1.observation, (1, 2))
                    )
            time_step2 = env2.reset()
            time_step2 = ts.TimeStep(
                    step_type=tf.reshape(time_step2.step_type, (1, )),
                    reward=tf.reshape(time_step2.reward, (1, )),
                    discount=time_step2.discount,
                    observation= tf.reshape(time_step2.observation, (1, 2))
                    )
            
            while not time_step1.is_last():
                # pierwsza grzałka - sterowanie
                action1 = pid1.control((time_step1.observation[0]), env1.envs[0].time)
                time_step_1 = env1.step(action1 if i != 34 else 0)
                time_step1 = ts.TimeStep(
                        step_type=tf.reshape(time_step_1.step_type, (1, )),
                        reward=tf.reshape(time_step_1.reward, (1, )),
                        discount=time_step_1.discount,
                        observation= tf.reshape(time_step_1.observation, (1, 2))
                        )

                state1, reward1, done1, info1 = time_step_1.observation, time_step1.reward, time_step1.step_type, 0

                # druga grzałka
                action2 = pid2.control((time_step2.observation[0]), env2.envs[0].time)
                time_step_2 = env2.step(action2 if i > 32 else 0)
                time_step2 = ts.TimeStep(
                        step_type=tf.reshape(time_step_2.step_type, (1, )),
                        reward=tf.reshape(time_step_2.reward, (1, )),
                        discount=time_step_2.discount,
                        observation= tf.reshape(time_step_2.observation, (1, 2))
                        )

                state2, reward2, done2, info2 = time_step_2.observation, time_step2.reward, time_step2.step_type, 0

                # zapis
                sum_diff += np.abs(time_step1.observation[:, 1].numpy()) + np.abs(time_step2.observation[:, 1].numpy())
                rewards[0].append(reward1)
                rewards[1].append(reward2)

                real_state1 = abnormalize_state(state1)
                real_state2 = abnormalize_state(state2)

                T[0].append(real_state1[:, 0])
                T[1].append(real_state2[:, 0])

                T_sp[0].append(real_state1[:, 1])
                T_sp[1].append(real_state2[:, 1])

                controls[0].append(action1)
                controls[1].append(action2)

                times.append(env1.envs[0].time)

                if len(times) % 20 == 0:
                    tqdm.write(f"time: {env1.envs[0].time}, T1: {real_state1[:, 0]}, , T_sp1: {real_state1[:, 1]} \
                    T2: {real_state2[:, 0]}, , T_sp2: {real_state2[:, 1]}, Q1: {env1.envs[0].lab.Q1()}, Q2: {env2.envs[0].lab.Q2()}")
                if real_state1[0, 0] >100 or real_state2[0, 0] > 100:
                    break
                    

            tqdm.write(str([policy_path, float(sum_diff)]))
        except KeyboardInterrupt:
            tqdm.write(f"next {i+1}")
        except StopIteration:
            tqdm.write(f"next {i+1}")
        env1.step(0)
        env2.step(0)

    print("Pierwsza grzałka")
    mean_energy = np.sum([controls[0][i]  * (times[i+1]-times[i]) for i in range(len(times)-1)]) / times[-1] *100
    mean_reward = np.sum([rewards[0][i]   * (times[i+1]-times[i]) for i in range(len(times)-1)]) / times[-1] 

    error = np.abs(np.array(T_sp[0])-np.array(T[0]))
    mean_error  = np.sum([error[i]     * (times[i+1]-times[i]) for i in range(len(times)-1)]) / times[-1] 

    comfort = [(times[i+1]-times[i]) if np.abs(T_sp[0][i] - T[0][i]) < 2 else 0 for i in range(len(times)-1)]
    comfort_time = np.sum(comfort)

    print(f"średnie zużycie energii:     {mean_energy:.02f} [%]")
    print(f"średnia nagroda:             {mean_reward:.02f} [1]")
    print(f"średni uchyb temperatury:    {mean_error:.02f}  [°C]")
    print(f"czas w komfortowym zakresie: {comfort_time:.02f}[s]")

    print("Druga grzałka")
    mean_energy = np.sum([controls[1][i]  * (times[i+1]-times[i]) for i in range(len(times)-1)]) / times[-1] *100
    mean_reward = np.sum([rewards[1][i]   * (times[i+1]-times[i]) for i in range(len(times)-1)]) / times[-1] 

    error = np.abs(np.array(T_sp[1])-np.array(T[1]))
    mean_error  = np.sum([error[i]     * (times[i+1]-times[i]) for i in range(len(times)-1)]) / times[-1] 

    comfort = [(times[i+1]-times[i]) if np.abs(T_sp[1][i] - T[1][i]) < 2 else 0 for i in range(len(times)-1)]
    comfort_time = np.sum(comfort)

    print(f"średnie zużycie energii:     {mean_energy:.02f} [%]")
    print(f"średnia nagroda:             {mean_reward:.02f} [1]")
    print(f"średni uchyb temperatury:    {mean_error:.02f}  [°C]")
    print(f"czas w komfortowym zakresie: {comfort_time:.02f}[s]")

    for i in range(2):
        TT = np.array(T[i]).reshape(-1)
        TT_sp = np.array(T_sp[i]).reshape(-1)
        Trewards = np.array(rewards[i]).reshape(-1)
        times = np.array(times).reshape(-1)

        plt.figure(figsize=(10, 5))
        plt.plot(times, TT)
        plt.plot(times, TT_sp)
        plt.xlabel("Czas [s]")
        plt.ylabel("Temperatura [°C]")
        plt.title('Temperatura')
        plt.legend(["Temperatura", "Wartość zadana"])
        plt.savefig(f'./plot/temperature{i+1}.png')

        plt.figure(figsize=(10, 5))
        plt.plot(times, tf.squeeze(Trewards))
        plt.xlabel("Czas [s]")
        plt.ylabel("Nagroda")
        plt.title('Nagroda')
        plt.savefig(f'./plot/rewards{i+1}.png')

        plt.figure(figsize=(10, 5))
        plt.plot(times, tf.squeeze(controls[i]))
        plt.xlabel("Czas [s]")
        plt.ylabel("Sterowanie")
        plt.title('Sterowanie')
        plt.savefig(f'plot/controls{i+1}.png')

        pd.DataFrame({"time":times, "T": T[i], "T_sp": T_sp[i], 
                      "rewards":rewards[i], "controls":controls[i]}).to_csv(f"./csv_data/test_policy{i+1}.csv")



if __name__ == '__main__':
    main()
