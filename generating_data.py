from PID import PID
from environmentv3 import Environment
from utils import plot_trajs, abnormalize_state
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pickle

def plot_pd_trajs(trajectory):
    """ Plot kontrolny wczytywanej trajektorii"""
    states = trajectory[["T_sp", "T_błąd"]]
    rewards = trajectory["Nagrody"]
    actions= trajectory["Akcje"]

    plt.figure()
    plt.plot(states)
    plt.title('trajectory states')
    plt.savefig('./plot/states_trajs.png')    
    plt.figure()
    plt.plot(rewards)
    plt.title('trajectory reward')
    plt.savefig('./plot/reward_trajs.png')
    plt.figure()
    plt.plot(actions)
    plt.title('trajectory actions')
    plt.savefig('./plot/actions_trajs.png')



def collect_step(environment, control_function):
    time_step = environment.current_time_step()
    action = control_function((time_step.observation),
                               environment.time)
    next_time_step = environment.step(action)

    if next_time_step.is_last():
        environment.reset()

    return time_step, action, next_time_step
    

def collect_data(env, control_function):
    states = []
    rewards = []
    actions = []
    times = []

    episode_running_flag = True
    while episode_running_flag:
        time_step, action, next_time_step = collect_step(env, control_function)
        states.append(time_step.observation)
        rewards.append(next_time_step.reward)
        actions.append(action)
        times.append(env.time)
        episode_running_flag = not next_time_step.is_last() 

    return states, rewards, actions, times

def analyze_data(data):
    print(data.head(5))
    print("\n")
    print(data.info())



def main(argv=None):
    # Inicjalizacja środowiska i regulatora
    env = Environment(discret=False, episode_time=300, seed=2137, e_coef=1)
    env.reset()
    pid = PID()

    # Zbieranie danych
    states, rewards, actions, times = collect_data(env, pid.control)
    del env

    # Konwersja na DataFrame
    states = pd.DataFrame(states, columns=["T_sp", "T_błąd"])
    rewards = pd.DataFrame(rewards, columns=["Nagrody"])
    actions = pd.DataFrame(actions, columns=["Akcje"])
    times = pd.DataFrame(times, columns=["Czas"])
    # states["T_sp"] = (states["T_sp"]-30) /40
    # states["T_błąd"] = (states["T_błąd"]+50) /100
    # Rozszerzenie przestrzeni stanu
    #states["T_błąd"] = states["T_zadana"] - states["T"]

    # states.to_csv("./csv_data/states.csv")
    # rewards.to_csv("./csv_data/rewards.csv")
    # actions.to_csv("./csv_data/actions.csv")

    #states = states.drop(["T_zadana"], axis=1)
    # rewards["Nagrody"] = states["T_błąd"].abs()

    trajectory = pd.concat((states, rewards, actions, times), axis=1)

    print("\n Trajektoria \n")
    analyze_data(trajectory) 
    trajectory = trajectory.dropna()  # Usuwanie wierszy z brakującymi wartościami
    # trajectory.to_csv("./csv_data/trajectory_powt.csv")
    # print(trajectory[trajectory.duplicated(keep=False)].head(60))
    trajectory = trajectory.drop_duplicates()

    scaler = StandardScaler()
    trajectory[['Nagrody']] = scaler.fit_transform(trajectory[['Nagrody']])

    # zapis scalera nagród
    with open('scaler.pkl', 'wb') as file:
        pickle.dump(scaler, file)

    print(" Oczyszczona, znormalizowana trajektoria")
    analyze_data(trajectory) 

    plot_pd_trajs(trajectory)
    trajectory.to_csv("./csv_data/trajectory8.csv")


if __name__ == '__main__':
    main()