from PID import PID
from environmentv3 import Environment

import pandas as pd
from sklearn.preprocessing import StandardScaler


def discretize(action):
    return (min(max(action, 100), 0)+12) // 25
    

def collect_step(environment, control_function):
    time_step = environment.current_time_step()
    action = control_function(time_step.observation, environment.time)
    next_time_step = environment.step(action)

    if next_time_step.is_last():
        environment.reset()

    return time_step, action, next_time_step
    

def collect_data(env, control_function):
    states = []
    rewards = []
    actions = []

    episode_running_flag = True
    while episode_running_flag:
        time_step, action, next_time_step = collect_step(env, control_function)
        states.append(time_step.observation)
        rewards.append(next_time_step.reward)
        actions.append(action)

        episode_running_flag = not next_time_step.is_last() 

    return states, rewards, actions

def analyze_data(data):
    print(data.head(5))
    print("\n")
    print(data.info())



def main(argv=None):

    # Inicjalizacja środowiska i  regulatora
    env = Environment(discret=False)
    pid = PID()

    # Zbieranie danych
    states, rewards, actions = collect_data(env, pid.control)
    del env

    # Konwersja na DataFrame
    states = pd.DataFrame(states, columns=["T", "T_zadana"])
    rewards = pd.DataFrame(rewards, columns=["Nagrody"])
    actions = pd.DataFrame(actions, columns=["Akcje"])

    states.to_csv("./csv_data/states.csv")
    rewards.to_csv("./csv_data/rewards.csv")
    actions.to_csv("./csv_data/actions.csv")

    # Rozszerzenie przestrzeni stanu
    states["T_błąd"] = states["T_zadana"] - states["T"]
    states['T_średnia']  = states['T'].rolling(window=10,center=True).mean()
    states['T_wariancja'] = states['T'].rolling(window=10,center=True).std()
    states['T_skośność']  = states['T'].rolling(window=10,center=True).skew()

    states['T_zadana_wariancja'] = states['T_zadana'].rolling(window=10,center=True).std()
    states['T_zadana_skośność']  = states['T_zadana'].rolling(window=10,center=True).skew()



    next_states = pd.DataFrame(states[1:])
    next_states = next_states.rename(columns={"T":"T'", "T_zadana":"T_zadana'"})

    trajectory = pd.concat((states[:-1], rewards[:-1], actions[:-1], next_states), axis=1)

    print("\n Trajektoria \n")
    analyze_data(trajectory) 
    trajectory = trajectory.dropna()  # Usuwanie wierszy z brakującymi wartościami
    trajectory = trajectory.drop_duplicates()

    scaler = StandardScaler()
    trajectory[["T", 
                "T_zadana", 
                "T'", 
                "T_zadana'"]] = trajectory[["T", 
                                            "T_zadana", 
                                            "T'", 
                                            "T_zadana'"]] / 100
    trajectory[['Nagrody']] = scaler.fit_transform(trajectory[['Nagrody']])

    print(" Oczyszczona, znormalizowana trajektoria")
    analyze_data(trajectory) 


if __name__ == '__main__':
    main()