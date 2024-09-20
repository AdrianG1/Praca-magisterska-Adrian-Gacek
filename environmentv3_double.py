import tclab
import random
import os
# Keep using keras-2 (tf-keras) rather than keras-3 (keras).
#os.environ['TF_USE_LEGACY_KERAS'] = '1'

#import tensorflow as tf
import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories.time_step import TimeStep
import contextlib
import pickle


# genertor nastaw temperatury
def setpoint_gen(clk, seed=123141):
    rng = random.Random(seed)
    lower_constraint = 240  # minimalny okres zmian nastaw temperatury [s]
    upper_constraint = 600  # maksymalny okres zmian nastaw temperatury [s]
    min_T = 30              # minimalna wartość nastaw temperatury [*C]
    max_T = 70              # maksymalna wartość nastaw temperatury [*C]

    last_change = 0                         # czas ostatniej zmiany nastawy
    T_sp = rng.randint(min_T, max_T)           # wartość następnej nastawy temperatury
    next_change = rng.randint(lower_constraint, upper_constraint)  # czas następnej zmiany temperatury
    
    while True:
        yield T_sp
        try:
            time = next(clk)
            if last_change + next_change <= time:
                T_sp = rng.randint(min_T, max_T)
                next_change = rng.randint(lower_constraint, upper_constraint)
                last_change = time
        except StopIteration:
            pass


class SystemState():
    size = 2


    def __init__(self):
        self.T_sp = 0      # nastawiona wartość temperatury
        # self.T = 0          # aktualna wartość temperatury
        self.T_diff = 0     # błąd sterowania


    def __str__(self):
        return f"State: [T_diff:{self.T_diff:.02f},T_sp:{self.T_sp:.02f}]"

    def as_tensor(self):
        return tf.constant([self.T_sp, self.T_diff], dtype=tf.float32)

    def as_array(self):
        return np.array([self.T_sp, self.T_diff], dtype=np.float32)

    

class Environment(py_environment.PyEnvironment):
    SPEEDUP = 100

    def __init__(self, discret=False, episode_time=60, seed=123141, 
                 num_actions=5, scaler_path=None, c_coef=1, e_coef=0, 
                 log_steps=False, env_step_time=1, connected=False, 
                 heater_no=1, lab=None, lab_clk=None):
        
        # parametry symulacji
        self.discret = discret                  # True jeśli akcje są dyskretne
        self.num_of_actions = num_actions       # określa liczbę akcji
        self.episode_time = episode_time * 60   # czas trwania eksperymentu
        self._seed = seed 
        self.log_step = log_steps
        self.env_step_time = env_step_time
        self.connected = connected
        self.lab = lab
        self.heater_no = heater_no
        self.clk = lab_clk

        self.c_coef = c_coef                    # współczynniki do wyznaczania ...  
        self.e_coef = e_coef                    # nagrody
        
        # inicjalizacja stanu początkowego
        self.state = SystemState()

        # inicjalizacja specyfikacji
        self.__gen_spec()

        # inicjalizacja pamięci aktualnego stanu
        self.time = 0
        self.reward = 0
        self._episode_ended = False
        self.last_action = 0

        #inicjalizacja time_step
        self._current_time_step = ts.restart(self.state.as_array())

        # scaler normalizacji nagród
        if scaler_path is None:
            self.__normalize_reward = lambda x: x
        elif isinstance(scaler_path, str):
            with open(scaler_path, 'rb') as file:
                self._scaler = pickle.load(file)
                self.__normalize_reward = lambda x: float(self._scaler.transform([[x]]))
        else:
            raise TypeError("scaler_path should be str or None")
        
        # inicjalizacja pliku logów
        if self.log_step:
            with open('environment.log', 'w') as file:
                file.writelines([f"T_sp,T_błąd,Nagrody,Akcje\n"])


    def reset(self):
        """Zwraca pierwszy time_step."""
        self._current_time_step = self._reset()
        return self._current_time_step

    def step(self, action):
        """Aplikuje akcję i zwraca time_step."""
        if self._current_time_step is None:
            return self.reset()
        self._current_time_step = self._step(action)
        return self._current_time_step

    def current_time_step(self):
        return self._current_time_step


    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def time_step_spec(self):
        return self._time_step_spec

    def _reset(self):
        # Otwieram nowe laboratorium z nowym zegarem i generatorem nastaw
        if self.clk is None and self.lab is not None:
            ValueError("Passed tclab didnt get corresponding clock as argument")
        if self.clk is not None and self.lab is None:
            ValueError("Passing clock without lab argument may cause errors")

        if self.lab is None:
            if self.connected:
                lab =  tclab.setup(connected=True)
            else:
                lab =  tclab.setup(connected=False, speedup=self.SPEEDUP)

            self.lab = lab()
            self.clk = tclab.clock(self.episode_time, step=self.env_step_time)

        if self.clk is not None:
            self._T_gen = setpoint_gen(self.clk, self._seed)


        self.__set_control(0)
        self.__state_update()
        self._episode_ended = False
        self._current_time_step = ts.restart(self.state.as_array())
        return self._current_time_step

    def _step(self, action):

        try:
            self.time = next(self.clk)
            self.done = (self.time >= self.episode_time)
        except StopIteration: # Jeśli czas epizodu się skończył
            self.done = True

        self.__set_control(action)
        self.__state_update()
        self.reward = self.__calculate_reward(action)

        #info = 0

        #return self.state.T, reward, done, info
        current_reward = np.array(self.reward, dtype=np.float32)
        current_state = self.state.as_array()
        if self.done:
            self._current_time_step = ts.termination(current_state, 
                                                     current_reward)
        else:
            self._current_time_step = ts.transition(current_state, 
                                                    current_reward)

        if self.log_step:
            with open('environment.log', 'a') as file:
                file.writelines([f"{self.state.T_sp},{self.state.T_diff},{current_reward},{action}\n"])

        return self._current_time_step

    def __calculate_reward(self, action):
        """ Wyznaczanie nagrody """

        # komfort temperaturowy
        abs_diff = np.abs(self.state.T_diff)
        sqrt_diff = np.sqrt(abs_diff)
        comfort =  -sqrt_diff #+3 if abs_diff < 4 else -sqrt_diff

        # # zużycie energii
        # # energy = action/(np.abs(self.state.T_diff)+1)

        # # nagłe zmiany
        energy = -np.abs(action - self.last_action)
        self.last_action = action
        return self.__normalize_reward(comfort * self.c_coef + self.e_coef * energy) #
    


    def __state_update(self):
        """aktualizacja stanów"""
        if self.heater_no == 1:
            T = self.lab.T1
        if self.heater_no == 2:
            T = self.lab.T2
        if self.heater_no == 3:
            T = self.lab.T3
        if self.heater_no == 4:
            T = self.lab.T4

        T_sp = next(self._T_gen)

        self.state.T_sp = T_sp
        self.state.T_diff = (T_sp - T)

        # if int(self.time) % 10 == 0:
        #     print(self.state) 

    def __set_control(self, action):
        """ Ustawianie sterowania """

        if self.discret:
            self.__set_Qn((100 / (self.num_of_actions - 1)) * action)
        else:
            self.__set_Qn(max(min(1, action), 0)*100)

    def __gen_spec(self):
        """ generowanie specyfikacji określającej parametry środowiska """

        if self.discret:
            self._action_spec = array_spec.BoundedArraySpec(
                shape=(), dtype=np.float32, minimum=0, maximum=self.num_of_actions-1, name='action')
        else:
            self._action_spec = array_spec.BoundedArraySpec(
                shape=(), dtype=np.float32, minimum=0, maximum=1.0, name='action')

        # Define the observation spec (state)
        self._observation_spec = array_spec.ArraySpec(
            shape=(self.state.size,), dtype=np.float32, name='observation')

        # Define the reward spec
        self._reward_spec = array_spec.ArraySpec(shape=(), dtype=np.float32, name='reward')

        # Define the discount spec
        self._discount_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.float32, name='discount', minimum=0.0, maximum=1.0)

        # Define the step type spec
        self._step_type_spec = array_spec.ArraySpec(shape=(), dtype=np.int32, name='step_type')

        self._time_step_spec = TimeStep(
            reward      = self._reward_spec,
            discount    = self._discount_spec,
            observation = self._observation_spec,
            step_type   = self._step_type_spec
        )

    def __set_Qn(self, a):
        if self.heater_no == 1:
            return self.lab.Q1(a)
        if self.heater_no == 2:
            return self.lab.Q2(a)
        if self.heater_no == 3:
            return self.lab.Q3(a)
        if self.heater_no == 4:
            return self.lab.Q4(a)

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec
