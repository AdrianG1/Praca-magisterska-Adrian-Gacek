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


# genertor nastaw temperatury
def setpoint_gen(clk, seed=123141):
    rng = random.Random(seed)
    lower_constraint = 200  # minimalny okres zmian nastaw temperatury [s]
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
    #EPISODE_TIME = 9999999999 * 60 #[s]
    C_COEF = 1              # waga składnika nagrody za odstępstwa temperatury od komfortu
    E_COEF = 0              # waga składnika nagrody za wykorzystaną energię
    COMFORT_CONSTR = 1      #[*C]  dopuszczalne odstępstwa od nastawionej wartości
    NUM_OF_ACTIONS = 5      # liczba akcji
    STEP = 0.5


    def __init__(self, discret=False, episode_time=60, seed=123141, num_actions=5):
        # parametry symulacji
        self.discret = discret # True jeśli akcje są dyskretne
        self.NUM_OF_ACTIONS = num_actions
        self.episode_time = episode_time * 60 
        self._seed = seed 
        # inicjalizacja stanu początkowego
        self.state = SystemState()

        # inicjalizacja specyfikacji
        self.__gen_spec()

        # inicjalizacja pamięci aktualnego stanu
        self.time = 0
        self.reward = 0
        self._episode_ended = False

        #inicjalizacja time_step
        self._current_time_step = ts.restart(self.state.as_array())


    def __del__(self):
        try:
            self.lab.close()
        except AttributeError:
            pass

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

        with open(os.devnull, 'w') as f:
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):

                # zamyka poprzednie laboratorium jeśli istnieje
                try:    
                    self.lab.close()
                except AttributeError:
                    pass

                # Otwieram nowe laboratorium z nowym zegarem i generatorem nastaw
                lab =  tclab.setup(connected=False, speedup=self.SPEEDUP)
                self.lab = lab()
                self.clk = tclab.clock(self.episode_time)
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
            self._current_time_step = ts.termination(current_state, current_reward)
        else:
            self._current_time_step = ts.transition(current_state, current_reward)
        return self._current_time_step

    def __calculate_reward(self, action):
        """ Wyznaczanie nagrody """

        # komfort temperaturowy
        diff = np.sqrt(np.abs(self.state.T_diff))
        comfort = -diff+4 if diff < 4 else -diff
        
        # zużycie energii
        energy = action / 100
        return comfort * self.C_COEF + energy * self.E_COEF

    def __state_update(self):
        """aktualizacja stanów normalizacja do wartości 0-1"""
        T =  self.lab.T1
        T_sp = next(self._T_gen)
        # self.state.T = T / 100             
        self.state.T_sp = T_sp
        self.state.T_diff = (T_sp - T) #/ 200 + 0.5

    def __set_control(self, action):
        """ Ustawianie sterowania """
        if self.discret:
            self.lab.Q1((100 / (self.NUM_OF_ACTIONS - 1)) * action)
        else:
            self.lab.Q1(max(min(1, action), 0)*100)

    def __gen_spec(self):
        """ generowanie specyfikacji określającej parametry środowiska """

        if self.discret:
            self._action_spec = array_spec.BoundedArraySpec(
                shape=(), dtype=np.float32, minimum=0, maximum=self.NUM_OF_ACTIONS-1, name='action')
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

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec
