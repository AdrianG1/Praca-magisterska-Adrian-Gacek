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
def setpoint_gen(clk):
    rng = random.Random(2137)
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
        self.T_sp = 0   # nastawiona wartość temperatury
        self.T = 0      # aktualna wartość temperatury
        # self.stdev = 0  # odchylenie standardowe
        # self.skew = 0   # skośność
        # self.diff = 0   # pochodna
        # self.eps = 0    # uchyb regulacji

    def __str__(self):
        return f"State: [T_sp:{self.T_sp:.02f},T:{self.T:.02f}]"

    def as_tensor(self):
        return tf.constant([self.T, self.T_sp], dtype=tf.float32)

    def as_array(self):
        return np.array([self.T, self.T_sp], dtype=np.float32)


class Environment(py_environment.PyEnvironment):
    SPEEDUP = 10000
    EPISODE_TIME = 90 * 60  #[s]
    C_COEF = 1              # waga składnika nagrody za odstępstwa temperatury od komfortu
    E_COEF = 0              # waga składnika nagrody za wykorzystaną energię
    COMFORT_CONSTR = 1      #[*C]  dopuszczalne odstępstwa od nastawionej wartości
    MEM_LEN = 10            # wielkość pamięci przechoowującej ostatnie temperatury
    NUM_OF_ACTIONS = 5      # liczba akcji
    STEP = 0.5


    def __init__(self, discret=False):
        # parametry symulacji
        self.discret = discret # True jeśli akcje są dyskretne

        # Inicjalizacja cyfrowego bliźniaka
        lab =  tclab.setup(connected=False, speedup=self.SPEEDUP)
        self.lab = lab()
        # self.clk = tclab.clock(self.EPISODE_TIME)
        self.clk = tclab.clock(self.EPISODE_TIME, step=self.STEP)

        # inicjalizacja generatora nastaw
        self._T_gen = setpoint_gen(self.clk)

        # inicjalizacja stanu początkowego
        self.state = SystemState()
        self.__state_update()

        # inicjalizacja specyfikacji
        self.__gen_spec()

        # inicjalizacja pamięci aktualnego stanu
        self.time = 0
        self.reward = 0
        self._episode_ended = False

        #inicjalizacja time_step
        self._current_time_step = ts.restart(self.state.as_array())


    def __del__(self):
        self.lab.close()

    def reset(self):
        """Return initial_time_step."""
        self._current_time_step = self._reset()
        return self._current_time_step

    def step(self, action):
        """Apply action and return new time_step."""
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
                self.lab.close()
                lab =  tclab.setup(connected=False, speedup=self.SPEEDUP)
                self.lab = lab()
                self.clk = tclab.clock(self.EPISODE_TIME, step=self.STEP)
                self._T_gen = setpoint_gen(self.clk)

        self.__set_control(0)
        self.__state_update()
        self._episode_ended = False
        self._current_time_step = ts.restart(self.state.as_array())
        return self._current_time_step

    def _step(self, action):

        try:
            self.time = next(self.clk)
            self.done = (self.time >= self.EPISODE_TIME)
        except StopIteration:
            self.done = True
        # print(self.time, self.state, self.done, action)
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
        diff = abs(self.state.T - self.state.T_sp)
        comfort = -diff
        # Wykorzystana energia TODO liniowo?
        energy = action / 100
        return comfort * self.C_COEF + energy * self.E_COEF

    def __state_update(self):
        self.state.T_sp = next(self._T_gen) - self.lab.T1
        self.state.T = self.lab.T1
        #self.state

    def __set_control(self, action):
        if self.discret:
            self.lab.Q1((100 / (self.NUM_OF_ACTIONS - 1)) * action)
        else:
            self.lab.Q1(max(min(100, action), 0))

    def __gen_spec(self):

        if self.discret:
            self._action_spec = array_spec.BoundedArraySpec(
                shape=(), dtype=np.float32, minimum=0, maximum=self.NUM_OF_ACTIONS-1, name='action')
        else:
            self._action_spec = array_spec.BoundedArraySpec(
                shape=(), dtype=np.float32, minimum=0, maximum=100.0, name='action')

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
