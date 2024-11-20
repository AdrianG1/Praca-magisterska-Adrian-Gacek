import tclab
import random
import os

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories.time_step import TimeStep
import contextlib
import pickle


def setpoint_gen(clk, seed=123141):
    """
    Generator yielding current setpoint based on time of clock

    :param clk: TCLab clock
    :param seed: seed for random generators  

    :yield: current setpoint
    """
    rng = random.Random(seed)
    lower_constraint = 240 # minimum period of temperature setting changes [s]
    upper_constraint = 600 # maximum period of temperature setting changes [s]
    min_T = 30             # minimum value of temperature setting [*C]
    max_T = 70             # maximum value of temperature setting [*C]

    last_change = 0                     # time of last setpoint change
    T_sp = rng.randint(min_T, max_T)    # value of next temperature setpoint
                                        # time of next temperature change
    next_change = rng.randint(lower_constraint, upper_constraint)  
    
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
    """
    Class representing state of environment (setpoint, control error)
    """

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
    """
    Class representing environment. Uses TCLab interface.
    """

    SPEEDUP = 100

    def __init__(self, discret=False, episode_time=60, seed=123141, 
                 num_actions=5, scaler_path=None, c_coef=1, e_coef=0, 
                 log_steps=False, env_step_time=1, connected=False):
        
        # parametry symulacji
        self.discret = discret                  # True if actions are discrete
        self.num_of_actions = num_actions       # defines number of discrete actions
        self.episode_time = episode_time * 60   # max time of experiment
        self._seed = seed                       # seed for random generators
        self.log_step = log_steps               # defines if logging env steps to file
        self.env_step_time = env_step_time      # time of 1 step in environment
        self.connected = connected              # True if connected with real device

        self.c_coef = c_coef                    # reward calculation coefficient (error)
        self.e_coef = e_coef                    # (energy)
        
        # init system state
        self.state = SystemState()

        # init specification
        self.__gen_spec()

        # init memory for data
        self.time = 0
        self.reward = 0
        self._episode_ended = False
        self.last_action = 0

        # init time_step
        self._current_time_step = ts.restart(self.state.as_array())

        # scaling reward normalization
        if scaler_path is None:
            self.__normalize_reward = lambda x: x
        elif isinstance(scaler_path, str):
            with open(scaler_path, 'rb') as file:
                self._scaler = pickle.load(file)
                self.__normalize_reward = lambda x: float(self._scaler.transform([[x]]))
        else:
            raise TypeError("scaler_path should be str or None")
        
        # init log file
        if self.log_step:
            with open('environment.log', 'w') as file:
                file.writelines([f"T_sp,T_błąd,Nagrody,Akcje\n"])


    def __del__(self):
        try:
            self.lab.close()
        except AttributeError:
            pass

    def reset(self):
        """ Returns first time step """
        self._current_time_step = self._reset()
        return self._current_time_step

    def step(self, action):
        """ Applies an action and returns a time_step. """
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
                # closes the previous lab if it exists
                try:    
                    self.lab.close()
                except AttributeError:
                    pass

        # opens a new lab with a new clock and preset generator
        if self.connected:
            lab =  tclab.setup(connected=True)
        else:
            lab =  tclab.setup(connected=False, speedup=self.SPEEDUP)

        self.lab = lab()
        self.clk = tclab.clock(self.episode_time, step=self.env_step_time)
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
        except StopIteration: # if episode ended
            self.done = True

        self.__set_control(action)
        self.__state_update()
        self.reward = self.__calculate_reward(action)

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
        """ Calculates rewards based on observation and action """

        # temperature comfort
        abs_diff = np.abs(self.state.T_diff)
        sqrt_diff = np.sqrt(abs_diff)
        comfort =  -sqrt_diff

        # rapid changes
        energy = -np.abs(action - self.last_action)
        self.last_action = action
        return self.__normalize_reward(comfort * self.c_coef + self.e_coef * energy) #
    


    def __state_update(self):
        """Update and normalize states to a range of 0-1.

        :return: 
        """

        T =  self.lab.T1
        T_sp = next(self._T_gen)
            
        self.state.T_sp = T_sp
        self.state.T_diff = (T_sp - T) 


    def __set_control(self, action):
        """Set control input based on the action, either discrete or continuous depending on the mode.

        :param action: The action to be applied.

        :return:
        """
    
        if self.discret:
            self.lab.Q1((100 / (self.num_of_actions - 1)) * action)
        else:
            self.lab.Q1(max(min(1, action), 0)*100)

    def __gen_spec(self):
        """
        Generate specifications defining the environment parameters.
        This method defines the action, observation, reward, discount, 
        and time step specifications for the environment.
        """

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

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec
