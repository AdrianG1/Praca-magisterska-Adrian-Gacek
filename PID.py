import tensorflow as tf
import numpy as np
from utils import abnormalize_state

Kc, tauI, tauD = 5.95 , 63, 0.0

class PID():

    def __init__(self):
        self.Kc   = 14.55 
        self.tauI = 60.06 # sec
        self.tauD = 15.02  # sec

        self.windup_guard = 2000

        self.prev_y = 0
        self.epsilon_integral = 0
        self.prev_t = 0


    def control(self, observation, time):
        real_state = abnormalize_state(observation)
        y, sp = real_state[0], real_state[1]
        error = sp - y



        dt = time - self.prev_t
        dy = y - self.prev_y

        self.epsilon_integral += error# * dt
        # if self.epsilon_integral < -self.windup_guard:
        #     self.epsilon_integral = -self.epsilon_integral
        # elif self.epsilon_integral > self.windup_guard:
        #     self.epsilon_integral = self.windup_guard

        derivative = dy / (dt + 1e-10) 

        control = self.Kc * (error + self.epsilon_integral/self.tauI + self.tauD*derivative) 
        control = min(100, max(control, 0)) / 100

        return control