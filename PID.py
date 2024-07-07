import tensorflow as tf
import numpy as np
from utils import abnormalize_state

class PID():
    Kc   = 6.0 
    tauI = 75.0 # sec
    tauD = 0.0  # sec

    windup_guard = 2000

    prev_y = 0
    epsilon_integral = 0
    prev_t = 0


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

        derivative = dy / dt if dt > 0 else 0

        control = self.Kc * (error + self.epsilon_integral/self.tauI + self.tauD*derivative) 
        control = min(100, max(control, 0)) / 100

        return control