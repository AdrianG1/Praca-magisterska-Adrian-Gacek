import tensorflow as tf
import numpy as np

class PID():
    Kc   = 6.0
    tauI = 75.0 # sec
    tauD = 0.0  # sec

    prev_y = 0
    eps_int = 0
    prev_t = 0


    def control(self, observation, time):
        y, sp = observation[0], observation[1]
        error = sp - y

        self.eps_int += error

        dt = time - self.prev_t
        dy = y - self.prev_y
        eps_dt = dy / dt if dt > 0 else 0

        control = self.Kc * (error + self.eps_int/self.tauI + self.tauD*eps_dt) 
        return control