import tclab
import gymnasium as gym
import random


# genertor nastaw temperatury
def setpoint_gen(clk):
    lower_constraint = 200  # minimalny okres zmian nastaw temperatury [s]
    upper_constraint = 600  # maksymalny okres zmian nastaw temperatury [s]

    last_change = 0                         # czas ostatniej zmiany nastawy
    T_sp = random.randint(30, 70)           # wartość następnej nastawy temperatury
    next_change = random.randint(lower_constraint, upper_constraint)  # czas następnej zmiany temperatury

    while True:
        yield T_sp

        time = next(clk)
        if last_change + next_change <= time:
            T_sp = random.randint(30, 70)
            next_change = random.randint(lower_constraint, upper_constraint)
            last_change = time
        

class SystemState():
    def __init__(self):
        # nastawiona wartość temperatury
        self.T_sp = 0
        # aktualna wartość temperatury
        self.T = 0

class Environment(gym.Env):
    SPEEDUP = 100
    EPISODE_TIME = 90 * 60 #[s]
    C_COEF = 1
    E_COEF = 0
    COMFORT_CONSTR = 1 #[*C]  dopuszczalne odstępstwa od nastawionej wartości

    def __init__(self):
        # parametry symulacji
        self.TCLab = tclab.setup(connected=False, speedup=self.SPEEDUP)

        # Inicjalizacja cyfrowego bliźniaka
        self.lab = tclab.TCLabModel() 
        self.clk = tclab.clock(self.EPISODE_TIME+2)
        
        # inicjalizacja generatora nastaw
        self._T_gen = setpoint_gen(self.clk)

        # inicjalizacja stanu początkowego
        self.state = SystemState()
        self.__state_update()




    def reset(self):
        self.clk = tclab.clock(self.EPISODE_TIME+2)
        self._T_gen = setpoint_gen(self.clk)
        return self.state

    def step(self, action):
        time = next(self.clk)
        self.__set_control(action)
        self.__state_update()
        reward = self.__calculate_reward(action)
        done = time >= self.EPISODE_TIME 
        info = 0

        return self.state, reward, done, info
    
    def __calculate_reward(self, action):
        comfort = 1 if abs(self.state.T - self.state.T_sp) <= self.COMFORT_CONSTR else 0
        energy = action / 100
        return comfort * self.C_COEF + energy * self.E_COEF

    def __state_update(self):
        self.state.T_sp = next(self._T_gen)
        self.state.T = self.lab.T1

    def __set_control(self, action):
        self.lab.Q1(max(min(100, action), 0))

