from state import State
import numpy as np
import random

class Env:

    def __init__(
            self,
            state_space_mapping:dict[str, State],
            policy: dict[State, str],
            gamma: float = 0.9):

        self.state_space_mapping = state_space_mapping
        self.state_space = list(state_space_mapping.values())
        self.gamma = gamma
        self.policy = policy

        self.V = {s : 0 for s in self.state_space}
        self.R = {s : [] for s in self.state_space}
        # self.initialize_random_policy

    def initialize_random_policy(self):
        np.random.seed(7)
        for s in self.state_space:
            action_space = s.action_space
            action_space_prob = np.random.random_sample(len(action_space))
            action_space_prob /= sum(action_space_prob)
            action_pdf = {a : p for a, p in zip(action_space, action_space_prob)}
            self.policy.update({s : action_pdf})
            
    def play_episode(self):
        s = np.random.choice(self.state_space)
        states = [s]
        rewards = [0]

        # Play an episode
        while not s.is_terminal():
            a = self.policy[s]
            s = s.move(self.state_space_mapping, a)
            r = s.get_reward()

            rewards.append(r)
            states.append(s)

        return states, rewards

    def fvmc(self):
        for k in range(50):
            states, rewards = self.play_episode()
            T = len(states)
            G = 0
            for t in range(T-2, -1, -1):
                G =  rewards[t+1] + self.gamma * G
                s = states[t]
                if s not in states[:t]:
                    self.R[s].append(G)
                    self.V[s] = np.mean(self.R[s])

    def render(self, nrows:int = 3, ncols:int = 4):
        value_grid = np.array([[0.0 for x in range(ncols)] for y in range(nrows)])

        for s in self.state_space:            
            value_grid[s.i][s.j] = self.V[s]

        print('\n', 'Value Function: \n')
        print(value_grid, '\n')
        

if __name__ == '__main__':

    import time
    from state import state_space_mapping, policy, s20


    start = time.time()
    env = Env(state_space_mapping, policy)
    env.fvmc()
    env.render()
    print(time.time() - start)
