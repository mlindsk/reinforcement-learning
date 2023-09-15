from state import State
import numpy as np
import matplotlib.pyplot as plt
import random

class Env:

    def __init__(
            self,
            state_space_mapping:dict[str, State],
            gamma: float = 0.9,
            epochs: int = 10000,
            max_loop_count: int = 10):

        self.state_space_mapping = state_space_mapping
        self.state_space = list(state_space_mapping.values())
        self.non_terminal_state_space = [s for s in self.state_space if not s.is_terminal()]
        self.gamma = gamma
        self.epochs = epochs
        self.terminate_loops_after = max_loop_count

        self.Q = {(s, a) : 0 for s in self.state_space for a in s.get_action_space()}
        self.G = {(s, a) : [] for s in self.state_space for a in s.get_action_space()}
        self.R = {s : [] for s in self.state_space}

        # Initialize random deterministic policy
        self.policy = {
            s : (np.random.choice(s.action_space) if not s.is_terminal() else None)
            for s in self.state_space}

        # Print initial policy grid
        policy_grid = np.array([['#' for x in range(4)] for y in range(3)])
        for s, a in self.policy.items():
            policy_grid[s.i][s.j] = a
        print('\n', 'Initial Policy Function: \n')
        print(policy_grid, '\n')            
        
    def initialize_random_policy(self):
        np.random.seed(7)
        for s in self.state_space:
            action_space = s.action_space
            action_space_prob = np.random.random_sample(len(action_space))
            action_space_prob /= sum(action_space_prob)
            action_pdf = {a : p for a, p in zip(action_space, action_space_prob)}
            self.policy.update({s : action_pdf})
            
    def play_episode(self):
        s = np.random.choice(self.non_terminal_state_space)
        a = np.random.choice(s.action_space)
        states = [s]
        actions = [a]
        rewards = [0]

        # Play an episode
        k = 0
        while not s.is_terminal():
            s = s.move(self.state_space_mapping, a)
            a = self.policy[s]
            r = s.get_reward()
            states.append(s)
            actions.append(a)
            rewards.append(r)

            if k == self.terminate_loops_after:
                return states, actions, rewards
            k += 1
            
        return states, actions, rewards

    def extract_argmax_action_from_Q(self, s: State):
        qas = {a : self.Q[(s, a)] for a in s.get_action_space()}
        max_a = None
        max_v = float("-inf")
        for a, v in qas.items():
            if v > max_v:
                max_v = v
                max_a = a
        return max_a

    def calculate_V_from_Q(self):
        qas = {s : self.Q[(s, a)] for a in s.get_action_space() for s in self.state_space}
    
    def mces(self):
        self.deltas = []
        for k in range(self.epochs):
            biggest_change = 0
            states, actions, rewards = self.play_episode()
            T = len(states)
            G = 0
            for t in range(T-2, -1, -1):
                G =  rewards[t+1] + self.gamma * G
                s = states[t]
                a = actions[t]
                if (s not in states[:t]) and (a not in actions[:t]):
                    self.G[(s,a)].append(G)
                    lG = 1 / len(self.G[(s,a)])
                    Q_old = self.Q[(s, a)]
                    self.Q[(s, a)] = Q_old + lG * (G - Q_old) # NOTE: This is the mean of G(s,a)
                    self.policy[s] = self.extract_argmax_action_from_Q(s)
                    biggest_change = max(biggest_change, np.abs(Q_old - self.Q[(s, a)]))
            self.deltas.append(biggest_change)

    def render(self, nrows:int = 3, ncols:int = 4):        
        policy_grid = np.array([['#' for x in range(ncols)] for y in range(nrows)])
        value_grid = np.array([[0.0 for x in range(ncols)] for y in range(nrows)])
        V = {s : self.Q[(s, self.extract_argmax_action_from_Q(s))] for s in self.non_terminal_state_space}
        
        for s in self.non_terminal_state_space:            
            value_grid[s.i][s.j] = V[s]

        for s, a in self.policy.items():
            policy_grid[s.i][s.j] = a

        print('\n', 'Optimal Value Function: \n')
        print(value_grid, '\n')

        print('\n', 'Optimal Policy Function: \n')
        print(policy_grid, '\n')

    def plot_deltas(self):
        plt.plot(self.deltas)
        plt.show()        

if __name__ == '__main__':

    import time
    from state import state_space_mapping


    start = time.time()
    env = Env(state_space_mapping)
    env.mces()
    env.render()
    env.plot_deltas()
    
    print(time.time() - start)
