from state import State
import numpy as np

class Env:

    def __init__(
            self,
            state_space_mapping:dict[str, State],
            transition_probs: dict[(State, str), dict[State, float]],
            gamma: float = 0.9,
            theta:float = 1e-3):

        self.state_space_mapping = state_space_mapping
        self.state_space = state_space_mapping.values()
        self.transition_probs = transition_probs
        self.gamma = gamma
        self.theta = theta
        self.V = {s : 0 for s in self.state_space}
        self.policy = dict()


    def get_transition_prob(self, s1, s2, a):
        t = self.transition_probs.get((s1, a), None)
        return t.get(s2, 0) if t else 0
        
    def value_iteration(self):
        self.it = 0
        while True:
            max_delta = 0
            for s1 in self.state_space:
                if not s1.is_terminal():
                    old_v = self.V[s1]
                    new_v = float('-inf')
                    for a in s1.get_action_space():
                        v = 0
                        for s2 in s1.reachable_states(self.state_space_mapping):
                            r = s2.get_reward()
                            p = self.get_transition_prob(s1, s2, a)
                            v += p * (r + self.gamma * self.V[s2])

                        if v > new_v:
                            new_v = v

                    self.V[s1] = new_v
                    max_delta = max(max_delta, np.abs(old_v - self.V[s1]))

            self.it += 1
            if max_delta < self.theta:
                break            

    def find_optimal_policy(self):
        for s1 in self.state_space:
            max_a = None
            max_value = float('-inf')
            if not s1.is_terminal():
                for a in s1.get_action_space():
                    v = 0 
                    for s2 in s1.reachable_states(self.state_space_mapping):
                        r = s2.get_reward()
                        p = self.get_transition_prob(s1, s2, a)
                        v += p * (r + self.gamma * self.V[s2])

                    if v > max_value:
                        max_value = v
                        max_a = a

                self.policy[s1] = max_a
            
    def play(self):
        self.value_iteration()
        self.find_optimal_policy()
        self.render()
        
    def render(self, nrows:int = 3, ncols:int = 4):
        policy_grid = np.array([['#' for x in range(ncols)] for y in range(nrows)])
        value_grid = np.array([[0.0 for x in range(ncols)] for y in range(nrows)])

        for s in self.state_space:            
            value_grid[s.i][s.j] = self.V[s]
            if not s.is_terminal():
                policy_grid[s.i][s.j] = self.policy[s]

        print('\n', 'Policy Function: \n')
        print(policy_grid, '\n')
        print('\n', 'Value Function: \n')
        print(value_grid, '\n')
        print('Iterations:', self.it)
        

if __name__ == '__main__':

    import time
    from state import state_space_mapping, s12, s22, s02, s13

    transition_probs = {
        (s, a) : {s.move(state_space_mapping, a) : 1}
        for s in state_space_mapping.values()
        for a in s.action_space 
    }

    windy = True
    if windy:
        transition_probs[(s12, "U")] = {s02 : 0.5, s13 : 0.5}
    
    # for k, v in transition_probs.items():
    #     print(k, v)

    start = time.time()
    env = Env(state_space_mapping, transition_probs)
    env.play()
    print(time.time() - start)
