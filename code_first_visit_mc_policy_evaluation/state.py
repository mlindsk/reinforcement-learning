class State:

    def __init__(
            self,
            i: int,
            j: int,
            action_space:list[str] = None,
            reward: float = 0,
            terminal: bool = False
    ):
        self.i = i
        self.j = j
        self.action_space = action_space
        self.reward = reward
        self.terminal = terminal
        
    def __eq__(self, other) -> bool:
        return (self.i == other.i) and (self.j == other.j)

    def __hash__(self):
        return hash((self.i, self.j))
    
    def __repr__(self) -> str:
        return f"{'t' if self.terminal else 's'}({self.i}, {self.j})"
        
    def __str__(self) -> str:
        return self.__repr__()

    def is_terminal(self):
        return self.terminal

    def get_action_space(self):
        return self.action_space

    def get_reward(self):
        return self.reward

    def move(self, state_space_mapping, action: str):
        s = {
            "U" : str(self.i - 1) + str(self.j),
            "R" : str(self.i) + str(self.j + 1),
            "D" : str(self.i + 1) + str(self.j),
            "L" : str(self.i) + str(self.j - 1)
        }[action]

        return state_space_mapping[s]
            
    def reachable_states(self, state_space_mapping:dict):
        states = {
            "U" : str(self.i - 1) + str(self.j),
            "R" : str(self.i) + str(self.j + 1),
            "D" : str(self.i + 1) + str(self.j),
            "L" : str(self.i) + str(self.j - 1)
        }

        rs = [states[a] for a in self.action_space]
        ns = [state_space_mapping[s] for s in rs]
        return ns

        
non_terminal_reward = 0
s00 = State(0, 0, action_space = ['R', 'D'], reward = non_terminal_reward)
s01 = State(0, 1, action_space = ['R', 'L'], reward = non_terminal_reward)
s02 = State(0, 2, action_space = ['R', 'D', 'L'], reward = non_terminal_reward)
s03 = State(0, 3, action_space = [], reward = 1, terminal = True)
s10 = State(1, 0, action_space = ['U', 'D'], reward = non_terminal_reward)
s12 = State(1, 2, action_space = ['U', 'D', 'R'], reward = non_terminal_reward)
s13 = State(1, 3, action_space = [], reward = -1, terminal = True)
s20 = State(2, 0, action_space = ['U', 'R'], reward = non_terminal_reward)
s21 = State(2, 1, action_space = ['R', 'L'], reward = non_terminal_reward)
s22 = State(2, 2, action_space = ['U', 'R', 'L'], reward = non_terminal_reward)
s23 = State(2, 3, action_space = ['U', 'L'], reward = non_terminal_reward)

states = [
    s00, s01, s02, s03,
    s10,      s12, s13,
    s20, s21, s22, s23    
]

state_space_mapping = {str(s.i) + str(s.j) : s for s in states}

policy = {
    s00 : "R",
    s01 : "R",
    s02 : "R",
    s03 : None,
    s10 : "U",
    s12 : "R",
    s13 : None,
    s20 : "U",
    s21 : "R",
    s22 : "R",
    s23 : "U"    
}
if __name__ == '__main__':


    # print(state_space_mapping)
    # print(s10.reachable_states(state_space_mapping))
    # print(s10.move2(state_space_mapping, 'D'))

    for s in StateSpace:
        print(s, s.reachable_states(state_space_mapping))
