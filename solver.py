import random
from Enum import enum
from abc import ABC, abstractmethod 
# States
# Actions

# State[action] => Prob To new States

# States 
# Actions
# Rewards
class UpdateRule(Enum):
    DURING_SWEEP = "DURING SWEEP"
    AFTER_SWEEP = "AFTER SWEEP"

class ValueIterator(ABC):
    def __init__(self, update_rule: UpdateRule):
        self.update_rule = update_rule

    @abstractmethod
    def get_states(self, states, **kwargs):
        pass

    @abc.abstractproperty
    def update_rule(self):
        pass

    
    def iterate(self, mdp, states, reward, gamma, V_0):
        V = V_0
        V_copy = None if self.update_rule != UpdateRule.AFTER_SWEEP else V_0 # Deep Copy
        
        # If update during sweep, use the same v
        # If not updating during sweep, store and update later
        
        for state in self.get_states(states):
            costs = []
            for action in state.actions:
                state_action_cost = reward(state, action) + gamma * mdp.prob(s, a) * V[s, a]

                costs.append(state_action_cost)

            new_cost = max(costs)
            
            if self.update_rule == UpdateRule.DURING_SWEEP:
                V[state] = new_cost
                
            elif self.update_rule == UpdateRule.AFTER_SWEEP:
                V_copy[state] = new_cost

        # Return values at the end
        if self.update_rule == UpdateRule.AFTER_SWEEP:
            return V_copy
        elif self.update_rule == UpdateRule.DURING_SWEEP:
            return V



class ClassicVI(ValueIterator):
    def get_states(self, states, **kwargs):
        return states

    @property
    def update_rule(self):
        return UpdateRule.AFTER_SWEEP

class RandomVI(ValueIterator):
    def __init__(self, k):
        super().__init__()
        self.k = k
    
    def get_state(self, states, **kwargs):
        return random.sample(states, k)

    @property
    def update_rule(self):
        return UpdateRule.AFTER_SWEEP

class CyclicVI(ValueIterator):
    def get_state(self, states, **kwargs):
        return states

    @property
    def update_rule(self):
        return UpdateRule.DURING_SWEEP

class RandomCyclicVI(ValueIterator):
    def get_state(self, states, **kwargs):
        return random.sample(states)

    @property
    def update_rule(self):
        return UpdateRule.DURING_SWEEP