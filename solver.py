import random
from enum import Enum
from abc import ABC, abstractmethod, abstractproperty
import copy
# States
# Actions

# States 
# Actions
# Rewards
class UpdateRule(Enum):
    DURING_SWEEP = "DURING SWEEP"
    AFTER_SWEEP = "AFTER SWEEP"

class ValueIterator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_states(self, states, **kwargs):
        pass

    @abstractproperty
    def update_rule(self):
        pass

    @property
    def name(self):
        return type(self).__name__

    
    def iterate(self, mdp, gamma, V_0):
        V = V_0
        V_copy = None if self.update_rule != UpdateRule.AFTER_SWEEP else copy.deepcopy(V_0)
        
        # If update during sweep, use the same v
        # If not updating during sweep, store and update later
        
        for state in self.get_states(mdp.states):
            costs = []
            for action in mdp.actions(state):
                # Immediate reward
                state_action_cost = mdp.reward(state, action)
                for (next_state, p) in mdp.transition(state, action):
                    # Values based on next states
                    state_action_cost += gamma * p * V[next_state]

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
    
    def get_states(self, states, **kwargs):
        return random.sample(states, self.k)

    @property
    def update_rule(self):
        return UpdateRule.AFTER_SWEEP

    @property
    def name(self):
        return "{} (k={})".format(type(self).__name__, self.k)

class CyclicVI(ValueIterator):
    def get_states(self, states, **kwargs):
        return states

    @property
    def update_rule(self):
        return UpdateRule.DURING_SWEEP

class RandomCyclicVI(ValueIterator):
    def get_states(self, states, **kwargs):
        return random.sample(states)

    @property
    def update_rule(self):
        return UpdateRule.DURING_SWEEP



class Solver:
    def __init__(self, iterator, mdp, gamma):
        self.iterator = iterator
        self.mdp = mdp
        self.gamma = gamma
        self.solution = None
        self.deltas = None

    def solve(self, steps=100):
        V = { state: 0 for state in self.mdp.states}
        deltas = []
        for i in range(steps):
            V_new = self.iterator.iterate(self.mdp, self.gamma, V)
            
            # Calculate the delta by seeing the biggest change between the two versions
            delta = max([abs(V_new[state] - V[state]) for state in V])
            deltas.append(delta)

            V = V_new

        self.solution = V
        self.deltas = deltas


    def plot_delta(self):
        deltas = self.deltas
        if deltas is not None:
            label = self.iterator.name
            plt.plot(deltas, label=label, alpha=0.3)
            plt.xlabel("Iteration")
            plt.ylabel("Delta")

