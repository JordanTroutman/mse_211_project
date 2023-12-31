from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np

import random
from enum import Enum
from abc import ABC, abstractmethod, abstractproperty
import copy
import itertools
import time
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

    def get_empirical_prob(self, counters, state, action):
        return 1

    @property
    def name(self):
        return type(self).__name__

    
    def iterate(self, mdp, gamma, V_0, counters):
        V = V_0
        V_copy = None if self.update_rule != UpdateRule.AFTER_SWEEP else copy.deepcopy(V_0)
        A = {}
        # If update during sweep, use the same v
        # If not updating during sweep, store and update later
        
        for state in self.get_states(mdp.states):
            costs = []
            max_cost = 0
            for action in mdp.actions(state):
                # Immediate reward
                state_action_cost = mdp.reward(state, action)
                for (next_state, p) in mdp.transition(state, action):
                    # Values based on next states
                    empirical_prob = self.get_empirical_prob(counters, state, action)
                    state_action_cost += gamma * p * empirical_prob * V[next_state]

                max_cost = max(max_cost, state_action_cost)

                # record the best action for the state
                if max_cost == state_action_cost:
                    counters[state][action] += 1

                costs.append((state_action_cost, action))
                
            (new_cost, _) = max(costs, default=(0, "NOACTION"), key=lambda x: x[0])
      
            # There can be multiple actions with the same policy value
            max_actions = list(map(lambda x: x[1], filter(lambda x: x[0] == new_cost, costs)))
            A[state] = max_actions
            
            if self.update_rule == UpdateRule.DURING_SWEEP:
                V[state] = max_cost

            elif self.update_rule == UpdateRule.AFTER_SWEEP:
                V_copy[state] = max_cost

        # Return values at the end
        if self.update_rule == UpdateRule.AFTER_SWEEP:
            return (V_copy, A)
        elif self.update_rule == UpdateRule.DURING_SWEEP:
            return (V, A)



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

class EmpiricalVI(ValueIterator):
    def __init__(self):
        super().__init__()

    def get_states(self, states, **kwargs):
        return states

    def get_empirical_prob(self, counters, state, action):
        counter = counters[state]
        if not counter:
            return 1
        total = 0
        for (action, count) in counter.items():
            total += count
        return counter[action] / total

    @property
    def update_rule(self):
        return UpdateRule.AFTER_SWEEP

class CyclicVI(ValueIterator):
    def get_states(self, states, **kwargs):
        return states

    @property
    def update_rule(self):
        return UpdateRule.DURING_SWEEP

class RandomCyclicVI(ValueIterator):
    def get_states(self, states, **kwargs):
        return random.sample(states, k=len(states))

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
        self.time_each_step = None
        self.policy = None

    def solve(self, steps=None, threshold=None):
        assert (steps is not None) != (threshold is not None), "either steps or threshold should be defined"
        V = { state: 0 for state in self.mdp.states}

        Policy = {}
        deltas = []
        time_each_step = []
        counters = defaultdict(Counter)

        step = 0
        while ((threshold is not None) and (len(deltas) == 0 or deltas[-1] > threshold)) or (steps is not None and step <= steps):
            t_0 = time.time()
            
            V_0 = copy.deepcopy(V)

            (V_new, A) = self.iterator.iterate(self.mdp, self.gamma, V, counters)
            
            # Calculate the delta by seeing the biggest change between the two versions
            delta = max([abs(V_new[state] - V_0[state]) for state in V])
            deltas.append(delta)

            V = V_new

            time_difference = time.time() - t_0
            time_each_step.append(time_difference)

            step += 1

            Policy = A

            

        self.solution = V
        self.deltas = deltas
        self.time_each_step = time_each_step
        self.policy = Policy


    def plot_delta(self):
        deltas = self.deltas
        if deltas is not None:
            label = self.iterator.name
            plt.plot(deltas, label=label, alpha=0.9)
            plt.xlabel("Iteration")
            plt.ylabel("Delta")


    def plot_time(self):
        time_steps = self.time_each_step
        if time_steps is not None:
            label = self.iterator.name
            plt.plot(time_steps, label=label)
            plt.xlabel("Time Step")
            plt.ylabel("Time (s)")
