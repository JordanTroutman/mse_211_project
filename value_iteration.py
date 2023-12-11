from abc import abstractmethod
import random
import itertools
import matplotlib.pyplot as plt
import numpy as np
import copy
from collections import Counter, defaultdict
import time

class IValueIteration:
    def __init__(self, game, sample_rate=1):
        self.game = game
        self.sample_rate = sample_rate

    @abstractmethod
    def run(self, V_new):
        raise NotImplemented

class ClassicValueIteration(IValueIteration):
    def __str__(self) -> str:
        return "ClassicVI"

    def run(self, V_new, counter):
        delta = 0
        start = time.time()
        for s in self.game.states:
            max_val = 0
            for a in self.game.actions(s):
                val = self.game.rewards(s)
                for (s_next, p) in self.game.transitions(s, a):
                    val += p * (self.game.gamma * self.game.V[s_next])
                max_val = max(max_val, val)
            V_new[s] = max_val
            delta = max(delta, abs(self.game.V[s] - V_new[s]))

        self.game.V = copy.deepcopy(V_new)
        return delta, time.time() - start

class EmpiricalValueIteration(IValueIteration):
    def __str__(self) -> str:
        return "EmpiricalVI"

    def run(self, V_new, counters):
        delta = 0
        start = time.time()
        for s in self.game.states:
            max_val = 0
            max_action = None

            # guess actions
            actions = self.game.actions(s)
            next_actions = set()
            for _ in range(1): # X times
                a = self.get_action(counters, s, actions)
                if a:
                    next_actions.add(a)

            for a in next_actions:
                val = self.game.rewards(s)
                for (s_next, p) in self.game.transitions(s, a):
                        val += p * self.game.gamma * self.game.V[s_next]
                max_val = max(max_val, val)

                if max_val == val:
                    max_action = a

            # record the best action
            if max_action:
                counters[s][max_action] += 1

            V_new[s] = max_val
            delta = max(delta, abs(self.game.V[s] - V_new[s]))

        self.game.V = copy.deepcopy(V_new)
        return delta, time.time() - start

    def get_action(self, counters, state, actions):
        counter = counters[state]
        if not counter:
            # random select
            return random.choice(actions)
        total = 0
        prefix_sum = {}
        for (action, count) in counter.items():
            total += count
            prefix_sum[total] = action
        
        r = random.randint(1, total)
        for p_sum, action in prefix_sum.items():
            if r < p_sum:
                return action

        return action


class RandomValueIteration(IValueIteration):
    def __str__(self) -> str:
        return "RandomVI"

    def run(self, V_new, counter):
        start = time.time()
        delta = 0
        # rather than go through all state values in each iteration
        # randomly select a subset of states
        sampled_states = random.sample(self.game.states, int(self.sample_rate * len(self.game.states)))
        for s in self.game.states:
            if s in sampled_states:
                max_val = 0
                for a in self.game.actions(s):
                    val = self.game.rewards(s)
                    for (s_next, p) in self.game.transitions(s, a):
                        val += p * self.game.gamma * self.game.V[s_next]
                    max_val = max(max_val, val)
                V_new[s] = max_val
            delta = max(delta, abs(self.game.V[s] - V_new[s]))

        self.game.V = copy.deepcopy(V_new)
        return delta, time.time() - start

class CyclicValueIteration(IValueIteration):
    def __str__(self) -> str:
        return "CyclicVI"

    def run(self, V_new, counter):
        start = time.time()
        delta = 0
        for s in self.game.states:
            max_val = 0
            v = self.game.V[s]
            for a in self.game.actions(s):
                val = self.game.rewards(s)
                for (s_next, p) in self.game.transitions(s, a):
                    val += p * (self.game.gamma * self.game.V[s_next])
                max_val = max(max_val, val)
            self.game.V[s] = max_val
            delta = max(delta, abs(v - self.game.V[s]))
        return delta, time.time() - start

class RandomCyclicValueIteration(IValueIteration):
    def __str__(self) -> str:
        return "RandomCyclicVI"

    def run(self, V_new, counter):
        start = time.time()
        delta = 0
        # rather than go through all state values in each iteration
        # randomly select a subset of states
        sampled_states = random.sample(self.game.states, int(self.sample_rate * len(self.game.states)))
        for s in self.game.states:
            v = self.game.V[s]
            if s in sampled_states:
                max_val = 0
                for a in self.game.actions(s):
                    val = self.game.rewards(s)
                    for (s_next, p) in self.game.transitions(s, a):
                        val += p * (self.game.gamma * self.game.V[s_next])
                    max_val = max(max_val, val)
                self.game.V[s] = max_val
            delta = max(delta, abs(v - self.game.V[s]))
        return delta, time.time() - start

class VISimulation:
    def __init__(self, method):
        self.method = method

    def simulate(self, max_iter, theta):
        res = []
        iter = 0
        states = list(itertools.product(list(range(5)), list(range(5))))
        counters = defaultdict(Counter)
        time_spents = []

        V_new = { state: 0 for state in states}
        while iter < max_iter:
            (delta, time_spent) = self.method.run(V_new, counters)
            res.append(delta)
            time_spents.append(time_spent)
            if delta < theta:
                break
            iter += 1

        print(f'{self.method}: # of iterations: {len(res)}')
        print(f'{self.method}: Average time per iteration: {sum(time_spents)/ len(time_spents)}')

        plt.plot(np.arange(len(res)) + 1, res, label=str(self.method), alpha=0.3)
        plt.xlabel("Iteration")
        plt.ylabel("Delta")
