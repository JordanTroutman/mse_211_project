from abc import abstractmethod
import random
import itertools
import matplotlib.pyplot as plt
import numpy as np
import copy

class IValueIteration:
    def __init__(self, game, sample_rate=1):
        self.game = game
        self.sample_rate = sample_rate
        self.gamma = 0.9

    @abstractmethod
    def run(self, V_new):
        raise NotImplemented

class ClassicValueIteration(IValueIteration):
    def __str__(self) -> str:
        return "ClassicVI"
    def run(self, V_new):
        delta = 0
        for s in self.game.states:
            max_val = 0
            for a in self.game.actions(s):
                val = self.game.reward(s, a)
                for (s_next, p) in self.game.transition(s, a):
                    val += p * (self.gamma * self.game.V[s_next])
                # for s_next in self.game.states:
                #     val += self.game.probs[s][s_next][a] * (self.game.gamma * self.game.V[s_next])
                max_val = max(max_val, val)
            V_new[s] = max_val
            delta = max(delta, abs(self.game.V[s] - V_new[s]))

        self.game.V = copy.deepcopy(V_new)
        return delta

class RandomValueIteration(IValueIteration):
    def __str__(self) -> str:
        return "RandomVI"

    def run(self, V_new):
        delta = 0
        # rather than go through all state values in each iteration
        # randomly select a subset of states
        sampled_states = random.sample(self.game.states, int(self.sample_rate * len(self.game.states)))
        for s in self.game.states:
            if s in sampled_states:
                max_val = 0
                for a in self.game.actions(s):
                    val = self.game.reward(s, a)
                    for (s_next, p) in self.game.transition(s, a):
                        val += p * self.gamma * self.game.V[s_next]
                    # for s_next in self.game.states:
                    #     val += self.game.probs[s][s_next][a] * (self.game.gamma * self.game.V[s_next])
                    max_val = max(max_val, val)
                V_new[s] = max_val
            delta = max(delta, abs(self.game.V[s] - V_new[s]))

        self.game.V = copy.deepcopy(V_new)
        return delta

class CyclicValueIteration(IValueIteration):
    def __str__(self) -> str:
        return "CyclicVI"

    def run(self, V_new):
        delta = 0
        for s in self.game.states:
            max_val = 0
            v = self.game.V[s]
            for a in self.game.actions(s):
                val = self.game.reward(s, a)
                for (s_next, p) in self.game.transition(s, a):
                    val += p * (self.gamma * self.game.V[s_next])
                # for s_next in self.game.states:
                #     val += self.game.probs[s][s_next][a] * (self.game.gamma * self.game.V[s_next])
                max_val = max(max_val, val)
            self.game.V[s] = max_val
            delta = max(delta, abs(v - self.game.V[s]))
        return delta

class RandomCyclicValueIteration(IValueIteration):
    def __str__(self) -> str:
        return "RandomCyclicVI"

    def run(self, V_new):
        delta = 0
        # rather than go through all state values in each iteration
        # randomly select a subset of states
        sampled_states = random.sample(self.game.states, int(self.sample_rate * len(self.game.states)))
        for s in self.game.states:
            v = self.game.V[s]
            if s in sampled_states:
                max_val = 0
                for a in self.game.actions(s):
                    val = self.game.reward(s, a)
                    for (s_next, p) in self.game.transition(s, a):
                        val += p * (self.gamma * self.game.V[s_next])
                    # for s_next in self.game.states:
                    #     val += self.game.probs[s][s_next][a] * (self.game.gamma * self.game.V[s_next])
                    max_val = max(max_val, val)
                self.game.V[s] = max_val
            delta = max(delta, abs(v - self.game.V[s]))
        return delta

class VISimulation:
    def __init__(self, method):
        self.method = method

    def simulate(self, max_iter, theta):
        res = []
        iter = 0
        states = list(itertools.product(list(range(5)), list(range(5))))
        V_new = { state: 0 for state in states}
        while iter < max_iter:
            delta = self.method.run(V_new)
            res.append(delta)
            if delta < theta:
                break
            iter += 1

        print(f'{self.method}: # of iterations: {len(res)}')

        plt.plot(np.arange(len(res)) + 1, res, label=str(self.method), alpha=0.3)
        plt.xlabel("Iteration")
        plt.ylabel("Delta")
        plt.legend()
