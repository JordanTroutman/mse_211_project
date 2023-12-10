from abc import abstractmethod
import random

class IValueIteration:
    def __init__(self, game, sample_rate=1):
        self.game = game
        self.sample_rate = sample_rate

    @abstractmethod
    def run(self, V_new):
        raise NotImplemented

class ClassicValueIteration(IValueIteration):
    def run(self, V_new):
        delta = 0
        for s in self.game.states:
            max_val = 0
            for a in self.game.actions:
                val = self.game.rewards[s]
                for s_next in self.game.states:
                    val += self.game.probs[s][s_next][a] * (self.game.gamma * self.game.V[s_next])
                max_val = max(max_val, val)
            V_new[s] = max_val
            delta = max(delta, abs(self.game.V[s] - V_new[s]))
        self.game.V = V_new
        return delta

class RandomValueIteration(IValueIteration):
    def run(self, V_new):
        delta = 0
        # rather than go through all state values in each iteration
        # randomly select a subset of states
        sampled_states = random.sample(self.game.states, int(self.sample_rate * len(self.game.states)))
        for s in sampled_states:
            max_val = 0
            for a in self.game.actions:
                val = self.game.rewards[s]
                for s_next in sampled_states:
                    val += self.game.probs[s][s_next][a] * (self.game.gamma * self.game.V[s_next])
                max_val = max(max_val, val)
            V_new[s] = max_val
            delta = max(delta, abs(self.game.V[s] - V_new[s]))
        V = V_new
        return delta

class CyclicValueIteration(IValueIteration):
    def run(self, V_new):
        delta = 0
        for s in self.game.states:
            max_val = 0
            v = self.game.V[s]
            for a in self.game.actions:
                val = self.game.rewards[s]
                for s_next in self.game.states:
                    val += self.game.probs[s][s_next][a] * (self.game.gamma * self.game.V[s_next])
                max_val = max(max_val, val)
            self.game.V[s] = max_val
            delta = max(delta, abs(v - self.game.V[s]))
        return delta

class RandomCyclicValueIteration(IValueIteration):
    def run(self, V_new):
        delta = 0
        # rather than go through all state values in each iteration
        # randomly select a subset of states
        sampled_states = random.sample(self.game.states, int(self.sample_rate * len(self.game.states)))
        for s in sampled_states:
            max_val = 0
            v = self.game.V[s]
            for a in self.game.actions:
                val = self.game.rewards[s]
                for s_next in sampled_states:
                    val += self.game.probs[s][s_next][a] * (self.game.gamma * self.game.V[s_next])
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
        V = [0, 0, 0, 0, 0]
        while iter < max_iter:
            V_new = [0, 0, 0, 0, 0]
            delta = self.method.run(V_new)
            V = V_new
            res.append(delta)
            if delta < theta:
                break
            iter += 1

        print(f'# of iterations: {len(res)}')
        fig, ax = plt.subplots(1, 1, figsize=(3, 2), dpi=200)
        ax.plot(np.arange(len(res)) + 1, res, marker='o', markersize=4,
                alpha=0.7, color='#2ca02c', label=r'$\theta= $' + "{:.2E}".format(theta))
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Delta')
        ax.legend()
        plt.tight_layout()
        plt.show()

        return res

# m = ClassicValueIteration(SimpleGame())
m = RandomValueIteration(SimpleGame(), sample_rate=0.9)
# m = CyclicValueIteration(SimpleGame())
# m = RandomCyclicValueIteration(SimpleGame(), sample_rate=0.8)

res = VISimulation(m).simulate(10000, 1e-100)