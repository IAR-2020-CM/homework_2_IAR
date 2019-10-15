import torch
import torch.nn.functional as F

from gridworld import GridWorld
from NeuralNetwork import MLP

from tqdm import trange

from matplotlib import pyplot as plt

def best_policy(Q):
    argmaxes = torch.argmax(Q, dim=2)
    return argmaxes

def e_greedy(game, episode, policy, Q, epsilon=0.3):
    argmaxes = torch.argmax(Q, dim=1)
    for state, _ in episode:
        state = int(state.item())
        for direction in range(len(policy[state])):
            policy[state][direction] = epsilon / game.action_space_size
            if direction == argmaxes[state]:
                policy[state][direction] += 1 - epsilon
    return policy

class Learning():

    def __init__(self, game: GridWorld, learning_rate=0.01, freeze_period=20):

        self.game = game

        self.freeze_period = freeze_period

        self.mlp = MLP()
        self.target_mlp = MLP()
        self.target_mlp.load_state_dict(self.mlp.state_dict())

        self.optimizer = torch.optim.SGD(self.mlp.parameters(),
                                         lr=learning_rate)


    def q_learning(self, N=1000):

        n_a = self.game.action_space_size
        n_s = self.game.state_space_size
        epsilon_greed = 0.01

        discount = 0.9
        max_t = 300

        rewards = torch.zeros((N, ))
        losses = torch.zeros((N, ))
        episode_duration = []

        for i in trange(N):
            s = torch.tensor(self.game.reset(), dtype=torch.float).unsqueeze(0)
            done = False
            for t in range(max_t):
                self.optimizer.zero_grad()
                q = self.mlp(s)
                a = (q.argmax(0)
                     if torch.rand((1, )) < epsilon_greed
                     else torch.randint(0, n_a, ()))

                s_prime, r, done = self.game.step(a)

                s_prime = torch.tensor(s_prime, dtype=torch.float).unsqueeze(0)
                target = (torch.tensor(0, dtype=torch.float)
                          if done
                          else discount * self.target_mlp(s_prime).max() + r)
                target = target.unsqueeze(0)
                loss = F.mse_loss(q[a].unsqueeze(0), target)
                losses[i] = loss
                loss.backward()
                s = s_prime
                rewards[i] += r
                self.optimizer.step()
                if done:
                    episode_duration.append(t)
                    break

            if i % self.freeze_period == self.freeze_period - 1:
                temp_state_dict = self.target_mlp.state_dict()
                self.target_mlp.load_state_dict(self.mlp.state_dict())
                self.mlp.load_state_dict(temp_state_dict)


            # update epsilon
            epsilon_greed = 0.2 + i / (N - 1) * 0.7

        return rewards, losses, episode_duration


if __name__ == "__main__":

    game = GridWorld(width=4, height=4)
    game.add_start(1, 1)
    game.add_goal(4, 4)
    learning = Learning(game, 0.001)

    rewards_ql, losses_ql, episode_len = learning.q_learning(10000)

    indices_ql = list(range(len(rewards_ql)))
    rewards_ql = [i.item() for i in rewards_ql.cumsum(0)]
    plt.figure(0)
    plt.plot(indices_ql, rewards_ql)
    plt.xlabel("Nombre d'itérations")
    plt.ylabel("Récompense cumulée")

    indices_ql = list(range(len(losses_ql)))
    losses_ql = [i.item() for i in losses_ql.cumsum(0)]
    plt.figure(1)
    plt.plot(indices_ql, losses_ql)
    plt.xlabel("Nombre d'itérations")
    plt.ylabel("Erreur cumulée")

    plt.show()
