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


    #def monte_carlo(self, batch_size=32, N=1000):

    #    for i in range(N):
    #        data = gen_episode(self.game)
    #        label = torch.full(data.shape, -1)
    #        label[-1] = 9
    #        dataset = torch.utils.data.TensorDataset(data, label)

    #        loader = torch.utils.data.DataLoader(dataset,
    #                                             shuffle=True,
    #                                             batch_size=batch_size)
    #        total_loss = 0
    #        for i, (state, action) in enumerate(loader):
    #            out = mlp(state)


    def q_learning(self, N=1000, batch_size=32):

        n_a = self.game.action_space_size
        n_s = self.game.state_space_size
        epsilon_greed = 0.01

        discount = 0.9
        max_t = 300

        cumulated_reward = torch.zeros((N, ))

        for i in trange(N):
            s = torch.tensor(self.game.reset(), dtype=torch.float).unsqueeze(0)
            done = False
            for t in range(max_t):
                self.optimizer.zero_grad()
                q = self.mlp(s)
                # TODO: look if epsilon_greed is stil needed
                breakpoint()
                a = (q.argmax(1).item()
                     if torch.rand((1, )).item() < epsilon_greed
                     else torch.randint(0, n_a, (1,)).item())

                s_prime, r, done = self.game.step(a)

                s_prime = torch.tensor(s_prime, dtype=torch.float).unsqueeze(0)
                target = (torch.tensor(0, dtype=torch.float)
                          if done
                          else self.target_mlp(s_prime).max() + r)
                target = target.unsqueeze(0)
                loss = F.mse_loss(q[a].unsqueeze(0), target)
                loss.backward()
                s = s_prime
                cumulated_reward[i] += r
                self.optimizer.step()
                if done:
                    break

            if i % self.freeze_period == self.freeze_period - 1:
                temp_state_dict = self.target_mlp.state_dict()
                self.target_mlp.load_state_dict(self.mlp.state_dict())
                self.mlp.load_state_dict(temp_state_dict)


            # update epsilon
            epsilon_greed = 0.2 + i / (N - 1) * 0.8

        return cumulated_reward


if __name__ == "__main__":

    game = GridWorld()
    game.add_start(1, 1)
    game.add_goal(9, 9)
    learning = Learning(game)

    cumulated_reward_ql = learning.q_learning(500)

    indices_ql = list(range(len(cumulated_reward_ql)))
    cumulated_reward_ql = [i.item() for i in cumulated_reward_ql]
    plt.figure(2)
    plt.plot(indices_ql, cumulated_reward_ql)
    plt.xlabel("Nombre d'itérations")
    plt.ylabel("Récompense")

    plt.show()
