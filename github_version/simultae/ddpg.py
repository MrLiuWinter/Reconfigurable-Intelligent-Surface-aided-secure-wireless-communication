import copy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch


class ExperienceReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        index = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[index]).to(self.device),
            torch.FloatTensor(self.action[index]).to(self.device),
            torch.FloatTensor(self.next_state[index]).to(self.device),
            torch.FloatTensor(self.reward[index]).to(self.device),
            torch.FloatTensor(self.not_done[index]).to(self.device)
        )

    def __len__(self):
        ''' 返回当前存储的量
        '''
        return self.size


# Implementation of the Deep Deterministic Policy Gradient algorithm (DDPG)
# Paper: https://arxiv.org/abs/1509.02971

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        hidden_dim1 = 300
        hidden_dim2 = 200

        self.l1 = nn.Linear(state_dim, hidden_dim1)
        self.l2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.l3 = nn.Linear(hidden_dim2, action_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)

    def forward(self, state):
        # a = torch.tanh(self.l1(state.float()))
        # a = self.bn1(a)
        # a = torch.tanh(self.l2(a))
        # a = self.bn2(a)
        # a = torch.tanh(self.l3(a))

        a = F.relu(self.l1(state.float()))
        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a))

        return a


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        hidden_dim1 = 300
        hidden_dim2 = 200

        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim1)
        self.l2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.l3 = nn.Linear(hidden_dim2, 1)

        self.bn1 = nn.BatchNorm1d(hidden_dim1)

    def forward(self, state, action):
        # q =self.l3(torch.tanh(self.l2(torch.cat([self.bn1(torch.tanh(self.l1(state.float()))), action], 1))))
        q = torch.cat([state, action], 1)
        q = F.relu(self.l1(q))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q


class DDPG(object):
    def __init__(self, state_dim, action_dim, cfg):
        actor_lr = 1e-3
        critic_lr = 1e-4
        actor_decay = 1e-5
        critic_decay = 1e-5
        self.device = cfg.device
        # Initialize the discount and target update rated
        self.discount = 0.99
        self.tau = 0.001

        # Initialize actor networks and optimizer
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=actor_decay)

        # Initialize critic networks and optimizer
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=critic_decay)

        self.replay_buffer = ExperienceReplayBuffer(state_dim, action_dim, cfg.memory_capacity)

    def select_action(self, state):
        self.actor.eval()
        state = torch.FloatTensor(state).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten().reshape(1, -1)

        return action

    def update_parameters(self, batch_size):
        # if len(self.replay_buffer) < batch_size:
        #     return 0, 0
        self.actor.train()

        # Sample from the experience replay buffer
        state, action, next_state, reward, not_done = self.replay_buffer.sample(batch_size)

        # Get the current Q-value estimate
        current_Q = self.critic(state, action)

        # Compute the target Q-value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * self.discount * target_Q).detach()

        # Compute the critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute the actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update the target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # return critic_loss.cpu().detach().numpy().item(), actor_loss.cpu().detach().numpy().item()

    # Save the model parameters
    def save(self, file_name):
        torch.save(self.critic.state_dict(), file_name + "_critic")
        torch.save(self.critic_optimizer.state_dict(), file_name + "_critic_optimizer")

        torch.save(self.actor.state_dict(), file_name + "_actor")
        torch.save(self.actor_optimizer.state_dict(), file_name + "_actor_optimizer")

    # Load the model parameters
    def load(self, file_name):
        self.critic.load_state_dict(torch.load(file_name + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(file_name + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(file_name + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(file_name + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
