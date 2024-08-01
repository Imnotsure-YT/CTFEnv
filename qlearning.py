import vmas
import os
from vmas import make_env
from vmas.simulator.core import World, Agent, Sphere
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color
from vmas.simulator.sensors import Lidar
from vmas.simulator.rendering import Geom
import torch
from torch import nn, optim
import torch.nn.functional 
import numpy as np
import random
import time

class QNetwork(nn.Module):
    def __init__(self, n_agents):
        self.n_agents = n_agents
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(14, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class QLearningAgent:
    def __init__(self, n_agents, epsilon, ep_decay, gamma, learning_rate):
        self.p_network = QNetwork(n_agents)
        self.t_network = QNetwork(n_agents)
        self.optimizer = optim.Adam(self.p_network.parameters(), lr=learning_rate)
        self.epsilon = epsilon
        self.ep_decay = ep_decay
        self.gamma = gamma
        self.step = 0

    def get_action(self, state, valid_actions):
        if random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.p_network(state_tensor)
                valid_q_values = q_values[0][valid_actions]
                return valid_actions[torch.argmax(valid_q_values).item()]
        else:
            return random.choice(valid_actions)

    def update(self, state, action, reward, next_state, done):
        # Convert state to a tensor if it's not already one
        if not isinstance(state, torch.Tensor):
            state_tensor = torch.tensor(state, dtype=torch.float32)
        else:
            state_tensor = state.float()

        # Stack the list of next_state tensors into a single tensor
        if isinstance(next_state, list):
            next_state_tensor = torch.stack(next_state)
        else:
            next_state_tensor = next_state.float()

        action_tensor = torch.tensor([action], dtype=torch.long)
        reward_tensor = torch.tensor([reward], dtype=torch.float)
        done_tensor = torch.tensor([done], dtype=torch.float)

        # Forward pass through the primary network
        q_values = self.p_network(state_tensor.unsqueeze(0))
        next_q_values = self.t_network(next_state_tensor)

        # Select the Q-value for the action taken
        q_value = q_values.gather(1, action_tensor.unsqueeze(1))
        next_q_value = next_q_values.max(1)[0].unsqueeze(1)

        # Calculate the expected Q-value
        expected_q_value = reward_tensor + (1 - done_tensor) * self.gamma * next_q_value

        # Compute the loss
        loss = nn.MSELoss()(q_value, expected_q_value.detach())

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def update_tnet(self):
        self.t_network.load_state_dict(self.p_network.state_dict())

    def save(self, path):
        torch.save({
            'model_state_dict': self.p_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'gamma': self.gamma
        }, path)

    def load(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.p_network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint['epsilon']
            self.gamma = checkpoint['gamma']
        else:
            print(f"No saved model found at {path}")

    def epsilon_decay(self):
        self.epsilon = max(0.01, self.epsilon * self.ep_decay)