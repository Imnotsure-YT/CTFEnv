import vmas
import os
from vmas import make_env
from vmas.simulator.core import World, Agent, Sphere, Dynamics
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

from custom_dynamics import CustomDynamics

class CaptureTheFlagScenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.n_agents = 2
        self.viewer_zoom = 1
        self.agent_radius = 0.05
        self.flag_radius = 0.03
        self.max_steps = 840
        self.blue_flag_captured = False
        self.red_flag_captured = False
        self.has_flag = [False for _ in range(self.n_agents)]

        # Define the world
        world = World(
            batch_dim=batch_dim,
            device=device,
            x_semidim=2,
            y_semidim=1,
            drag=0.25,
        )

        world.agents.clear()

        # Spawn the n agents and add them to the world
        for i in range(self.n_agents):
            agent = Agent(
                name=f"agent_{i}",
                color=Color.BLUE if i % 2 == 0 else Color.RED,
                collide=False,
                shape=Sphere(radius=self.agent_radius),
                u_range=1.0,
                dynamics=CustomDynamics(),
                action_size=2
            )
            if i % 2 == 0:
                # Agent is blue, spawn on the left side
                agent.state.p_pos = np.array([np.random.uniform(-1.8, 0), np.random.uniform(-1, 1)])
            else:
                # Agent is red, spawn on the right side
                agent.state.p_pos = np.array([np.random.uniform(1.8, 0), np.random.uniform(-1, 1)])
            
            world.add_agent(agent)  # Properly add agent to the world

        # Create and position flags
        self.blue_flag = Agent(
            name="blue_flag",
            color=Color.BLUE,
            collide=False,
            shape=Sphere(radius=self.flag_radius)
        )
        self.red_flag = Agent(
            name="red_flag",
            color=Color.RED,
            collide=False,
            shape=Sphere(radius=self.flag_radius)
        )
        self.blue_flag.state.p_pos = np.array([-1.9, 0])
        self.red_flag.state.p_pos = np.array([1.9, 0])
        world.add_agent(self.blue_flag)
        world.add_agent(self.red_flag)

        return world


    def reset_world(self, world):
        self.blue_flag_captured = False
        self.red_flag_captured = False
        self.has_flag = [False for _ in range(self.n_agents)]
        # Reset agent positions
        for i, agent in enumerate(world.agents):
            # "spawn" each agent randomly
            if i % 2 == 0:
                # blue agent, spawn on left
                agent.state.p_pos = np.array([np.random.uniform(-1.8, 0), np.random.uniform(-1, 1)])
            else:
                # red agent, spawn on right
                agent.state.p_pos = np.array([np.random.uniform(1.8, 0), np.random.uniform(-1, 1)])

        # reset flags
        self.blue_flag.state.p_pos = np.array([-1.9, 0])
        self.red_flag.state.p_pos = np.array([1.9, 0])

    def reset_world_at(self, env_index=None):
        # for the sake of this version, reset_world_at is not useful since we are not using parallel training.
        # reset_world_at is required for the code.
        return 0

    def check_win(self, team):
        if team == "blue" and self.flag_red.state.p_pos[0] < 0:
            self.red_flag_captured = True
        elif team == "red" and self.flag_blue.state.p_pos[0] > 0:
            self.blue_flag_captured = True

    def is_safe(self, player):
        if self.has_flag[int(player.name[6])]:
            return False
        if player.color == "blue":
            return player.state.p_pos[0] < 0 or player.state.p_pos[0] > 1.8
        elif player.color == "red":
            return player.state.p_pos[0] > 0 or player.state.p_pos[0] < -1.8
        
    def check_tag(self, tagger, taggee):
        # we assume that this function is only called when tagger and taggee are on opposing teams.
        side = tagger.color
        if side == "blue" and tagger.state.p_pos[0] > 0:
            if np.linalg.norm(tagger.state.p_pos - taggee.state.p_pos) < 0.002 and not self.is_safe(taggee):
                self.has_flag[int([taggee.name[6]])] = False
                self.blue_flag.state.p_pos = np.array([-1.9, 0])
                self.resolve_tagged(taggee)
                return True

        elif side == "red" and tagger.state.p_pos[0] < 0:
            if np.linalg.norm(tagger.state.p_pos - taggee.state.p_pos) < 0.002 and not self.is_safe(taggee):
                self.has_flag[int([taggee.name[6]])] = False
                self.red_flag.state.p_pos = np.array([1.9, 0])
                self.resolve_tagged(taggee)
                return True

    def observation(self, agent):
        # Ensure all positions are converted to PyTorch tensors
        all_agents_positions = [torch.tensor(other_agent.state.p_pos, dtype=torch.float32) for other_agent in self.world.agents]
        blue_flag_position = torch.tensor(self.blue_flag.state.p_pos, dtype=torch.float32)
        red_flag_position = torch.tensor(self.red_flag.state.p_pos, dtype=torch.float32)

        # Combine the positions into a single tensor
        # Use unsqueeze to add a batch dimension and then flatten
        all_positions = torch.cat([pos.unsqueeze(0) for pos in all_agents_positions], dim=0).flatten()
        flags_positions = torch.cat([blue_flag_position.unsqueeze(0), red_flag_position.unsqueeze(0)], dim=0).flatten()
        positions = torch.cat([all_positions, flags_positions], dim=0)

        # Combine the agent's state (position and velocity) with the positions tensor
        agent_state_pos = torch.tensor(agent.state.p_pos, dtype=torch.float32) if not isinstance(agent.state.p_pos, torch.Tensor) else agent.state.p_pos
        return torch.cat([agent_state_pos, positions], dim=0)


    def reward(self, agent):
        # Initialize reward
        reward = -0.1

        # Check if the game is done
        if self.is_done(self.world):
            # Determine rewards based on flag capture status
            if agent.color == "blue":
                if self.red_flag_captured:
                    reward = 1  # Reward for capturing the opposing team's flag
                else:
                    reward = -1  # Penalty if the opposing team's flag was not captured
            elif agent.color == "red":
                if self.blue_flag_captured:
                    reward = 1  # Reward for capturing the opposing team's flag
                else:
                    reward = -1  # Penalty if the opposing team's flag was not captured

            # Check for a draw
        reward_tensor = torch.tensor(reward, dtype=torch.float32)
        print(f"Agent {agent.name} reward: {reward_tensor}")
        return reward_tensor

    def resolve_tagged(self, agent):
        # Move tagged agent to specified location
        agent.state.p_pos = np.array([0, 1.4])

    def check_flag_collision(self, agent):
        blue_distance = np.linalg.norm(agent.state.p_pos - self.blue_flag.state.p_pos)
        red_distance = np.linalg.norm(agent.state.p_pos - self.red_flag.state.p_pos)

        if blue_distance < (self.agent_radius + self.flag_radius):
            self.has_flag[int(agent.name[6])] = True
            self.blue_flag.state.p_pos = agent.state.p_pos  # Attach the flag to the agent

        if red_distance < (self.agent_radius + self.flag_radius):
            self.has_flag[int(agent.name[6])] = True
            self.red_flag.state.p_pos = agent.state.p_pos  # Attach the flag to the agent

    def is_done(self, world):
        if self.blue_flag_captured or self.red_flag_captured:
            return True
        return False
    
    def valid_actions(self, agent):
        actions = [0]  # Default action: do nothing

        # Check movement in the upward direction
        if agent.state.p_pos[1] + 0.001 <= 1:
            actions.append(1)  # Move up

        # Check movement in the downward direction
        if agent.state.p_pos[1] - 0.001 >= -1:
            actions.append(2)  # Move down

        # Check movement in the left direction
        if agent.state.p_pos[0] - 0.001 >= -1.8:
            actions.append(3)  # Move left

        # Check movement in the right direction
        if agent.state.p_pos[0] + 0.001 <= 1.8:
            actions.append(4)  # Move right

        return actions

    def resolve_action_index(self, action_index):
        actions = {
            0: [0, 0],
            1: [0, 1],
            2: [0, -1],
            3: [-1, 0],
            4: [1, 0]
        }
        return actions[action_index]

    def move(self, agent, action_index):
        move_vector = self.resolve_action_index(action_index)
        action_tensor = torch.tensor(move_vector, dtype=torch.float32, device="cpu").unsqueeze(0) * 0.04  # Unsqueeze to add batch dimension
        agent.action.u = action_tensor
        agent.dynamics.process_action()
    
    def get_action(agent, continuous, env, random_action):
        if not random_action:
            if continuous:
                action = -agent.action.u_range_tensor.expand(env.batch_dim, agent.action_size)
            else:
                action = torch.tensor([1], device=env.device, dtype=torch.long).expand(env.batch_dim, 1)
        else:
            action = env.get_random_action(agent)
        return action.clone()

scenario = CaptureTheFlagScenario()

env = vmas.make_env(
    scenario = scenario,
    num_envs = 1,
    device = 'cpu',
    continuous_actions = True,
    max_steps = 840,
    seed = None
)

