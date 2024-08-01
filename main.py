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

# file imports for organization
from ctf import CaptureTheFlagScenario
from qlearning import QNetwork, QLearningAgent

file_path = "Code/VSCode/Personal/captureTheFlagMARL/saves/"
const_ht = 100000
const_mil = 1000000
num_ht_episodes = 1
num_mil_episodes = 0
episodes = num_ht_episodes * const_ht + num_mil_episodes * const_mil
previous_episodes = 0

load_agent_file_path = f"agent_saves_{previous_episodes}/"
save_file_path = f"agent_saves_{episodes}/"

load_path = os.path.join(file_path, load_agent_file_path)
save_path = os.path.join(file_path, save_file_path)

scenario = CaptureTheFlagScenario()
env = vmas.make_env(
    scenario = scenario,
    num_envs = 1,
    device = 'cpu',
    continuous_actions = True,
    max_steps = 840,
    seed = None
)

def run_game(num_episodes, epsilon=0.5, ep_decay=0.99, gamma=0.99, learning_rate=0.001):
    scenario.make_world(batch_dim=1, device="cpu")
    # Initialize agents
    agents = [QLearningAgent(n_agents=len(env.world.agents), epsilon=epsilon, ep_decay=ep_decay, gamma=gamma, learning_rate=learning_rate) for _ in range(len(env.world.agents))]
    # Optionally load saved agent states
    # for i in range(len(env.world.agents)):
    #     path = os.path.join(load_path, i)
    #     agents[i].load(path)

    for episode in range(num_episodes):
        state = env.reset()  # Reset the environment at the start of each episode
        done = False
        while not done:
            states = []
            actions_list = []
            rewards = []
            # Collect observations and decide actions for each agent
            for i, agent in enumerate(env.world.agents):
                current_state = scenario.observation(agent)
                states.append(current_state)
                valid_actions = scenario.valid_actions(agent)
                action = agents[i].get_action(current_state, valid_actions)
                actions_list.append(action)
                scenario.move(agent, action)

            # Check for flag collisions and tags
            for i, agent1 in enumerate(env.world.agents):
                if "flag" not in agent1.name:
                    scenario.check_flag_collision(agent1)
                    print(agent.action.u)
                for j, agent2 in enumerate(env.world.agents):
                    if "flag" not in agent1.name and "flag" not in agent2.name:
                        if i != j and agent1.color != agent2.color:  # Ensure different agents and opposing teams
                            scenario.check_tag(agent1, agent2)

            # Step the environment with the collected actions
            next_state, reward, done, info = env.step([agent.action.u for agent in env.world.agents])

            # Update each agent based on the outcomes
            for i, agent in enumerate(env.world.agents):
                agents[i].update(states[i], actions_list[i], reward[i], next_state, done)
                rewards.append(reward[i])  # Collect individual rewards for logging

            # Prepare for the next iteration
            current_state = next_state

            # Decay epsilon after each episode for exploration/exploitation balance
            for agent in agents:
                agent.epsilon_decay()

            # Periodically update target network weights
            if episode % 300 == 0:
                for agent in agents:
                    agent.update_tnet()

            # Log results at the end of each episode
            print(scenario.is_done)
            print(f"Episode {episode} running. Reward array: {rewards}")
            print(f"Current state: {current_state}")
        print(f"Episode {episode} finished. Reward array: {rewards}")

    return agents

env.world.agents
agents = run_game(episodes)
for i, agent in enumerate(agents):
    path = os.path.join(save_path, i)
    with open(path) as rFile:
        agent[i].save(path)
