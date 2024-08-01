from vmas.simulator.core import Dynamics

class CustomDynamics(Dynamics):
    @property
    def needed_action_size(self):
        return 2
    def process_action(self):
        # Convert the action tensor to a NumPy array
        action = self.agent.action.u.cpu().numpy().squeeze()
        # Update the agent's position based on the action
        self.agent.state.p_pos += action

    