import ECAgent.Core as Core
import ECAgent.Environments as ENV
import numpy as np

import src.Gather as GATHER

MOVEMENT_MATRIX = [
    [0, 1], # UP
    [0, -1], # DOWN
    [-1, 0], # LEFT
    [1, 0] # RIGHT
]


class ResourceComponent(Core.Component):
    """ Resource Component Class responsible for keeping track of agent resources.
        # TODO Add Support for multiple resource types """
    def __init__(self, agent : Core.Agent, model: Core.Model):
        super().__init__(agent, model)
        self.wealth = 0
        self.resources = 0


class ModeComponent(Core.Component):
    """Mode Component Determines whether Agent will look at h or f pheromone when where it should move to."""
    def __init__(self, agent : Core.Agent, model: Core.Model):
        super().__init__(agent, model)
        self.home = False  # By default, agents look at the f pheromone.
        self.last_loc = None


class PheromoneComponent(Core.Component):
    def __init__(self, agent : Core.Agent, model: Core.Model):
        super().__init__(agent, model)
        self.f_pheromones = np.zeros(model.environment.width ** 2)
        self.h_pheromones = np.zeros(model.environment.width ** 2)


class PheromoneCommunicationComponent(Core.Component):
    def __init__(self, agent : Core.Agent, model: Core.Model, communication_network):
        super().__init__(agent, model)
        self.communication_network = communication_network


class BaseAgent(Core.Agent):
    """Base Class for all GATHER Agents.
        Class adds a ResourceComponent to the agent."""

    def __init__(self, id: str, model: Core.Model):
        super().__init__(id, model)
        self.add_component(ResourceComponent(self, model))


class AntAgent(Core.Agent):
    """GATHER Agent that uses pheromones to determine movement direction.
        Class adds a ResourceComponent to the agent."""

    def __init__(self, id: str, model: Core.Model):
        super().__init__(id, model)
        self.add_component(ResourceComponent(self, model))
        self.add_component(ModeComponent(self, model))
        self.add_component(PheromoneComponent(self, model))


class RandomMovementSystem(Core.System):
    """Dummy System for testing movement mechanics in GATHER."""

    def __init__(self, id: str, model: Core.Model):
        super().__init__(id, model)

    def execute(self):
        for agent in self.model.environment:
            move_dir = self.model.random.choice(MOVEMENT_MATRIX)
            self.model.environment.move(agent, *move_dir)


class CostSystem(Core.System):
    def __init__(self, id: str, model: Core.Model, cost, cost_frequency):
        super().__init__(id, model, frequency=cost_frequency)
        self.cost = cost

    def execute(self):
        for agent in self.model.environment:
            agent[ResourceComponent].wealth = max(agent[ResourceComponent].wealth - 1, 0)

        self.model.reset()


class PheromoneMovementSystem(Core.System):

    MOVE_MODE_PROBABILITY = 0
    MOVE_MODE_ARGMAX = 1

    def __init__(self, id: str, model: Core.Model, communication_network, agent_random_chance: float = 0.0,
                 move_mode: int = MOVE_MODE_PROBABILITY, detect: bool = False):
        super().__init__(id, model)
        self.agent_random_chance = agent_random_chance
        self.move_mode = move_mode
        self.model.environment.add_component(PheromoneCommunicationComponent(self.model.environment, self.model,
                                                                             communication_network))
        self.detect = detect

    def execute(self):

        resource_cells = self.model.environment.cells[GATHER.Gather.RESOURCE_KEY]
        for agent in self.model.environment:

            # Agents can move up, down , left and right. This is equivalent to searching their neumann neighbourhood.
            candidate_cells = self.model.environment.get_neumann_neighbours(agent[ENV.PositionComponent], ret_type = tuple)
            # Agents can't move to their previous location
            if agent[ModeComponent].last_loc is not None and agent[ModeComponent].last_loc in candidate_cells:
                candidate_cells.remove(agent[ModeComponent].last_loc)

            cell_ids = [ENV.discrete_grid_pos_to_id(c[0], c[1], self.model.environment.width) for c in candidate_cells]

            # Should agents take a random action if there are more than 3 agents, take a random action
            # Taken from Prevention of Ant Mills in Pheromone-Based Search Algorithm for Robot Swarms by Cheraghi et al.
            detect = len(self.model.environment.get_agents_at(*agent[ENV.PositionComponent].xy())
                         ) > 4 if self.detect else False

            if self.model.random.random() < self.agent_random_chance or detect:  # Random Move
                new_pos = self.model.random.choice(candidate_cells)
            else:
                # Calculate agent's pheromone representation
                fcells = np.zeros(self.model.environment.width ** 2)
                hcells = np.zeros(self.model.environment.width ** 2)
                for other in self.model.environment:
                    fcells += other[PheromoneComponent].f_pheromones * self.model.environment[
                        PheromoneCommunicationComponent].communication_network[agent.id][other.id]
                    hcells += other[PheromoneComponent].h_pheromones * self.model.environment[
                        PheromoneCommunicationComponent].communication_network[agent.id][other.id]

                tcells = hcells if agent[ModeComponent].home else fcells
                pheromones = [tcells[c] for c in cell_ids]

                sum_p = sum(pheromones)
                if sum_p < 0.001:  # If there are no pheromones that allow the agent to make an informed choice.
                    new_pos = self.model.random.choice(candidate_cells)
                elif self.move_mode == PheromoneMovementSystem.MOVE_MODE_PROBABILITY:
                    weights_p = []
                    total_p = 0.0
                    for weight in pheromones:
                        total_p += weight / sum_p
                        weights_p.append(total_p)

                    i_p = -1  # Index of selected pheromone
                    r = self.model.random.random()
                    for i, v in enumerate(weights_p):
                        if r < v:
                            i_p = i
                            break

                    new_pos = candidate_cells[i_p]
                else:  # Move mode argmax
                    new_pos = candidate_cells[np.argmax(pheromones)]

            # Update Position
            agent[ModeComponent].last_loc = agent[ENV.PositionComponent].xyz()
            self.model.environment.move_to(agent, new_pos[0], new_pos[1])


class PheromoneDepositSystem(Core.System):

    MOVE_MODE_PROBABILITY = 0
    MOVE_MODE_ARGMAX = 1

    def __init__(self, id: str, model: Core.Model, deposit_rate: float, hdecay_rate: float, fdecay_rate: float,
                 gamma: float = 0.99, eta: float = 0.1, move_mode: int = MOVE_MODE_PROBABILITY):
        super().__init__(id, model)

        self.deposit_rate = deposit_rate  # TODO: Could be interesting if the deposit rate was based on the agent's success
        self.hdecay_rate = 1.0 - hdecay_rate
        self.fdecay_rate = 1.0 - fdecay_rate
        self.gamma = gamma
        self.move_mode = move_mode
        self.eta = eta

    def execute(self):
        resource_cells = self.model.environment.cells[GATHER.Gather.RESOURCE_KEY].to_numpy()

        for agent in self.model.environment:

            agent[PheromoneComponent].f_pheromones *= self.fdecay_rate
            agent[PheromoneComponent].h_pheromones *= self.hdecay_rate

            agent[PheromoneComponent].f_pheromones[agent[PheromoneComponent].f_pheromones < 0.001] = 0.0
            agent[PheromoneComponent].h_pheromones[agent[PheromoneComponent].h_pheromones < 0.001] = 0.0

            pos_id = ENV.discrete_grid_pos_to_id(agent[ENV.PositionComponent].x, agent[ENV.PositionComponent].y,
                                        self.model.environment.width)
            last_loc_id = ENV.discrete_grid_pos_to_id(agent[ModeComponent].last_loc[0], agent[ModeComponent].last_loc[1],
                                        self.model.environment.width)

            if agent[ModeComponent].home:  # If agent is looking for home.
                reward = 0
                if resource_cells[pos_id] == 0:  # Nest cells have an id of 0
                    agent[ModeComponent].home = False  # Agent must now search for food again.
                    agent[ResourceComponent].wealth += agent[ResourceComponent].resources  # Increase Agent Wealth
                    self.model.environment[GATHER.EnvResourceComponent].resources += agent[ResourceComponent].resources  # Keep track of total resources carried
                    agent[ResourceComponent].resources = 0  # Reset carrying of resources
                    reward = 1

                # Update food pheromone
                if self.move_mode == PheromoneDepositSystem.MOVE_MODE_PROBABILITY:
                    agent[PheromoneComponent].f_pheromones[pos_id] += self.deposit_rate
                else:  # Update pheromones using 'Q-learning'
                    agent[PheromoneComponent].h_pheromones[last_loc_id] += self.eta * (
                        reward + self.gamma * (agent[PheromoneComponent].h_pheromones[pos_id] -
                                          agent[PheromoneComponent].h_pheromones[last_loc_id])
                    )
                    agent[PheromoneComponent].f_pheromones[pos_id] += self.eta * (
                            reward + self.gamma * (agent[PheromoneComponent].h_pheromones[last_loc_id] -
                                                   agent[PheromoneComponent].f_pheromones[pos_id])
                    )

            elif resource_cells[pos_id] > 1:  # Note: empty cells are assumed to have an id of 1.
                resource_cells[pos_id] = 1  # Empty the resources (1 is void).
                agent[ModeComponent].home = True
                agent[ResourceComponent].resources += 1

                if self.move_mode == PheromoneDepositSystem.MOVE_MODE_PROBABILITY:
                    agent[PheromoneComponent].h_pheromones[pos_id] += self.deposit_rate
                else:  # Update pheromones using 'Q-learning', reward is 1 because agent found food
                    agent[PheromoneComponent].f_pheromones[last_loc_id] += self.eta * (
                            1 + self.gamma * (agent[PheromoneComponent].f_pheromones[pos_id] -
                                              agent[PheromoneComponent].f_pheromones[last_loc_id])
                    )

                # Change agent's last loc to its current location so it can turn around.
                agent[ModeComponent].last_loc = agent[ENV.PositionComponent].xyz()

            else:  # If agent is looking for food and didn't find any.
                if self.move_mode == PheromoneDepositSystem.MOVE_MODE_PROBABILITY:
                    agent[PheromoneComponent].h_pheromones[pos_id] += self.deposit_rate
                else:  # Update pheromones using 'Q-learning'. Agent was and still is looking for food, no reward
                    agent[PheromoneComponent].f_pheromones[last_loc_id] += self.eta * (
                            self.gamma * (agent[PheromoneComponent].f_pheromones[pos_id] -
                                              agent[PheromoneComponent].f_pheromones[last_loc_id])
                    )
                    agent[PheromoneComponent].h_pheromones[pos_id] += self.eta * (
                            1 + self.gamma * (agent[PheromoneComponent].f_pheromones[last_loc_id] -
                                              agent[PheromoneComponent].h_pheromones[pos_id])
                    )  # Agents perform better when there is a constant signal detailing where 'home' is
