import ECAgent.Core as Core
import ECAgent.Environments as ENV

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


class RandomMovementSystem(Core.System):
    """Dummy System for testing movement mechanics in GATHER."""

    def __init__(self, id: str, model: Core.Model):
        super().__init__(id, model)

    def execute(self):
        for agent in self.model.environment:
            move_dir = self.model.random.choice(MOVEMENT_MATRIX)
            self.model.environment.move(agent, *move_dir)


class PheromoneMovementSystem(Core.System):

    HOME_KEY = 'h_pheromones'
    FOOD_KEY = 'f_pheromones'

    def __init__(self, id: str, model: Core.Model, agent_random_chance : float = 0.05):
        super().__init__(id, model)
        self.agent_random_chance = agent_random_chance

        # Add Pheromone layers
        generator = ENV.ConstantGenerator(0.0)
        model.environment.addCellComponent(PheromoneMovementSystem.FOOD_KEY, generator)
        model.environment.addCellComponent(PheromoneMovementSystem.HOME_KEY, generator)

    def execute(self):

        # Get resources data
        fcells = self.model.environment.cells[PheromoneMovementSystem.FOOD_KEY].to_numpy()
        hcells = self.model.environment.cells[PheromoneMovementSystem.HOME_KEY].to_numpy()

        resource_cells = self.model.environment.cells[GATHER.Gather.RESOURCE_KEY]

        for agent in self.model.environment:

            # Agents can move up, down , left and right. This is equivalent to searching their neumann neighbourhood.
            candidate_cells = self.model.environment.get_neumann_neighbours(agent[ENV.PositionComponent], ret_type = tuple)
            # Agents can't move to their previous location
            if agent[ModeComponent].last_loc is not None and agent[ModeComponent].last_loc in candidate_cells:
                candidate_cells.remove(agent[ModeComponent].last_loc)

            cell_ids = [ENV.discrete_grid_pos_to_id(c[0], c[1], self.model.environment.width) for c in candidate_cells]

            if agent[ModeComponent].home:
                home_flag = sum([1 for c in cell_ids if resource_cells[c] == 0]) > 0
                food_flag = False
            else:
                home_flag = False
                food_flag = sum([1 for c in cell_ids if resource_cells[c] > 1]) > 0

            if home_flag:  # Found home
                new_pos = self.model.random.choice(
                    [c for i, c in enumerate(candidate_cells) if resource_cells[cell_ids[i]] == 0]
                )
            elif food_flag:  # Found food.
                new_pos = self.model.random.choice(
                    [c for i, c in enumerate(candidate_cells) if resource_cells[cell_ids[i]] > 1]
                )
            elif self.model.random.random() < self.agent_random_chance:  # Random Move
                new_pos = self.model.random.choice(candidate_cells)
            else:
                tcells = hcells if agent[ModeComponent].home else fcells
                pheromones = [tcells[c] for c in cell_ids]

                sum_p = sum(pheromones)
                if sum_p < 0.001:  # If there are no pheromones that allow the agent to make an informed choice.
                    new_pos = self.model.random.choice(candidate_cells)
                else:
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


            # Update Position
            agent[ModeComponent].last_loc = agent[ENV.PositionComponent].xyz()
            self.model.environment.move_to(agent, new_pos[0], new_pos[1])


class PheromoneDepositSystem(Core.System):

    def __init__(self, id : str, model : Core.Model, deposit_rate : float, decay_rate : float):
        super().__init__(id, model)

        self.deposit_rate = deposit_rate  # TODO: Could be interesting if the deposit rate was based on the agent's success
        self.decay_rate = 1.0 - decay_rate


    def execute(self):
        # Get resources data
        fcells = self.model.environment.cells[PheromoneMovementSystem.FOOD_KEY].to_numpy() * self.decay_rate
        hcells = self.model.environment.cells[PheromoneMovementSystem.HOME_KEY].to_numpy() * self.decay_rate
        resource_cells = self.model.environment.cells[GATHER.Gather.RESOURCE_KEY].to_numpy()

        fcells[fcells < 0.001] = 0.0
        hcells[hcells < 0.001] = 0.0

        for agent in self.model.environment:

            pos_id = ENV.discrete_grid_pos_to_id(agent[ENV.PositionComponent].x, agent[ENV.PositionComponent].y,
                                        self.model.environment.width)

            if agent[ModeComponent].home:  # If agent is looking for home.
                if resource_cells[pos_id] == 0:  # Nest cells have an id of 0
                    agent[ModeComponent].home = False  # Agent must now search for food again.
                    agent[ResourceComponent].wealth += agent[ResourceComponent].resources  # Increase Agent Wealth
                    self.model.environment[GATHER.EnvResourceComponent].resources += agent[ResourceComponent].resources  # Keep track of total resources carried
                    agent[ResourceComponent].resources = 0  # Reset carrying of resources

                fcells[pos_id] += self.deposit_rate  # Update food pheromone

            elif resource_cells[pos_id] > 1:  # Note: empty cells are assumed to have an id of 1.
                resource_cells[pos_id] = 1  # Empty the resources (1 is void).
                agent[ModeComponent].home = True
                agent[ResourceComponent].resources += 1
                # Change agent's last loc to its current location so it can turn around.
                agent[ModeComponent].last_loc = agent[ENV.PositionComponent].xyz()
                hcells[pos_id] += self.deposit_rate

            else:  # If agent is looking for food and didn't find any.
                hcells[pos_id] += self.deposit_rate

        #  Update the environment's cells
        self.model.environment.cells.update({
            PheromoneMovementSystem.FOOD_KEY : fcells,
            PheromoneMovementSystem.HOME_KEY: hcells,
            GATHER.Gather.RESOURCE_KEY : resource_cells})
