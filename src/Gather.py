
import ECAgent.Core as Core
from ECAgent.Environments import discrete_grid_pos_to_id, GridWorld

import src.Agents as Agents

import matplotlib.pyplot as plt
import numpy as np


""" 1. Environment
    2. Movement 
    3. Consumption 
    4. Reproduction 
    5. Information Sharing
    6. Loggers"""

class Gather(Core.Model):

    """ Model Class of GATHER ABM.
        Abiding by the ECS design paradigm, the class is just a container for the systems and agents that will
        execute / occupy a simulation. """

    # Resource Distribution Enums
    # Scenarios Taken from Hecker et al.'s Evolving Error Tolerance in Biologically-Inspired iAnt Robots (2013)
    ENV_RANDOM = 0
    ENV_CLUSTERED = 1
    ENV_POWER = 2

    # Env Layer Keys
    RESOURCE_KEY = 'resources'
    HOME_KEY = 'ishome'

    seed = None

    default_resource_distribution = [0.0, 0.7, 0.2, 0.1]
    default_index_names = ['void', 'rock', 'iron', 'gold']

    def __init__(self, env_size : int, home_size : int, deposit_rate : float, decay_rate : float,
                 resource_distribution : list = None, index_names : list = None, seed : int = None):
        """ Initializes class and creates GridWorld Environment of width AND height = size.
            Adds RESOURCES and HOME cell to the environment '"""
        super().__init__(seed = seed if seed is None else Gather.seed)
        self.environment = GridWorld(self, env_size, env_size)

        self.resource_distribution = Gather.default_resource_distribution if resource_distribution is None else resource_distribution
        self.index_names = Gather.default_index_names if index_names is None else index_names

        # Create summed weight distribution for resources
        # Note: This assumes weights sum to 1.0 (Will need to normalize values if that cannot safely be assumed)
        summed_weights = []
        total = 0.0
        for weight in self.resource_distribution:
            total += weight
            summed_weights.append(total)

        # Create base resources to that the active resources will reset to
        temp_ref = self
        def base_resource_generator(pos, cells):
            r = temp_ref.random.random()
            for i, v in enumerate(summed_weights):
                if r < v:
                    return i + 1
            return len(summed_weights)

        self.environment.add_cell_component('base_' + Gather.RESOURCE_KEY, base_resource_generator)

        # Add resources to environment
        self.environment.add_cell_component(Gather.RESOURCE_KEY,
                            np.copy(self.environment.cells['base_' + Gather.RESOURCE_KEY].to_numpy()))

        # Create home (base) of size home_size x home_size
        self.home_locs = []
        home_x, home_y = self.random.randrange(home_size, env_size - home_size), self.random.randrange(home_size, env_size - home_size)
        col_loc = self.environment.cells.columns.get_loc(Gather.RESOURCE_KEY)
        for x in range(home_size):
            for y in range(home_size):
                self.home_locs.append((home_x + x, home_y + y))
                self.environment.cells.iloc[discrete_grid_pos_to_id(*self.home_locs[-1], env_size), col_loc] = 0


        # Create EnvResourceComponent to Store Environment's Resource variables
        self.environment.add_component(EnvResourceComponent(self.environment, self, env_size, home_size, home_x, home_y,
                                                           resource_distribution, index_names))

        # Add Systems
        # self.systems.add_system(Agents.RandomMovementSystem('RMS', self))
        self.systems.add_system(Agents.PheromoneMovementSystem('PMS', self))
        self.systems.add_system(Agents.PheromoneDepositSystem('PDS', self, deposit_rate, decay_rate))

        # Will generate a figure of the environment.
        # fig, ax = plt.subplots()
        # ax.imshow(self.environment.cells[Gather.RESOURCE_KEY].to_numpy().reshape(env_size,env_size))
        # fig.savefig('./test.png')
        # plt.close(fig)

    def reset(self):
        """ Resets the environment. """
        self.environment.cells[Gather.RESOURCE_KEY] = self.environment.cells['base_' + Gather.RESOURCE_KEY].copy()

        # Reset Agent Positions
        for agent in self.environment:
            self.environment.move_to(agent, *(self.random.choice(self.home_locs)))

class EnvResourceComponent(Core.Component):
    """ Class that stores data related to the Gather.RESOURCE_KEY layer
        Variables:
        env_size -> The size of the environment along one axis. The environment's full size is env_size x env_size.
        home_size -> The size of the homebase along one axis. The homebase's full size is home_size x home_size.
        home -> two-tuple that stores the coordinates for the top-left most cell of the homebase.
        resource_distribution -> list that stores the approximate distribution of resources scattered across the environment.
            Each value stores the probability of that type of resource occurring. All values in the array should sum to 1.0.
        index_names -> list that stores the names of the resource types. The index of the resource corresponds to its probability
            in the resource_distribution variable.
        """
    def __init__(self, agent : Core.Agent, model: Core.Model, env_size : int, home_size : int, home_x : int, home_y : int,
                 resource_distribution : list, index_names : list):

        super().__init__(agent, model)
        self.env_size = env_size
        self.home_size = home_size
        self.home = (home_x, home_y)
        self.resource_distribution = resource_distribution
        self.index_names = index_names
        self.resources = 0