
import ECAgent.Core as Core
from ECAgent.Environments import discreteGridPosToID, GridWorld, PositionComponent

import matplotlib.pyplot as plt

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

    default_resource_distribution = [0.7, 0.15, 0.1, 0.05]
    default_index_names = ['void', 'rock', 'iron', 'gold']

    def __init__(self, env_size : int, home_size : int, resource_distribution : list = None, index_names : list = None,
                 seed : int = None):
        """ Initializes class and creates GridWorld Environment of width AND height = size.
            Adds RESOURCES and HOME cell to the environment '"""
        super().__init__(seed = seed if seed is None else Gather.seed)
        self.environment = GridWorld(env_size, env_size, self)

        resource_distribution = Gather.default_resource_distribution if resource_distribution is None else resource_distribution
        index_names = Gather.default_index_names if index_names is None else index_names

        # Create summed weight distribution for resources
        # Note: This assumes weights sum to 1.0 (Will need to normalize values if that cannot safely be assumed)
        summed_weights = []
        total = 0.0
        for weight in resource_distribution:
            total += weight
            summed_weights.append(total)

        # Create base resources to that the active resources will reset too
        temp_ref = self
        def base_resource_generator(pos, cells):
            r = temp_ref.random.random()
            for i, v in enumerate(summed_weights):
                if r < v:
                    return i + 1
            return len(summed_weights)
        self.environment.addCellComponent('base_' + Gather.RESOURCE_KEY, base_resource_generator)

        # Add resources to environment
        def resource_generator(pos, cells):
            return cells['base_' + Gather.RESOURCE_KEY][discreteGridPosToID(pos[0], pos[1], env_size)]

        self.environment.addCellComponent(Gather.RESOURCE_KEY, resource_generator)

        # Create home (base) of size home_size x home_size
        home_x, home_y = self.random.randrange(home_size, env_size - home_size), self.random.randrange(home_size, env_size - home_size)
        col_loc = self.environment.cells.columns.get_loc(Gather.RESOURCE_KEY)
        for x in range(home_size):
            for y in range(home_size):
                self.environment.cells.iloc[discreteGridPosToID(home_x + x, home_y + y, env_size), col_loc] = 0

        # Create EnvResourceComponent to Store Environment's Resource variables
        self.environment.addComponent(EnvResourceComponent(self.environment, self, env_size, home_size, home_x, home_y,
                                                           resource_distribution, index_names))

        fig, ax = plt.subplots()
        ax.imshow(self.environment.cells[Gather.RESOURCE_KEY].to_numpy().reshape(env_size,env_size))
        fig.savefig('./test.png')
        plt.close(fig)

    def reset(self):
        """ Resets the environment. """
        self.environment.cells[Gather.RESOURCE_KEY] = self.environment.cells['base_' + Gather.RESOURCE_KEY].copy()

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