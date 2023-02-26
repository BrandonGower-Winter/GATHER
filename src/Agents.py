import ECAgent.Core as Core

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
        self.resources = 0

class BaseAgent(Core.Agent):
    """ Base Class for all GATHER Agents.
        Class adds a ResourceComponent to the agent. """

    def __init__(self, id: str, model: Core.Model):
        super().__init__(id, model)
        self.addComponent(ResourceComponent(self, model))

class RandomMovementSystem(Core.System):
    """ Dummy System for testing movement mechanics in GATHER."""

    def __init__(self, id: str, model: Core.Model):
        super().__init__(id, model)

    def execute(self):
        for agent in self.model.environment:
            move_dir = self.model.random.choice(MOVEMENT_MATRIX)
            self.model.environment.move(agent, *move_dir)
