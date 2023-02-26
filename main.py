import ECAgent.Environments as ENV
import matplotlib.pyplot as plt
import numpy as np

from src.Gather import Gather
from src.Agents import BaseAgent

ENV_SIZE = 50
NEST_SIZE = 4
NUM_AGENTS = 100

ITERATIONS = 100


def main():

    # Create Model
    model  = Gather(ENV_SIZE, NEST_SIZE)

    # Add Agents to the environment
    for i in range(NUM_AGENTS):
        choice = model.random.choice(model.home_locs)
        model.environment.add_agent(BaseAgent(i, model), choice[0], choice[1])

    # Run Model
    #model.execute(ITERATIONS)
    agent_val = len(model.resource_distribution) + 1
    for i in range(ITERATIONS):
        model.execute()

        # Will generate a series of figures of the environment.
        fig, ax = plt.subplots()
        img = np.copy(model.environment.cells[Gather.RESOURCE_KEY].to_numpy()).reshape(ENV_SIZE,ENV_SIZE)

        for agent in model.environment:
            x, y = agent[ENV.PositionComponent].xy()
            img[y][x] = agent_val

        ax.imshow(img, cmap='Set1')
        fig.savefig(f'./output/iteration_{i}.png')
        plt.close(fig)


if __name__ == '__main__':
    main()