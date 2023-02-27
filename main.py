import ECAgent.Environments as ENV
import matplotlib.pyplot as plt
import numpy as np

from src.Gather import Gather, EnvResourceComponent
from src.Agents import *

ENV_SIZE = 50
NEST_SIZE = 4
NUM_AGENTS = 100

DEPOSIT_RATE = 0.25
DECAY_RATE = 0.1

ITERATIONS = 1000

VIS = True

def main():

    # Create Model
    model  = Gather(ENV_SIZE, NEST_SIZE, DEPOSIT_RATE, DECAY_RATE)

    # Add Agents to the environment
    for i in range(NUM_AGENTS):
        model.environment.add_agent(AntAgent(i, model), *model.random.choice(model.home_locs))

    # Run Model
    #model.execute(ITERATIONS)
    agent_val = len(model.resource_distribution) + 1
    for i in range(ITERATIONS):
        model.execute()

        print(f'Resources Collected: {model.environment[EnvResourceComponent].resources}')

        if VIS:
            # Will generate a series of figures of the environment.
            fig, ax = plt.subplots()
            img = np.copy(model.environment.cells[Gather.RESOURCE_KEY].to_numpy()).reshape(ENV_SIZE,ENV_SIZE)

            for agent in model.environment:
                x, y = agent[ENV.PositionComponent].xy()
                img[y][x] = agent_val

            ax.imshow(img, cmap='Set1')
            fig.savefig(f'./output/iteration_{i}.png')
            plt.close(fig)

    if VIS:
        fig, ax = plt.subplots()
        img = np.copy(model.environment.cells[PheromoneMovementSystem.FOOD_KEY].to_numpy()).reshape(ENV_SIZE,ENV_SIZE)
        ax.imshow(img)
        fig.savefig('./food_vis.png')
        plt.close(fig)

        fig, ax = plt.subplots()
        img = np.copy(model.environment.cells[PheromoneMovementSystem.HOME_KEY].to_numpy()).reshape(ENV_SIZE,ENV_SIZE)
        ax.imshow(img)
        fig.savefig('./home_vis.png')
        plt.close(fig)

if __name__ == '__main__':
    main()