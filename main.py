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

COST = 0
COST_FREQUENCY = 100000

VIS = True

ENV_MODE = 0
NETWORK_MODE = 6

def get_communication_network():
    if NETWORK_MODE == 0:
        print('MODE: Zero Communication')
        return np.zeros((NUM_AGENTS, NUM_AGENTS))
    elif NETWORK_MODE == 1:
        print('MODE: Full Communication')
        return np.ones((NUM_AGENTS, NUM_AGENTS))
    elif NETWORK_MODE == 2:
        print('MODE: Self Communication')
        net = np.zeros((NUM_AGENTS, NUM_AGENTS))
        for i in range(NUM_AGENTS):
            net[i][i] = 1.0
        return net
    elif NETWORK_MODE == 3:
        print('MODE: Ring Communication')
        net = np.zeros((NUM_AGENTS, NUM_AGENTS))
        for i in range(NUM_AGENTS):
            net[i][i] = 1.0
            net[i][i-1] = 1.0
            net[i][(i+1) % NUM_AGENTS] = 1.0
        return net
    elif NETWORK_MODE == 4:
        print('MODE: Authoritarian Communication')
        net = np.zeros((NUM_AGENTS, NUM_AGENTS))
        for i in range(NUM_AGENTS):
            if i < 0.1 * NUM_AGENTS:
                net[i] = 1.0
            else:
                net[i][i] = 1.0
        return net
    elif NETWORK_MODE == 5:
        print('MODE: Other Communication')
        net = np.ones((NUM_AGENTS, NUM_AGENTS))
        for i in range(NUM_AGENTS):
            net[i][i] = 0.0
        return net
    elif NETWORK_MODE == 6:
        print('MODE: Islands Communication')
        net = np.zeros((NUM_AGENTS, NUM_AGENTS))
        modulo = int(NUM_AGENTS // 5)
        for i in range(NUM_AGENTS):
            for j in range(NUM_AGENTS):
                if i % modulo == j % modulo:
                    net[i][j] = 1.0
        return net
    return None


def gini(data):
    sorted_x = np.sort(data)
    n = len(sorted_x)
    cumx = np.cumsum(sorted_x, dtype=float)
    # The above formula, with all weights equal to 1 simplifies to:
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n if cumx[-1] > 0.0 else 0.0


def main():
    communication_network  = get_communication_network()

    # Create Model
    model  = Gather(ENV_SIZE, NEST_SIZE, DEPOSIT_RATE, DECAY_RATE, communication_network, COST, COST_FREQUENCY)

    # Add Agents to the environment
    for i in range(NUM_AGENTS):
        model.environment.add_agent(AntAgent(i, model), *model.random.choice(model.home_locs))

    # Run Model
    #model.execute(ITERATIONS)
    agent_val = len(model.resource_distribution) + 1
    for i in range(ITERATIONS):
        model.execute()

        wealth_arr = np.array([agent[ResourceComponent].wealth for agent in model.environment])
        print(f'Resources Collected: {model.environment[EnvResourceComponent].resources} Gini: {gini(wealth_arr)}')
        if VIS:
            # Will generate a series of figures of the environment.
            fig, ax = plt.subplots()
            img = np.copy(model.environment.cells[Gather.RESOURCE_KEY].to_numpy()).reshape(ENV_SIZE,ENV_SIZE)

            for agent in model.environment:
                x, y = agent[ENV.PositionComponent].xy()
                if agent[ModeComponent].home:
                    img[y][x] = agent_val
                else:
                    img[y][x] = agent_val + 1

            ax.imshow(img, cmap='Set1')
            fig.savefig(f'./output/iteration_{i}.png')
            plt.close(fig)

    if VIS:
        fcells = np.zeros(model.environment.width ** 2)
        hcells = np.zeros(model.environment.width ** 2)

        for agent in model.environment:
            fcells += agent[PheromoneComponent].f_pheromones
            hcells += agent[PheromoneComponent].h_pheromones

        fig, ax = plt.subplots()
        img = fcells.reshape(ENV_SIZE,ENV_SIZE)
        ax.imshow(img)
        fig.savefig('./food_vis.png')
        plt.close(fig)

        fig, ax = plt.subplots()
        img = hcells.reshape(ENV_SIZE,ENV_SIZE)
        ax.imshow(img)
        fig.savefig('./home_vis.png')
        plt.close(fig)

if __name__ == '__main__':
    main()