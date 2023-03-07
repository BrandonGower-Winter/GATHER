import argparse
import ECAgent.Environments as ENV
import matplotlib.pyplot as plt
import numpy as np
import sys

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

VIS = False

ENV_MODE = 0
NETWORK_MODE = 0

def get_communication_network(mode : int):
    if mode == 0:
        print('MODE: Zero Communication')
        return np.zeros((NUM_AGENTS, NUM_AGENTS))
    elif mode == 1:
        print('MODE: Full Communication')
        return np.ones((NUM_AGENTS, NUM_AGENTS))
    elif mode == 2:
        print('MODE: Self Communication')
        net = np.zeros((NUM_AGENTS, NUM_AGENTS))
        for i in range(NUM_AGENTS):
            net[i][i] = 1.0
        return net
    elif mode == 3:
        print('MODE: Ring Communication')
        net = np.zeros((NUM_AGENTS, NUM_AGENTS))
        for i in range(NUM_AGENTS):
            net[i][i] = 1.0
            net[i][i-1] = 1.0
            net[i][(i+1) % NUM_AGENTS] = 1.0
        return net
    elif mode == 4:
        print('MODE: Authoritarian Communication')
        net = np.zeros((NUM_AGENTS, NUM_AGENTS))
        for i in range(NUM_AGENTS):
            if i < 0.1 * NUM_AGENTS:
                net[i] = 1.0
            else:
                net[i][i] = 1.0
        return net
    elif mode == 5:
        print('MODE: Other Communication')
        net = np.ones((NUM_AGENTS, NUM_AGENTS))
        for i in range(NUM_AGENTS):
            net[i][i] = 0.0
        return net
    elif mode == 6:
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


def parseArgs():
    """Create GATHER Parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--size', help='The size of the environment. In all instances the environment will be a square gridworld.',
                        default=ENV_SIZE, type=int)
    parser.add_argument('-n', '--nest', help='The size of the nest. In all instances the nest will be square.',
                        default=NEST_SIZE, type=int)
    parser.add_argument('-a', '--agents', help='The number of agents to initialize.', default=NUM_AGENTS, type=int)
    parser.add_argument('--deposit', help='The amount of pheromone dropped by the agents.', default=DEPOSIT_RATE, type=float)
    parser.add_argument('--decay', help='The rate at which the pheromone evaporates.', default=DECAY_RATE, type=float)
    parser.add_argument('-i', '--iterations', help='How long the simulation should run for.',
                        default=ITERATIONS, type=int)
    parser.add_argument('--mode', help='What mode the environment should be initialized to', default=ENV_MODE, type=int)
    parser.add_argument('--network', help='The type of network to initialize', default=NETWORK_MODE, type=int)
    parser.add_argument('-v', '--visualize', help='Whether environment images should be written to the output/ directory',
                        action='store_true')
    parser.add_argument('--seed', help="Specify the seed for the Model's pseudorandom number generator", default=None,
                        type=int)

    return parser.parse_args()


def main():

    args = parseArgs()
    communication_network  = get_communication_network(args.network)

    # Create Model
    model  = Gather(args.size, args.nest, args.deposit, args.decay, communication_network, COST, COST_FREQUENCY,
                    args.mode, seed = args.seed)

    # Add Agents to the environment
    for i in range(args.agents):
        model.environment.add_agent(AntAgent(i, model), *model.random.choice(model.home_locs))

    # Run Model
    #model.execute(ITERATIONS)
    agent_val = len(model.resource_distribution) + 1
    for i in range(args.iterations):
        model.execute()
        wealth_arr = np.array([agent[ResourceComponent].wealth for agent in model.environment])
        print(f'\r{i+1}/{ITERATIONS} - Collected: {model.environment[EnvResourceComponent].resources} Gini: {gini(wealth_arr)}',
              file=sys.stdout, flush=True, end='\r')

        if args.visualize:
            # Will generate a series of figures of the environment.
            fig, ax = plt.subplots()
            img = np.copy(model.environment.cells[Gather.RESOURCE_KEY].to_numpy()).reshape(ENV_SIZE,ENV_SIZE)

            for agent in model.environment:
                x, y = agent[ENV.PositionComponent].xy()
                img[y][x] = agent_val

            ax.imshow(img, cmap='Set1')
            fig.savefig(f'./output/iteration_{i}.png')
            plt.close(fig)

    print()
    if args.visualize:
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