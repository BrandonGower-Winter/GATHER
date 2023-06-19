# GATHER: A hunter-gatherer foraging simulator
*GATHER* is an agent-based model (ABM)... TODO 
## Installation

**Note:** This document assumes you are using a UNIX-based OS (i.e. Ubuntu or MacOS)

**Note:** Please make sure that you are using a version of `ECAgent >= 0.5.0`

To use GATHER, first clone the repo and navigate to the *GATHER/* directory using your favourite terminal application.
We first need to create a virtual environment for GATHER and we do so by typing:

`> make`

into your terminal. That should create a virtual environment with all the necessary modules needed to run NeoCOOP.
Next activate the virtual environment by typing:

`> source ./venv/bin/activate`

## Running a Single Simulation:

To run an instance of GATHER, first make sure you have installed all the necessary modules and have the virtual environment activated.

To run GATHER, type the following:
```bash
> python main.py
```

For help with GATHER, type:
```bash
> python main.py -h
```

which will output:
```bash
usage: main.py [-h] [-s SIZE] [-n NEST] [-a A] [--deposit DEPOSIT] [--decay DECAY] 
    [-i I] [--mode MODE] [--network NETWORK] [-v] [--seed SEED]

optional arguments:
  -h, --help            show this help message and exit
  -s SIZE, --size SIZE  The size of the environment. In all instances the environment will be a 
                        square gridworld.
  -n NEST, --nest NEST  The size of the nest. In all instances the nest will be square.
  -a A, --agents A      The number of agents to initialize.
  --deposit DEPOSIT     The amount of pheromone dropped by the agents.
  --decay DECAY         The rate at which the pheromone evaporates.
  -i I, --iterations I  How long the simulation should run for.
  --mode MODE           What mode the environment should be initialized to
  --network NETWORK     The type of network to initialize
  -v, --visualize       Whether environment images should be written to the output/ directory
  --seed SEED           Specify the seed for the Model's pseudorandom number generator

```
## Visualizing a Single Simulation:

TODO

## Running Multiple Simulations at Once:

To run a batch of simulations, use:
```shell
./batchrunner.sh OUTPUT_DIR "ARGS"
```
where `OUTPUT_DIR` is where the output for each simulation should
be written to (grouped by seed) and `"ARGS"` is a string containing
the optional parameters that each simulation in the batch will be run with.

For example, `"--hdecay .01 --fdecay .01 --agents 100"` will run
all simulations in the batch with an `hdecay` and `fdecay` of `0.01`
with a populations of `100` agents.

**Note**: You must surround `ARGS` with double quotes `" "`. 

## Processing Batch Simulations:

Use `batch2cv.py` as follows:
```python
python batch2csv.py -i INPUT_DIR -o OUTPUT_DIR
```
Where `INPUT_DIR` is the directory containing the outputs produced 
by a batch simulation and `OUTPUT_DIR` is the directory where the
CSV files will be written to. 

## Plotting Graphs:

TODO
