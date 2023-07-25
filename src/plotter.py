import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                        help='The directory containing the batch files.', type=str)
    parser.add_argument('-o', '--output', help='The name of the CSV file to write to.', type=str)
    args = parser.parse_args()

    c_df = pd.read_csv(args.input + '/collected.csv')
    g_df = pd.read_csv(args.input + '/gini.csv')

    iterations = np.arange(len(c_df['mean']))

    fig, ax = plt.subplots()
    ax.set_title('Collected Resources Averaged over 50 runs.')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Collected')
    ax.plot(iterations, c_df['mean'])
    ax.fill_between(iterations, c_df['mean'] - c_df['stddev'], c_df['mean'] + c_df['stddev'], alpha=0.35)
    fig.savefig(args.output + '/collected.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()