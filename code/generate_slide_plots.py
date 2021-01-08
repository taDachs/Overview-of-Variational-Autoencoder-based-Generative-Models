#!/usr/bin/env python3

import argparse
import os

from plots import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='script for generating the plots used in the presentation')
    parser.add_argument('--dst', metavar='[DESTINATION PATH]', type=str, required=True, help='path to plot destination')

    args = parser.parse_args()
    plot_path = args.dst

    fig, _ = plot_dataset_to_dist()
    fig.savefig(os.path.join(plot_path, 'generative_mode.pgf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(plot_path, 'generative_mode.pdf'), dpi=300, bbox_inches='tight')

    fig, _ = plot_encoding_comparison()
    fig.savefig(os.path.join(plot_path, 'encoding_comparison.pgf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(plot_path, 'encoding_comparison.pdf'), dpi=300, bbox_inches='tight')

    fig, _ = plot_beta_disentanglement()
    fig.savefig(os.path.join(plot_path, 'beta_disentanglement.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(plot_path, 'beta_disentanglement.pgf'), dpi=300, bbox_inches='tight')
