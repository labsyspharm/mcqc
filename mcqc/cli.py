import argparse
import pandas as pd
import numpy as np
import re
import sys
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
from . import __version__


def compute_gmm(data):

    data = np.log(data).values.reshape(-1, 1)

    # select number components based on Bayesian Information Criterion (BIC)

    best_bic = np.inf  # initializing with a high value for minimization
    best_n_components = -1

    for n_components in range(1, 3 + 1):
        gmm = GaussianMixture(n_components=n_components, random_state=0)
        gmm.fit(data)
        bic = gmm.bic(data)  # calculate BIC

        if bic < best_bic:
            best_bic = bic
            best_n_components = n_components

    # fit a Gaussian mixture model to histogram data using best_n_components
    gmm = GaussianMixture(n_components=best_n_components, random_state=0)
    gmm.fit(data)

    # generate points (along the DNA intensity histogram X range) for computing GMM
    x_min = data.min()
    x_max = data.max()
    x = np.linspace(x_min, x_max, 100)
    log_prob = gmm.score_samples(x.reshape(-1, 1))
    pdf = np.exp(log_prob)

    # find the GMM component with the tallest peak
    peak_maxs = []
    for i in range(gmm.n_components):
        pdf = (
            (gmm.weights_[i] *
             norm.pdf(x, gmm.means_[i, 0], np.sqrt(gmm.covariances_[i, 0, 0])))
        )
        peak_maxs.append(pdf.max())
    comp_index = np.argmax(peak_maxs)

    comp_mean = gmm.means_[comp_index, 0]
    comp_std = np.sqrt(gmm.covariances_[comp_index, 0, 0])
    dmin = np.exp(norm.ppf(0.005, comp_mean, comp_std))
    dmax = np.exp(norm.ppf(0.995, comp_mean, comp_std))

    return dmin, dmax


def error(msg):
    print(f"mcqc: error: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--output-filtered')
    parser.add_argument(
        '--version', action='version', version=f'mcqc {__version__}'
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    dna_re = r'(?i).*(dna|hoechst)'
    dna_columns = df.columns.str.match(dna_re)
    if not any(dna_columns):
        error("Second column name must contain 'DNA' or 'Hoechst' (case insensitive)")
    dna_idx = list(dna_columns).index(True)

    if 'Area' not in df.columns:
        error("No 'Area' column found")

    df['qc_reason'] = None

    # Flag cells with abnormal DNA intensity
    data = df.iloc[:, dna_idx]
    dmin, dmax = compute_gmm(data)
    drop = (data < dmin) | (data > dmax)
    df.loc[drop, 'qc_reason'] = 'intensity_dna'

    # Flag cells with abnormal area (size)
    data = df.loc[:, "Area"]
    dmin, dmax = compute_gmm(data)
    drop = (data < dmin) | (data > dmax)
    df.loc[drop, 'qc_reason'] = 'area'

    df.to_csv(args.output, index=False)
    if args.output_filtered:
        df[df['qc_reason'].isna()].iloc[:, :-1].to_csv(args.output_filtered, index=False)


if __name__ == '__main__':
    main()
