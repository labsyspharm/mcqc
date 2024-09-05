import argparse
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import norm


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--output-filtered')
    args = parser.parse_args()
    
    df = pd.read_csv(args.input)
    data = df.iloc[:, 1]

    dmin, dmax = compute_gmm(data)

    drop_intensity = (data < dmin) | (data > dmax)
    df['qc_reason'] = None
    df.loc[drop_intensity, 'qc_reason'] = 'intensity' 

    df.to_csv(args.output, index=False)
    if args.output_filtered:
    	df[df['qc_reason'].isna()].iloc[:, :-1].to_csv(args.output_filtered, index=False)


if __name__ == '__main__':
    main()