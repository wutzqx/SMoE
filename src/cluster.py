import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
from sklearn.metrics import silhouette_score



def extract_frequency_features(time_series, sampling_rate=1000, n_components=10):

    if time_series.shape[0] < time_series.shape[1]:
        time_series = time_series.T

    n_channels, n_samples = time_series.shape

    freqs = np.fft.fftfreq(n_samples, 1 / sampling_rate)
    fft_values = np.fft.fft(time_series, axis=1)

    positive_freq_mask = freqs >= 0
    magnitude_spectrum = np.abs(fft_values[:, positive_freq_mask])
    freqs_positive = freqs[positive_freq_mask]

    feature_list = []

    freq_bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 100)
    }

    for channel_idx in range(n_channels):
        channel_features = []

        for band_name, (low_freq, high_freq) in freq_bands.items():
            band_mask = (freqs_positive >= low_freq) & (freqs_positive <= high_freq)
            band_energy = np.sum(magnitude_spectrum[channel_idx, band_mask])
            channel_features.append(band_energy)

        spectrum_centroid = np.sum(freqs_positive * magnitude_spectrum[channel_idx]) / np.sum(
            magnitude_spectrum[channel_idx])
        channel_features.append(spectrum_centroid)

        weighted_diff = np.abs(freqs_positive - spectrum_centroid)
        spectral_bandwidth = np.sum(weighted_diff * magnitude_spectrum[channel_idx]) / np.sum(
            magnitude_spectrum[channel_idx])
        channel_features.append(spectral_bandwidth)

        feature_list.append(channel_features)

    features = np.array(feature_list)

    pca = PCA(n_components=min(n_components, features.shape[1]))
    reduced_features = pca.fit_transform(features)

    return reduced_features

def cluster_channels(data_train, data_test, n_clusters=0):
    '''
    This code will be made public after publication.
    '''
    print('This code will be made public after publication.')
    cluster_labels, grouped_data_train, grouped_data_test, cluster_counts=0
    return cluster_labels, grouped_data_train, grouped_data_test, cluster_counts

def cluster_rerank(data, cluster_labels):
    data = data.transpose(0, 1)
    r_data = []
    counts = Counter(cluster_labels)
    n_counts = [0] * len(counts)
    for i in range(1, len(n_counts)):
        n_counts[i] = n_counts[i-1] + counts[i-1]
    for i in range(len(cluster_labels)):
        cl = cluster_labels[i]
        r_data.append(data[n_counts[cl]])
        n_counts[cl] += 1
    r_data = torch.stack(r_data, dim=0)
    r_data = r_data.transpose(0, 1)
    return r_data

def reduce_dimension_with_pca(data, n_components=20):

    pca = PCA(n_components=n_components)

    reduced_data = pca.fit_transform(data)

    return reduced_data