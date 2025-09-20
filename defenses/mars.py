from copy import deepcopy
from typing import List, Any, Dict

import torch
import logging
import os
import numpy as np
import sklearn.metrics.pairwise as smp
import hdbscan
from defenses.fedavg import FedAvg

logger = logging.getLogger('logger')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import wasserstein_distance
from torch import optim, nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans
import csv
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def calculate_tpr_fpr(l, n, m):
	actual_labels = np.array([1 if i < m else 0 for i in range(n)])

	detected_labels = np.array(l)
	TP = np.sum((actual_labels == 1) & (detected_labels == -1))
	FP = np.sum((actual_labels == 0) & (detected_labels == -1))
	FN = np.sum((actual_labels == 1) & (detected_labels == 0))
	TN = np.sum((actual_labels == 0) & (detected_labels == 0))

	TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
	FPR = FP / (FP + TN) if (FP + TN) > 0 else 0

	return TPR, FPR

def compute_wasserstein_distance_matrix(l):
	n = len(l)
	distance_matrix = np.zeros((n, n))
	for i in range(n):
		for j in range(i + 1, n):
			distance = wasserstein_distance(l[i], l[j])
			distance_matrix[i, j] = distance
			distance_matrix[j, i] = distance
	print(distance_matrix)
	return distance_matrix

def kmeans_wasserstein_clustering(l, k=2):
	distance_matrix = compute_wasserstein_distance_matrix(l)
	kmeans = KMeans(n_clusters=k, random_state=0).fit(distance_matrix)
	return kmeans

def compute_cluster_centers(l, labels, k):
	clusters = [[] for _ in range(k)]
	for i, label in enumerate(labels):
		clusters[label].append(l[i])

	centers = [np.mean(cluster, axis=0) for cluster in clusters]
	return centers

def detect_anomaly(l, kmeans, threshold):
	labels = kmeans.labels_
	centers = compute_cluster_centers(l, labels, k=2)

	center_distance = wasserstein_distance(centers[0], centers[1])
	print(center_distance)
	if center_distance <= threshold:
		return np.zeros_like(labels)
	else:
		center_norms = [np.linalg.norm(center, ord=1) for center in centers]
		print(center_norms)
		anomaly_cluster = np.argmax(center_norms)
		anomaly_labels = np.where(labels == anomaly_cluster, -1, 0)
		return anomaly_labels

def cluster_and_detect_anomalies(l, threshold):
	kmeans = kmeans_wasserstein_clustering(l)
	anomaly_labels = detect_anomaly(l, kmeans, threshold)
	return anomaly_labels

def CBE(net):
	params = net.state_dict()
	all_lips = []
	u = 0.05
	for name, m in net.named_modules():
		if isinstance(m, nn.BatchNorm2d):
			# Ensure no NaN in running_var
			m.running_var = torch.where(m.running_var != m.running_var, torch.zeros_like(m.running_var),
										m.running_var)  # replace NaN with 0
			# Ensure no negative values
			m.running_var = torch.where(m.running_var < 0, torch.zeros_like(m.running_var),
										m.running_var)  # replace negative with 0

			std = (m.running_var + 1e-5).sqrt()  # add small value to prevent sqrt(0)
			weight = m.weight
			# print(std)

			channel_lips = []
			for idx in range(weight.shape[0]):
				w = conv.weight[idx].reshape(conv.weight.shape[1], -1) * (weight[idx] /std[idx]).abs()
				channel_lips.append(torch.svd(w.cpu())[1].max())

			channel_lips = torch.Tensor(channel_lips)
			threshold = torch.quantile(channel_lips, 1 - u)
			top_k_lips = channel_lips[channel_lips >= threshold]
			all_lips.append(top_k_lips)

			index = torch.where(channel_lips >= threshold)[0]
			params[name + '.weight'][index] = 0
			params[name + '.bias'][index] = 0

			# print(f'{name} - Top {u * 100}% Lipschitz indices:', index)

		# Convolutional layer should be followed by a BN layer by default
		elif isinstance(m, nn.Conv2d):
			conv = m

	# net.load_state_dict(params)

	# Concatenate all top k% Lipschitz constants into a 1D vector
	all_lips_vector = torch.cat(all_lips)
	# print('All layers - Top k% Lipschitz constants:', all_lips_vector)
	return np.array(all_lips_vector)



class MARS(FedAvg):
    def aggr(self, weight_accumulator, global_model,idx):
        l = []
        for i in idx:
            updates_name = f'{self.params.folder_path}/saved_updates/update_{i}.pth'
            loaded_params = torch.load(updates_name)
            local_model = deepcopy(global_model)
            for name, data in loaded_params.items():
                if self.check_ignored_weights(name):
                    continue
                local_model.state_dict()[name].add_(data)
            l.append(CBE(local_model))
        result = cluster_and_detect_anomalies(l, 0.03)
        print(result)
        TPR, FPR = calculate_tpr_fpr(result,20,4)
        with open(f'{self.params.folder_path}/tpr_fpr_results.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([TPR, FPR])
        for i,index in enumerate(idx):
            if result[i] == -1:
                continue

            update_name = f'{self.params.folder_path}/saved_updates/update_{index}.pth'
            loaded_params = torch.load(update_name)
            self.accumulate_weights(weight_accumulator, loaded_params)

        return weight_accumulator