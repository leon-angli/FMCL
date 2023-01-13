import networkx as nx
import numpy as np
import pandas as pd
import time
import glob
from falconn import *
from fastmap import fastmap_L2, fastmap_subG
from fastmap_paspd import fastmap_PASPD, fastmap_PASPD_subG, subgraphs
from centrality import Centrality
from makedata import graphfile_loader, movingAI_visual
from visualization import topk_3D, visual_graph
from sklearn.mixture import GaussianMixture
from collections import Counter
from scipy import stats
from sklearn import metrics



class FMCL():
	def __init__(self, measure, G, K=4, e=0.0001):
		self.G = G
		self.K = K
		self.e = e
		self.V = len(self.G.nodes())
		self.set_measure(measure)


	def set_measure(self, measure):
		self.measure = measure
		if self.measure == 'clo':
			self.power = 0.5
		elif self.measure == 'har':
			self.power = 1
		elif self.measure in ['cfc', 'eig', 'bet']:
			self.power = 1


	def embedding(self):
		start_time = time.time()
		if self.measure != 'cfc':
			self.P = fastmap_L2(self.G, self.K, self.e, self.power)
		elif self.measure =='cfc':
			self.P = fastmap_PASPD(self.G, self.K, self.e, set_num=2)
			#self.P = fastmap_L2(self.G, self.K, self.e, self.power)
		self.embedding_time = time.time() - start_time


	def sub_embedding(self, real_V):
		start_time = time.time()
		if self.measure != 'cfc':
			self.sub_P = fastmap_subG(self.G, real_V, self.K, self.e, self.power)
		elif self.measure =='cfc':
			self.sub_P = fastmap_PASPD_subG(self.G, real_V, self.K, self.e, set_num=2)
		self.embedding_time = time.time() - start_time


	def set_LSH(self, P):
		N, d = P.shape
		P = P.astype(np.float32)
		P -= np.mean(P, axis=0)
		lsh = LSHIndex(get_default_parameters(N, d))
		lsh.setup(P)
		self.LSH = lsh.construct_query_object()
		self.LSH.set_num_probes(N)
		return P

 
	def topk_vertices(self, k):
		start_time = time.time()
		self.embedding()
		self.P = self.set_LSH(self.P)
		if self.measure in ['clo', 'cfc']:
			center = np.zeros(self.P.shape[1], dtype=np.float32)
			vertices = self.LSH.find_k_nearest_neighbors(center, k)
		elif self.measure == 'har':
			center = self.harmonic_center(self.P)
			vertices = self.LSH.find_k_nearest_neighbors(center, k)
		elif self.measure == 'eig':
			vertices = self.eigen_2layer_GMM_topkV(self.P, k)
		elif self.measure == 'bet':
			vertices = self.between_topkV(k)
		return vertices, time.time() - start_time + self.embedding_time


	def real_topk_vertices(self, real_V, k, approach='adp'):
		start_time = time.time()
		if approach == 'adp':
			self.embedding()
			self.sub_P = self.set_LSH(self.P[real_V])
		elif approach == 'sub':
			self.sub_embedding(real_V)
			self.sub_P = self.set_LSH(self.sub_P)
		if self.measure in ['clo', 'cfc']:
			center = np.zeros(self.sub_P.shape[1], dtype=np.float32)
			vertices = self.LSH.find_k_nearest_neighbors(center, k)
		elif self.measure == 'har':
			center = self.harmonic_center(self.sub_P)
			vertices = self.LSH.find_k_nearest_neighbors(center, k)
		elif self.measure == 'eig':
			vertices = self.eigen_2layer_GMM_topkV(self.sub_P, k)
		return vertices, time.time() - start_time + self.embedding_time
		


	def get_GMM_center_index(self, means):
		gmm_centers = np.array(means, dtype=np.float32)
		self.gmm_center_index = []
		for center in gmm_centers:
			self.gmm_center_index.append(self.LSH.find_nearest_neighbor(center))


	def eigen_2layer_GMM_topkV(self, P, k, layer_components=3):
		GMM = GaussianMixture(n_components=layer_components)
		labels = GMM.fit_predict(P)
		sub_centers, sub_weights = [], []
		for component in range(layer_components):
			sub_GMM = GaussianMixture(n_components=layer_components)
			sub_GMM.fit(P[np.where(labels == component)])
			sub_centers.extend(sub_GMM.means_.tolist())
			sub_weights.extend(sub_GMM.weights_ * GMM.weights_[component])
		#self.get_GMM_center_index(sub_centers)
		center = np.array(sub_centers[np.argmax(sub_weights)], dtype=np.float32)
		vertices = self.LSH.find_k_nearest_neighbors(center, k)
		return vertices


	def find_best_GMM(self, is_soft=False, min_components=2, max_components=10):
		best_silhouette_score, best_GMM, best_membership = -1, None, None
		for components in range(min_components, max_components + 1):
			GMM = GaussianMixture(n_components=components)
			labels = GMM.fit_predict(self.P)
			silhouette_score = metrics.silhouette_score(self.P, labels)
			if silhouette_score > best_silhouette_score:
				best_silhouette_score = silhouette_score
				best_GMM, best_membership = GMM, labels
				if is_soft:
					best_membership = best_GMM.predict_proba(self.P)
		return best_GMM, best_membership


	def between_topkV(self, k):
		GMM, probas = self.find_best_GMM(is_soft=True)
		scores = self.between_scores(probas, GMM.weights_)
		return np.argsort(scores)[::-1][:k]


	def between_scores(self, probas, weights):
		V, components = probas.shape
		scores = np.zeros(V)
		for v in range(V):
			for c1 in range(components):
				for c2 in range(c1 + 1, components):
					scores[v] += probas[v, c1] * probas[v, c2] * weights[c1] * weights[c2]
		return scores

 
	def harmonic_center(self, P, epoch=100, lr=0.001):
		V, center = len(P), np.random.rand(P.shape[1]).astype(np.float32)
		for t in range(epoch):
			GD = np.zeros(P.shape[1])
			for i in range(V):
				GD += -(center - P[i]) / np.power(self.euclidean_distance(P[i], center), 3)
			center += lr * (GD / V)
		return center


	def euclidean_distance(self, v1, v2):
		return np.power(np.sum(np.power(v1 - v2, 2)), 0.5/self.power)



def DCG(topkV, centralities):
	DCG = 0
	for i in range(len(topkV)):
		DCG += centralities[topkV[i]] / np.log2(i + 2)
	return DCG


def nDCG(fm_topkV, gt_topkV, centralities):
	return DCG(fm_topkV, centralities) / DCG(gt_topkV, centralities)


def resutls_latex(results, columns, filename='table.tex'):
	df = pd.DataFrame(results)
	df.columns = columns
	df = df.sort_values(['Size'], ascending=[True])
	latex_table = df.to_latex(index=False)
	with open(filename, 'w+') as tex:
		tex.write(latex_table)


def topk_tests(measure, path, k_value, trial=1, outfile='table.tex'):
	files = glob.glob(path + '*.col')
	files.sort()
	results = []
	for file in files:
		try:
			result = topk_test(measure, file, k_value, trial)
			results.append(result)
			print(result)
		except Exception as e:
			print(e)
	resutls_latex(results, ['Instance', 'Size', 'NetworkX', 'FastMap', 'nDCG'], outfile)


def topk_test(measure, file, k_value, trial=1):
	name = file[file.rindex('/') + 1: file.rindex('.')]
	G = graphfile_loader(file)
	size = (len(G.nodes()), len(G.edges()))
	mean_fmk_time, mean_gtk_time = 0, 0
	mean_nDCG = 0
	for t in range(trial):
		fmcl, gt = FMCL(measure, G), Centrality(measure, G)
		gt_centralities, gt_topkV, gt_time, gtk_time = gt.topk_vertices(k_value)
		fm_topkV, fmk_time = fmcl.topk_vertices(k_value)
		mean_fmk_time, mean_gtk_time = mean_fmk_time + fmk_time, mean_gtk_time + gtk_time
		mean_nDCG += nDCG(fm_topkV, gt_topkV, gt_centralities)
	mean_fmk_time, mean_gtk_time = round(mean_fmk_time / trial, 2), round(mean_gtk_time / trial, 2)
	mean_nDCG = round(mean_nDCG / trial, 4)
	#topk_3D(fmcl.P, gt_topkV, fm_topkV, mean_nDCG, 'eig_3d/' + dataset + '/' + name + '.png', gt_centralities)
	fmcl.gmm_center_index = []
	#movingAI_visual(name, gt_topkV, fm_topkV, fmcl.gmm_center_index, f'movingAI_{measure}/' + name + '.png', 'datasets/mAIclusters/')
	#visual_graph(G, gt_topkV, fm_topkV, f'g_{dataset}_{measure}/{name}.png')
	return [name, size, mean_gtk_time, mean_fmk_time, mean_nDCG]


if __name__ == '__main__':
	k_value = 10
	measures = ['clo', 'har', 'cfc', 'eig', 'bet']
	datasets = ['DIMACS', 'wDIMACS', 'TSP', 'Tree', 'SmallWorld', 'movingAI', 'movingAI2']

	measures = ['eig']
	datasets = ['wDIMACS']

	for measure in measures:
		for dataset in datasets:
			path, table_file = 'datasets/' + dataset + '/', 'tables/' + dataset + '_' + measure +'.tex'
			#table_file = 'test.tex'
			topk_tests(measure, path, k_value, 1, table_file)


