import networkx as nx
import numpy as np
import pandas as pd
import time
import glob
import sys
sys.path.append("..")
from fastmap import fastmap_subG
from fastmap_paspd import PASPD_distances, subgraphs, fastmap_PASPD_subG
from makedata import graphfile_loader



class RealCentra():
	def __init__(self, G, real_V, method='APSP', metrics='dijkstra', epsilon=1):
		self.G = G
		self.real_V = list(real_V)
		self.real_Vset = set(real_V)
		self.real_Vnum = len(real_V)
		self.get_distance_mat(method, metrics, epsilon)
		

	def get_distance_mat(self, method, metrics, epsilon):
		self.method, self.metrics, self.epsilon = method, metrics, epsilon
		start_time = time.time()
		if self.metrics == 'paspd' and self.method != 'FM':
			self.subGs = subgraphs(self.G, 2)
		if self.method == 'APSP':
			self.distance_mat = self.APSP_distances()
		elif self.method == 'DH':
			self.distance_mat = self.DH_distances()
		elif self.method == 'FM':
			self.distance_mat = self.FM_distances()
		self.distance_mat[self.distance_mat < self.epsilon] = self.epsilon
		np.fill_diagonal(self.distance_mat, 0)
		self.distance_time = time.time() - start_time


	def APSP_distances(self):
		distance_mat = np.zeros((self.real_Vnum, self.real_Vnum))
		for v1 in range(0, self.real_Vnum - 1):
			if self.metrics == 'dijkstra':
				dijkstra_distances = nx.single_source_dijkstra_path_length(self.G, self.real_V[v1])
			elif self.metrics == 'paspd':
				dijkstra_distances = PASPD_distances(self.G, self.subGs, 2, self.real_V[v1])
			for v2 in range(v1 + 1, self.real_Vnum):
				distance_mat[v1, v2] = distance_mat[v2, v1] =  dijkstra_distances[self.real_V[v2]]
		return distance_mat


	def DH_distances(self, sample_size=5):
		if sample_size >= self.real_Vnum:
			return self.APSP_distances()
		distance_mat = np.zeros((self.real_Vnum, self.real_Vnum))
		sample_V, sample_trees = np.random.choice(self.real_V, sample_size, False), {}
		for v in sample_V:
			if self.metrics == 'dijkstra':
				sample_trees[v] = nx.single_source_dijkstra_path_length(self.G, v)
			elif self.metrics == 'paspd':
				sample_trees[v] = PASPD_distances(self.G, self.subGs, 2, v)
		for v1 in range(0, self.real_Vnum - 1):
			for v2 in range(v1 + 1, self.real_Vnum):
				if self.real_V[v1] in sample_V:
					distance_mat[v1, v2] = distance_mat[v2, v1] = sample_trees[self.real_V[v1]][self.real_V[v2]]
				elif self.real_V[v2] in sample_V:
					distance_mat[v1, v2] = distance_mat[v2, v1] = sample_trees[self.real_V[v2]][self.real_V[v1]]
				else:
					samples2_v1, samples2_v2 = np.zeros(sample_size), np.zeros(sample_size)
					for c in range(sample_size):
						samples2_v1[c] = sample_trees[sample_V[c]][self.real_V[v1]]
						samples2_v2[c] = sample_trees[sample_V[c]][self.real_V[v2]]
						distance_mat[v1, v2] = distance_mat[v2, v1] = np.max(np.abs(samples2_v2 - samples2_v1))
		return distance_mat


	def FM_distances(self, K=4):
		distance_mat = np.zeros((self.real_Vnum, self.real_Vnum))
		if self.metrics == 'dijkstra':
			P = fastmap_subG(self.G, self.real_V, K)
		elif self.metrics == 'paspd':
			P = fastmap_PASPD_subG(self.G, self.real_V, K)
		for v1 in range(0, self.real_Vnum - 1):
			for v2 in range(v1 + 1, self.real_Vnum):
				distance_mat[v1, v2] = distance_mat[v2, v1] = np.sqrt(np.sum(np.power(P[v1] - P[v2], 2)))
		return distance_mat


	def topk_vertices(self, measure, k):
		start_time = time.time()
		if measure in ['clo', 'cfc']:
			all_scores = self.closeness_scores()
		elif measure == 'har':
			all_scores = self.harmonic_scores()
		elif measure == 'eig':
			all_scores = self.eigenvector_scores()
		k_vertices = np.argsort(all_scores)[::-1][:k]
		return k_vertices, time.time() - start_time + self.distance_time, all_scores


	def closeness_scores(self):
		return (self.real_Vnum - 1) / np.sum(self.distance_mat, axis=1)


	def harmonic_scores(self):
		distance_mat_helper = np.copy(self.distance_mat)
		np.fill_diagonal(distance_mat_helper, 1)
		distance_mat_helper = 1/distance_mat_helper
		scores =  np.sum(distance_mat_helper, axis=1) - 1
		return scores


	def eigenvector_scores(self):
		#eig_vals, eig_vecs = np.linalg.eig(self.distance_mat)
		trans_mat = np.copy(self.distance_mat)
		np.fill_diagonal(trans_mat, 1)
		trans_mat = 1 / trans_mat
		np.fill_diagonal(trans_mat, 0)
		trans_mat = trans_mat / np.sum(trans_mat, axis=1)[:, None]
		trans_mat = np.transpose(trans_mat)
		eig_vals, eig_vecs = np.linalg.eig(trans_mat)
		largest_eig_idx = np.argmax(eig_vals)
		scores = np.abs(eig_vecs[:, largest_eig_idx])
		return scores



if __name__ == '__main__':
	file = "../datasets/wDIMACS/games120.col"
	G, frac = graphfile_loader(file), 0.5
	real_V = np.sort(np.random.choice(G.nodes, int(len(G.nodes) * frac), False))


	method, measure, k = 'FM', 'eig', 10
	solver = RealCentra(G, real_V, method)
	k_vertices, total_time, all_scores = solver.topk_vertices(measure, k)