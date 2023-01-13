import networkx as nx
import numpy as np
from collections import defaultdict
import time


class Centrality():
	def __init__(self, measure, G):
		self.set_measure(measure)
		self.G = G
		self.V = len(G.nodes())


	def set_measure(self, measure):
		self.measure = measure
		if self.measure == 'clo':
			self.all_centrality_function = self.all_closeness
		elif self.measure == 'har':
			self.all_centrality_function = self.all_harmonic
		elif self.measure == 'bet':
			self.all_centrality_function = self.all_betweenness
		elif self.measure == 'cfc':
			self.all_centrality_function = self.all_current_flow_closeness
		elif self.measure == 'cfb':
			self.all_centrality_function = self.all_current_flow_betweenness
		elif self.measure == 'eig':
			self.all_centrality_function = self.all_eigenvector


	def closeness(self, x):
		distances = nx.single_source_dijkstra_path_length(self.G, x)
		distances = defaultdict(lambda: float('inf'), distances)
		cur_sum = 0
		for v in range(self.V):
			cur_sum += distances[v] if distances[v] != float('inf') else 0
		centrality = 0 if cur_sum == 0 else 1 / cur_sum
		return centrality


	def harmonic(self, x):
		distances = nx.single_source_dijkstra_path_length(self.G, x)
		distances = defaultdict(lambda: float('inf'), distances)
		centrality = 0
		for v in range(self.V):
			distance = distances[v] if distances[v] != float('inf') else 0
			centrality += 0 if distance == 0 else 1 / distance
		return centrality


	def all_closeness(self):
		start_time = time.time()
		centralities = nx.closeness_centrality(self.G, distance='weight')
		return centralities, time.time() - start_time


	def all_harmonic(self):
		start_time = time.time()
		centralities = nx.harmonic_centrality(self.G, distance='weight')
		return centralities, time.time() - start_time


	def all_betweenness(self):
		start_time = time.time()
		centralities = nx.betweenness_centrality(self.G, weight='weight')
		#centralities = self.disturbed_bet()
		return centralities, time.time() - start_time


	def disturbed_G(self):
		G = self.G.copy()
		for edge in G.edges:
			G[edge[0]][edge[1]]['weight'] = np.random.uniform(0.5, 1.5)
		return G


	def disturbed_bet(self, trial=5):
		centralities = defaultdict(lambda: 0)
		for t in range(trial):
			G = self.disturbed_G()
			cur_centralities = nx.betweenness_centrality(G, weight='weight')
			for v in range(self.V):
				centralities[v] += cur_centralities[v]
		return centralities


	def all_current_flow_closeness(self):
		start_time = time.time()
		centralities = nx.current_flow_closeness_centrality(self.G, weight='weight')
		return centralities, time.time() - start_time


	def all_current_flow_betweenness(self):
		start_time = time.time()
		centralities = nx.current_flow_betweenness_centrality(self.G, weight='weight')
		return centralities, time.time() - start_time


	def all_eigenvector(self):
		start_time = time.time()
		#centralities = nx.eigenvector_centrality(self.G, max_iter=5000, weight='weight')
		trans_mat = np.array(nx.to_numpy_matrix(self.G))
		for v1 in range(self.V - 1):
			for v2 in range(v1 + 1, self.V):
				if trans_mat[v1, v2] != 0:
					trans_mat[v1, v2] = trans_mat[v2, v1] = 1 / trans_mat[v1, v2]
		trans_mat = trans_mat / np.sum(trans_mat, axis=1)[:, None]
		trans_mat = np.transpose(trans_mat)
		eig_vals, eig_vecs = np.linalg.eig(trans_mat)
		largest_eig_idx = np.argmax(eig_vals)
		centralities_vec = np.abs(eig_vecs[:, largest_eig_idx])
		centralities = {}
		for v in range(self.V):
			centralities[v] = centralities_vec[v]
		return centralities, time.time() - start_time


	def topk_vertices(self, k):
		centralities, run_time = self.all_centrality_function()
		start_time = time.time()
		vertices = [vertex for vertex, centrality in sorted(centralities.items(), key=lambda item: -item[1])]
		return centralities, vertices[:k], run_time, time.time() - start_time + run_time