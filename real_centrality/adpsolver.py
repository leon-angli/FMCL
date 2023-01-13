import networkx as nx
import networkit as nk
import numpy as np
import time
import glob
import sys
from collections import defaultdict
sys.path.append("..")
from makedata import graphfile_loader



class AdaptSolver():
	def __init__(self, G, real_V, weighted=True):
		self.weighted = weighted
		self.G = G
		self.nkG = nk.nxadapter.nx2nk(self.G, weightAttr='weight' if self.weighted else None)
		self.real_V = list(real_V)
		self.real_Vset = set(real_V)
		self.real_Vnum = len(real_V)


	def topk_vertices(self, measure, k):
		start_time = time.time()
		if measure == 'clo':
			all_scores = self.approx_closeness_scores()
			k_vertices = np.argsort(all_scores)[::-1][:k]
		if measure == 'har':
			nk_solver = nk.centrality.TopHarmonicCloseness(self.nkG, k, not self.weighted, self.real_V)
			nk_solver.run()
			actual_k_vertices = nk_solver.topkNodesList()
			k_vertices = [self.real_V.index(v) for v in actual_k_vertices]
		return k_vertices, time.time() - start_time


	def approx_closeness_scores(self, n_samples=10, epsilon=0.1):
		sample_V = np.random.choice(self.real_V, min(n_samples, self.real_Vnum), False)
		sample_Vset = set(sample_V)
		real_V_mCset = self.real_Vset - sample_Vset
		sample_dijk_distances = {}
		pivots, pivot_distances = {}, defaultdict(lambda: float('inf'))
		for v_c in sample_V:
			sample_dijk_distances[v_c] = nx.single_source_dijkstra_path_length(self.G, v_c)
			for v_j in real_V_mCset:
				if sample_dijk_distances[v_c][v_j] < pivot_distances[v_j]:
					pivots[v_j] = v_c
					pivot_distances[v_j] = sample_dijk_distances[v_c][v_j]
		scores = np.zeros(self.real_Vnum)
		for v_idx in range(self.real_Vnum):
			v_j = self.real_V[v_idx]
			if v_j in sample_Vset:
				for v_i in self.real_V:
					scores[v_idx] += sample_dijk_distances[v_j][v_i]
			else:
				v_j_pivot, dist_threshold = pivots[v_j], pivot_distances[v_j] / epsilon
				L, LC, LC_term = 0, 0, 0
				for v_i in self.real_V:
					if sample_dijk_distances[v_j_pivot][v_i] <= dist_threshold:
						L += 1
						if v_i in sample_Vset:
							LC += 1
							LC_term += sample_dijk_distances[v_i][v_j]
					elif v_i in sample_Vset:
						scores[v_idx] += sample_dijk_distances[v_i][v_j]
					else:
						scores[v_idx] += sample_dijk_distances[v_j_pivot][v_i]
				scores[v_idx] += L / LC * LC_term
		scores = (self.real_Vnum - 1) / scores
		return scores



if __name__ == '__main__':
	file = "../datasets/wDIMACS/games120.col"
	G, frac = graphfile_loader(file), 0.5
	real_V = np.sort(np.random.choice(G.nodes, int(len(G.nodes) * frac), False))


	measure, k = 'clo', 10
	solver = AdaptSolver(G, real_V)
	k_vertices, total_time, all_scores = solver.topk_vertices(measure, k)
