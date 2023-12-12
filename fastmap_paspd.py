import networkx as nx
import numpy as np
import math
import random



def subgraphs(G, set_num, steps=10):
	subgraphs = []
	for i in range(set_num):
		generate_subgraphs(G, subgraphs, steps)
	return subgraphs


def generate_subgraphs(G, subgraphs, steps=10):
	edges = list(G.edges())
	random.shuffle(edges)
	edges_num = len(edges)
	stepsize = math.ceil(edges_num / steps)
	pre_G = G
	for i in range(5):
		cur_G = pre_G.copy()
		cur_G.remove_edges_from(edges[i * stepsize: min((i + 1) * stepsize, edges_num)])
		subgraphs.append(cur_G)
		pre_G = cur_G


def PASPD_distances(graph, subgraphs, set_num, source):
	distances = get_distances([graph], source) * set_num
	distances += get_distances(subgraphs, source)
	return distances


def get_distances(graphs, source):
	V = len(graphs[0].nodes())
	dijkstra_distances = []
	distances = np.zeros(V)
	for G in graphs:
		dijkstra_distances.append(nx.single_source_dijkstra_path_length(G, source))	
	max_distances = []
	for dijkstra_distance in dijkstra_distances:
		max_distances.append(2 * (np.max(list(dijkstra_distance.values()))))
	for v in range(V):
		for dijkstra_distance, max_dis in zip(dijkstra_distances, max_distances):
			if v in dijkstra_distance:
				distances[v] += dijkstra_distance[v]
			else:
				distances[v] += max_dis
	return distances


def fastmap_PASPD(G, K, e=0.0001, set_num=2, C=10):
	subGs = subgraphs(G, set_num)
	V = len(G.nodes())
	P = np.zeros((V, K))
	for k in range(K):
		v_a = np.random.randint(V)
		v_b = v_a
		for t in range(C):
			d_ai = PASPD_distances(G, subGs, set_num, v_a)
			d_ai_new2 = np.power(d_ai, 2) - np.sum(np.power(P[v_a, :k] - P[:, :k], 2), axis=1)
			v_c = np.argmax(d_ai_new2)
			if v_c == v_b:
				break
			elif t < C - 1:
				v_b = v_a
				v_a = v_c
				d_ib_new2 = d_ai_new2
		d_ab_new2 = d_ai_new2[v_b]
		if d_ab_new2 < e:
			P = P[:, :k]
			break
		P[:, k] = (d_ai_new2 + d_ab_new2 - d_ib_new2) / (2 * np.sqrt(d_ab_new2))
	return P



def fastmap_PASPD_subG(G, sub_V, K, e=0.0001, set_num=2, C=10):
	size, subV2idx = len(sub_V), {}
	for v_idx in range(size):
		subV2idx[sub_V[v_idx]] = v_idx
	subGs = subgraphs(G, set_num)
	P = np.zeros((size, K))
	for k in range(K):
		v_a = np.random.choice(sub_V, 1)[0]
		v_b = v_a
		for t in range(C):
			d_ai_all = PASPD_distances(G, subGs, set_num, v_a)
			d_ai_new2 = np.power(d_ai_all[sub_V], 2) - np.sum(np.power(P[subV2idx[v_a], :k] - P[:, :k], 2), axis=1)
			v_c = sub_V[np.argmax(d_ai_new2)]
			if v_c == v_b:
				break
			else:
				v_b = v_a
				v_a = v_c
				d_ib_new2 = d_ai_new2
		d_ab_new2 = d_ai_new2[subV2idx[v_b]]
		if d_ab_new2 < e:
			P = P[:, :k]
			break
		P[:, k] = (d_ai_new2 + d_ab_new2 - d_ib_new2) / (2 * np.sqrt(d_ab_new2))
	return P