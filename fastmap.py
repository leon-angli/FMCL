import networkx as nx
import numpy as np



# return the shortest path distance of each node for a given graph G and a source node
def get_distance(G, source, q=1):
	V = len(G.nodes())
	dijkstra_distance = nx.single_source_dijkstra_path_length(G, source)
	max_distance = np.max(list(dijkstra_distance.values()))
	distance = np.ones(V) * (2 * max_distance)
	for node, dist in dijkstra_distance.items():
		distance[node] = dist
	distance = np.power(distance, q)
	return distance


# embedding a given graph into K-dimensional points
# G: undirected single-view networkx graph
# K: user-specifeid K-dimensional space for the embedding
# e: a threshold for early stop (the default value can be used directly)
# return: numpy array with size (V, K)
def fastmap_L2(G, K, e=0.0001, q=1, C=10):
	V = len(G.nodes())
	P = np.zeros((V, K))
	for k in range(K):
		v_a = np.random.randint(V)
		v_b = v_a
		for t in range(C):
			d_ai = get_distance(G, v_a, q)
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


def fastmap_subG(G, sub_V, K, e=0.0001, q=1, C=10):
	size, subV2idx = len(sub_V), {}
	for v_idx in range(size):
		subV2idx[sub_V[v_idx]] = v_idx
	P = np.zeros((size, K))
	for k in range(K):
		v_a = np.random.choice(sub_V, 1)[0]
		v_b = v_a
		for t in range(C):
			d_ai_all = get_distance(G, v_a, q)
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