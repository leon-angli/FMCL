import networkx as nx
import numpy as np
import os
import shutil
import glob
import warnings
import matplotlib.pyplot as plt
import matplotlib
warnings.simplefilter(action='ignore', category=FutureWarning)


def create_folder(path):
	try:
		os.mkdir(path)
	except:
		shutil.rmtree(path)
		os.mkdir(path)


# node start from 1
def graph_file(G, filename, max_weight=1, known_weight=False):
	V, E = len(G.nodes()), len(G.edges())
	filelines = [['p', 'edge', str(V), str(E)]]
	for edge in G.edges():
		v1, v2 = edge[0], edge[1]
		weight = np.random.randint(1, max_weight + 1) if not known_weight else G[v1][v2]['weight']
		filelines.append(['e', str(v1 + 1), str(v2 + 1), str(weight)])
	filetext = '\n'.join([' '.join(line) for line in filelines])
	with open(filename, 'w+') as file:
		file.write(filetext)


def smallworld_graphs(N, K, P, path, max_weight=1):
	create_folder(path)
	for n in N:
		for k in K:
			for p in P:
				filename = path + 'n' + str(n).zfill(4) + 'k' + str(k) + 'p' + str(p) + '.col'
				G = nx.newman_watts_strogatz_graph(n, k, p)
				graph_file(G, filename, max_weight)


def randomtree_graphs(N, path, max_weight=10):
	create_folder(path)
	for n in N:
		filename = path + 'n' + str(n).zfill(4) + '.col'
		G = nx.random_tree(n)
		graph_file(G, filename, max_weight)


def DIMACS_graphs(input_path, output_path, max_weight=1):
	create_folder(output_path)
	files = glob.glob(input_path + '*.col')
	files.sort()
	for file in files:
		name = file[file.rindex('/') + 1: file.rindex('.')]
		filename = output_path + name + '.col'
		G = graphfile_loader(file)
		graph_file(G, filename, max_weight)


def coordinates_graph(coordinates):
	G = nx.Graph()
	V, edges = len(coordinates), []
	G.add_nodes_from(range(V))
	for v1 in range(V):
		for v2 in range(v1 + 1, V):
			edges.append((v1, v2, np.sqrt(np.sum(np.power(coordinates[v1] - coordinates[v2], 2)))))
	G.add_weighted_edges_from(edges)
	return G


def TSP_graphs(input_path, output_path):
	create_folder(output_path)
	files = glob.glob(input_path + '*.tsp')
	files.sort()
	for file in files:
		name = file[file.rindex('/') + 1: file.rindex('.')]
		filename = output_path + name + '.col'
		coordinates = np.loadtxt(file)[:, 1:]
		G = coordinates_graph(coordinates)
		filename = output_path + name + '.col'
		graph_file(G, filename, 1, True)


def index_map(strmap):
	index = 0
	height, width = len(strmap), len(strmap[0])
	indexmap, v2location = np.ones((height, width), dtype=int) * -1, {}
	for x in range(height):
		for y in range(width):
			if strmap[x][y] == '.' or strmap[x][y] == 'G':
				indexmap[x, y] = index
				v2location[index] = (x, y)
				index += 1
	return indexmap, index, v2location


def indexmap2graph(indexmap, V):
	height, width = indexmap.shape
	G = nx.Graph()
	G.add_nodes_from(range(V))
	edges = []
	for x in range(height):
		for y in range(width): 
			if indexmap[x, y] == -1:
				continue
			if x + 1 < height and indexmap[x + 1, y] != -1:
				edges.append([indexmap[x, y], indexmap[x + 1, y], 1])
			if y + 1 < width and indexmap[x, y + 1] != -1:
				edges.append([indexmap[x, y], indexmap[x, y + 1], 1])
	G.add_weighted_edges_from(edges)
	return G


def get_strmap(input_file):
	with open(input_file) as file_content:
		filelines = file_content.read().splitlines()
	strmap = filelines[4:]
	return strmap


def movingAI_visual(name, topkV=[], fm_topkV=[], gmm_centers=[], output_file='map.png', folder='datasets/orimAI/'):
	colors = ['k', 'w']
	strmap = get_strmap(folder + name + '.map')
	indexmap, V, v2location = index_map(strmap)
	height, width = indexmap.shape
	fig, ax = plt.subplots()
	for x in range(height):
		for y in range(width):
			pixel = plt.Rectangle((x, y), width=1, height=1, linewidth=0, color=colors[int(strmap[x][y] in ['.', 'G'])])
			ax.add_artist(pixel)
	for v in topkV:
		#size = 2
		#location = (v2location[v][0] - size/2, v2location[v][1] - size/2)
		#topv = plt.Rectangle(location, width=size, height=size, linewidth=0, color='b')
		topv = plt.Rectangle((v2location[v][0], v2location[v][1]), width=1, height=1, linewidth=0, color='b')
		ax.add_artist(topv)
	for v in fm_topkV:
		topv = plt.Rectangle((v2location[v][0], v2location[v][1]), width=1, height=1, linewidth=0, color='r')
		ax.add_artist(topv)
	for v in gmm_centers:
		center = plt.Rectangle((v2location[v][0], v2location[v][1]), width=1, height=1, linewidth=0, color='gold')
		ax.add_artist(center)
	plt.ion()
	plt.title(name)
	plt.xlim(0, height)
	plt.ylim(0, width)
	plt.savefig(output_file, dpi=300)
	plt.close()


def movingAI_graph(input_file, output_file):
	strmap = get_strmap(input_file)
	indexmap, V, _ = index_map(strmap)
	G = indexmap2graph(indexmap, V)
	if not nx.is_connected(G):
		return
	graph_file(G, output_file, 1, True)


def movingAI_graphs(input_path, output_path):
	create_folder(output_path)
	files = glob.glob(input_path + '*.map')
	files.sort()
	for file in files:
		name = file[file.rindex('/') + 1: file.rindex('.')]
		filename = output_path + name + '.col'
		movingAI_graph(file, filename)


def graphfile_loader(filename, default_weight=1):
	G = nx.Graph()
	edges = []
	with open(filename) as file:
		filelines = file.readlines()
		for i in range(len(filelines)):
			if filelines[i][0] == 'p':
				info = filelines[i].strip().split(' ')
				V = int(info[2])
				G.add_nodes_from(range(V))
				break
		edgelines = filelines[i + 1:]
		for edgeline in edgelines:
			if edgeline[0] == 'e':
				edge = edgeline.strip().split(' ')
				weight = default_weight if len(edge) == 3 else float(edge[3])
				edges.append((int(edge[1]) - 1, int(edge[2]) - 1, weight))
		G.add_weighted_edges_from(edges)
	return G



if __name__ == '__main__':
	input_path, output_path = 'datasets/mAIclusters/', 'datasets/wMovingAI/'
	#smallworld_graphs(range(100, 1001, 100) + , [4, 6], [0.3, 0.6], output_path)
	#randomtree_graphs(range(100, 2001, 100), output_path, 1)
	#DIMACS_graphs('datasets/DIMACS_original/', 'datasets/wDIMACS/', 10)
	#TSP_graphs(input_path, output_path)
	#movingAI_graphs(input_path, output_path)
