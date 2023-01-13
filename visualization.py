import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from makedata import create_folder
import networkx as nx



def plot_distribution(gt_centralities, fm_centralities, name='distribution'):
	gt_cvalues = list(gt_centralities.values())
	gt_cvalues = np.array(gt_cvalues) / max(gt_cvalues)
	fm_cvalues = list(fm_centralities.values())
	fm_cvalues = np.array(fm_cvalues) / max(fm_cvalues)
	sns.distplot(gt_cvalues, label='GroundTruth')
	sns.distplot(fm_cvalues, label='FastMap')
	plt.legend()
	plt.savefig(name + '.png')
	plt.close()


def centrality_distribution(measure, path, folder):
	files = glob.glob(path + '*.col')
	files.sort()
	img_path = path + folder
	create_folder(img_path)
	for file in files:
		G = graphfile_loader(file)
		name = file[file.rindex('/') + 1: file.rindex('.')]
		fmcl, gt = FMCL(G), Centrality(G)
		if measure == 'clo':
			fm_centralities, fm_time = fmcl.all_centrality(fmcl.closeness)
			gt_centralities, gt_topkV= gt.all_closeness()
		elif measure == 'har':
			fm_centralities, fm_time = fmcl.all_centrality(fmcl.harmonic)
			gt_centralities, gt_topkV= gt.all_harmonic()
		plot_distribution(gt_centralities, fm_centralities, img_path + name)


def topk_3D(P, topkV, fm_topkV, nDCG, fig_name='test.png', centralities=None):
	try:
	    fig = plt.figure()
	    if P.shape[1] >= 3:
	    	ax = fig.add_subplot(111, projection='3d')
	    for v in range(P.shape[0]):
	    	size = 12
	    	if v in fm_topkV:
	    		color = 'r'
	    	elif v in topkV:
	    		color = 'b'
	    	else:
	    		color, size = 'g', 4
	    	point = P[v]
	    	if P.shape[1] >= 3:
	    		ax.scatter(point[0], point[1], point[2], color=color, marker='o', s=size)
	    	else:
	    		plt.scatter(point[0], point[1], color=color, marker='o', s=size)
	    	#if v in topkV and centralities is not None:
	    	#	if P.shape[1] >= 3:
	    	#		ax.text(point[0], point[1], point[2], str(round(centralities[v], 4)), size=5, zorder=1, color='k')
	    	#	else:
	    	#		plt.text(point[0], point[1], str(round(centralities[v], 4)), size=5, zorder=1,color='k') 
	    plt.ion()
	    plt.title('nDCG = ' + str(nDCG))
	    plt.savefig(fig_name, dpi=300)
	    plt.close()
	except Exception as e:
		print(e)
		plt.close()


def visual_graph(G, topkV, fm_topkV, fig_name='test.png'):
	nodes_color = []
	for node in G.nodes:
		if node in fm_topkV:
			nodes_color.append('r')
		elif node in topkV:
			nodes_color.append('b')
		else:
			nodes_color.append('silver')
	nx.draw(G, node_color=nodes_color, node_size=20)
	plt.ion()
	plt.show()
	plt.savefig(fig_name, dpi=300)
	plt.close()