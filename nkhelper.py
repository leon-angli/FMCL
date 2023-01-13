import networkit as nk
import numpy as np
import time
import glob
from makedata import graphfile_loader
from centrality import Centrality
from fmcl import FMCL, nDCG, resutls_latex



class NkHelper():
	def __init__(self, measure, nxG, weighted=True):
		self.measure = measure
		self.G = nk.nxadapter.nx2nk(nxG, weightAttr='weight' if weighted else None)
		self.V = len(nxG.nodes())
		self.weighted = weighted


	def topk_vertices(self, k):
		start_time = time.time()
		if self.measure == 'clo':
			vertice = self.clo_topkV(k)
		elif self.measure == 'har':
			vertice = self.har_topkV(k)
		elif self.measure == 'cfc':
			vertice = self.cfc_tokV(k)
		return vertice, time.time() - start_time


	def clo_topkV(self, k):
		if self.weighted:
			nk_solver = nk.centrality.ApproxCloseness(self.G, 10)#int(self.V) / 10)
			nk_solver.run()
			vertice = np.array(nk_solver.scores()).argsort()[::-1][:k]
		else:
			nk_solver = nk.centrality.TopCloseness(self.G, k)
			nk_solver.run()
			vertice = nk_solver.topkNodesList()
		return vertice


	def har_topkV(self, k):
		if self.weighted:
			nk_solver = nk.centrality.TopHarmonicCloseness(self.G, k, False)
		else:
			nk_solver = nk.centrality.TopHarmonicCloseness(self.G, k, True)
		nk_solver.run()
		return nk_solver.topkNodesList()


	def cfc_tokV(self, k):
		if not self.weighted:
			nk_solver = nk.centrality.ApproxElectricalCloseness(self.G, k)
			nk_solver.run()
			vertice = np.array(nk_solver.getDiagonal()).argsort()[::-1][:k]
		else:
			vertice = None
		return vertice



def topk_test(measure, file, weighted, k_value, trial=1):
	name = file[file.rindex('/') + 1: file.rindex('.')]
	G = graphfile_loader(file)
	size = (len(G.nodes()), len(G.edges()))
	mean_fmk_time, mean_nkk_time, mean_gtk_time = 0, 0, 0
	mean_fm_nDCG, mean_nk_nDCG = 0, 0
	for t in range(trial):
		fmcl, nkit, gt = FMCL(measure, G), NkHelper(measure, G, weighted), Centrality(measure, G)
		gt_centralities, gt_topkV, gt_time, gtk_time = gt.topk_vertices(k_value)
		nk_topkV, nkk_time = nkit.topk_vertices(k_value)
		fm_topkV, fmk_time = fmcl.topk_vertices(k_value)
		if nk_topkV is None:
			return None
		mean_fmk_time, mean_nkk_time, mean_gtk_time = mean_fmk_time + fmk_time, mean_nkk_time + nkk_time, mean_gtk_time + gtk_time
		mean_fm_nDCG += nDCG(fm_topkV, gt_topkV, gt_centralities)
		mean_nk_nDCG += nDCG(nk_topkV, gt_topkV, gt_centralities)
	mean_fmk_time, mean_nkk_time, mean_gtk_time = round(mean_fmk_time / trial, 4), round(mean_nkk_time / trial, 4), round(mean_gtk_time / trial, 4)
	mean_fm_nDCG, mean_nk_nDCG = round(mean_fm_nDCG / trial, 4), round(mean_nk_nDCG / trial, 4)
	return [name, size, mean_gtk_time, mean_nkk_time, mean_fmk_time, mean_nk_nDCG, mean_fm_nDCG]


def topk_tests(measure, path, weighted, k_value, trial=1, outfile='table.tex'):
	files = glob.glob(path + '*.col')
	files.sort()
	results = []
	for file in files:
		try:
			result = topk_test(measure, file, weighted, k_value, trial)
			if result is None:
				return
			results.append(result)
			print(result)
		except Exception as e:
			print(e)
	resutls_latex(results, ['Instance', 'Size', 'NetworkX', 'Networkit', 'FastMap', 'NK nDCG', 'FM nDCG'], outfile)



if __name__ == '__main__':
	k_value = 1
	measures = ['clo', 'har', 'cfc']
	datasets = ['DIMACS', 'wDIMACS', 'TSP', 'movingAI', 'movingAI2', 'Tree', 'SmallWorld']
	is_weighted = [False, True, True, False, False, True, False]


	measures = ['clo']#, 'har', 'cfc']
	datasets = ['Lollipop']
	is_weighted = [True]

	for measure in measures:
		for dataset, weighted in zip(datasets, is_weighted):
			path, table_file = 'datasets/' + dataset + '/', 'nktables/' + dataset + '_' + measure +'.tex'
			#table_file = 'test.tex'
			topk_tests(measure, path, weighted, k_value, 1, table_file)


