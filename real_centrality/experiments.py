import numpy as np
import glob
from baseline import RealCentra
import sys
sys.path.append("..")
from makedata import graphfile_loader
from fmcl import FMCL, nDCG, resutls_latex



def real_topk_test(file, frac, measure, k, trial=1):
	name = file[file.rindex('/') + 1: file.rindex('.')]
	G = graphfile_loader(file)
	size = (len(G.nodes()), len(G.edges()))
	metrics = 'dijkstra' if measure != 'cfc' else 'paspd'
	ave_apsp_time, ave_apsp_nDCG = 0, 0
	ave_dh_time, ave_dh_nDCG = 0, 0
	ave_fm_time, ave_fm_nDCG = 0, 0
	ave_fmcl_time, ave_fmcl_nDCG = 0, 0
	ave_fmcl2_time, ave_fmcl2_nDCG = 0, 0
	for t in range(trial):
		real_V = np.sort(np.random.choice(G.nodes, int(size[0] * frac), False))
		apsp = RealCentra(G, real_V, 'APSP', metrics)
		dh, fm = RealCentra(G, real_V, 'DH', metrics), RealCentra(G, real_V, 'FM', metrics)
		fmcl = FMCL(measure, G)
		apsp_topkV, apsp_time, apsp_scores = apsp.topk_vertices(measure, k)
		dh_topkV, dh_time, _ = dh.topk_vertices(measure, k)
		fm_topkV, fm_time, _ = fm.topk_vertices(measure, k)
		fmcl_topkV, fmcl_time = fmcl.real_topk_vertices(real_V, k, 'adp')
		fmcl2_topkV, fmcl2_time = fmcl.real_topk_vertices(real_V, k, 'sub')
		ave_apsp_time += apsp_time
		ave_dh_time, ave_dh_nDCG = ave_dh_time + dh_time, ave_dh_nDCG + nDCG(dh_topkV, apsp_topkV, apsp_scores)
		ave_fm_time, ave_fm_nDCG = ave_fm_time + fm_time, ave_fm_nDCG + nDCG(fm_topkV, apsp_topkV, apsp_scores)
		ave_fmcl_time, ave_fmcl_nDCG = ave_fmcl_time + fmcl_time, ave_fmcl_nDCG + nDCG(fmcl_topkV, apsp_topkV, apsp_scores)
		ave_fmcl2_time, ave_fmcl2_nDCG = ave_fmcl2_time + fmcl2_time, ave_fmcl2_nDCG + nDCG(fmcl2_topkV, apsp_topkV, apsp_scores)
	ave_apsp_time = round(ave_apsp_time / trial, 2)
	ave_dh_time, ave_dh_nDCG = round(ave_dh_time / trial, 2), round(ave_dh_nDCG / trial, 4)
	ave_fm_time, ave_fm_nDCG = round(ave_fm_time / trial, 2), round(ave_fm_nDCG / trial, 4)
	ave_fmcl_time, ave_fmcl_nDCG = round(ave_fmcl_time / trial, 2), round(ave_fmcl_nDCG / trial, 4)
	ave_fmcl2_time, ave_fmcl2_nDCG = round(ave_fmcl2_time / trial, 2), round(ave_fmcl2_nDCG / trial, 4)
	return ([name, size, ave_apsp_time, ave_dh_time, ave_fm_time, ave_fmcl_time, ave_fmcl2_time,
		ave_dh_nDCG, ave_fm_nDCG, ave_fmcl_nDCG, ave_fmcl2_nDCG])


def real_topk_tests(path, frac, measure, k, trial=1, outfile='table.tex'):
	files = glob.glob(path + '*.col')
	files.sort()
	results = []
	for file in files:
		try:
			result = real_topk_test(file, frac, measure, k, trial)
			results.append(result)
			print(result)
		except Exception as e:
			print(e)
	resutls_latex(results,
		['Instance', 'Size', 'APSP', 'DH', 'FM', 'FMCL', 'FMCL2', 'DH nDCG', 'FM nDCG', 'FMCL nDCG', 'FMCL2 nDCG'], outfile)



if __name__ == '__main__':
	frac, k, trial = 0.5, 10, 1
	measures = ['clo', 'har', 'eig']
	datasets = ['DIMACS', 'wDIMACS', 'TSP', 'Tree', 'SmallWorld', 'movingAI', 'movingAI2']

	measures = ['cfc']
	datasets = ['movingAI', 'movingAI2']

	for measure in measures:
		for dataset in datasets:
			path, table_file = '../datasets/' + dataset + '/', 'tables/' + dataset + '_' + measure +'.tex'
			#\table_file = 'test.tex'
			real_topk_tests(path, frac, measure, k, trial, table_file)