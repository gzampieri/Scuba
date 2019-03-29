
"""
    This file is part of Scuba.
    Copyright (C) 2016 by Guido Zampieri, Dinh Tran, Michele Donini.

    Scuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    Scuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with Scuba. If not, see <http://www.gnu.org/licenses/>.


@author: Guido Zampieri

"""


import numpy as np
import input_output as io
import evaluation as ev



class Task():
	""" General type of task for Scuba. 
	
	Parameters and Attributes:
	kernel_path -- Path of kernel matrices.
	output_path -- Path for output files.
	ms -- Object of class ModelSelection.
	gene_indeces -- Dictionary whose keys are genes represented by kernels and whose values are corresponding indeces.
	kernel_names -- List of kernel file names.
	
	"""

	def __init__(self, kernel_path, output_path, ms):

		self.gene_indeces = io.load_all_genes(kernel_path)
		self.kernel_names, kernel_list = io.load_kernels(kernel_path, self.gene_indeces)
		self.output_path = output_path
		self.ms = ms
		self.ms.set_kernel_list(kernel_list)
		


	def do (self):
		""" Abstract method. """
		
		return



	def get_rank(self, genes, train_labels):
		""" Get scores to prioritize genes.

		Parameters:
		genes -- List of genes to rank.
		train_labels -- List of labels for training.
		"""
		
		self.ms.set_folds(genes, train_labels)
		self.ms.search()
		best_lambda = self.ms.get_best_lambda()
		
		if len(genes) == len(self.gene_indeces.keys()):
			if type(best_lambda) is float:
				scores, weights = self.ms.fit(train_labels, self.ms.kernels, None, False, best_lambda)
			else:
				scores, weights = self.ms.fit(train_labels, self.ms.kernels, None, False, best_lambda[0], best_lambda[1])
		else:
			train_kernel_list = io.extract_submatrices(genes, genes, self.ms.kernels)
			if type(best_lambda) is float:
				scores, weights = self.ms.fit(train_labels, train_kernel_list, None, False, best_lambda)
			else:
				scores, weights = self.ms.fit(train_labels, train_kernel_list, None, False, best_lambda[0], best_lambda[1])
		
		io.save_details(self.output_path, best_lambda, self.kernel_names, weights)

		return scores





class Prediction(Task):
	""" Gene prioritization by heterogeneous kernel integration. 
	
	Parameters and Attributes:
	Task -- Class representing a general type of task for Scuba.
	kernel_path -- Path of kernel matrices.
	seed_path -- Path of seed genes.
	candidate_path -- Path of candidate genes.
	output_path -- Path for output files.
	ms -- Object of class ModelSelection.
	seed_list -- List of seed genes.
	candidate_list -- List of candidate genes (empty by default).
	"""

	def __init__(self, kernel_path, seed_path, candidate_path, output_path, ms):
		Task.__init__(self, kernel_path, output_path, ms)
		
		self.seed_list = io.load_indeces(seed_path, self.gene_indeces)
		if candidate_path == None:
			self.candidate_list = None
		else:
			self.candidate_list = io.load_indeces(candidate_path, self.gene_indeces)
			



	def do(self):
		""" Prioritize genes and output results. 
		
		Notes:
		Training is always performed with all genes represented by kernel matrices, and all are prioritized.
		If a candidate set is provided, their ranking is saved in an additional file.
		"""
		
		train_genes = range(len(self.gene_indeces.keys()))
		train_labels = [-1]*len(self.gene_indeces.keys())
		for index in self.seed_list:
			train_labels[index] = 1
		self.ms.easy.set_Ktr_list(self.ms.kernels, True)
		scores = self.get_rank(train_genes, train_labels)
		
		test_indeces = [idx for idx in range(len(train_labels)) if train_labels[idx] == -1]
		test_genes = list(np.take(train_genes, test_indeces))
		test_scores = np.take(scores, test_indeces)
		io.save_rank(self.output_path, test_scores, test_genes, self.gene_indeces)

		if self.candidate_list != None:
			test_scores = np.take(scores, self.candidate_list)
			io.save_rank(self.output_path+'_candidates', test_scores, self.candidate_list, self.gene_indeces)
		
		return





class Validation(Task):
	""" Unbiased evaluation of Scuba. 
	
	Parameters and Attributes:
	Task -- Class representing a general type of task for Scuba.
	kernel_path -- Path of kernel matrices.
	seed_path -- Path of the list of seed genes lists' paths.
	candidate_path -- Path of the list of candidate genes lists' paths.
	output_path -- Path for output files.
	ms -- Object of class ModelSelection.
	seed_list -- List of seed genes lists.
	target_names -- List of test genes' names.
	target_list -- List of test genes to be retrieved.
	candidate_list -- List of candidate genes lists.
	
	Notes:
	Following the unbiased evaluation of Bornigen et al, the number of seed lists, target genes and candidate lists is 42.
	
	Reference:
	Bornigen D et al. An unbiased evaluation of gene prioritization tools. Bioinformatics. 2012;28:3081-88.
	"""

	def __init__(self, kernel_path, seed_path, candidate_path, output_path, ms):
		Task.__init__(self, kernel_path, output_path, ms)
		
		self.seed_list = io.load_multiple_indeces(seed_path, self.gene_indeces)
		self.target_names, self.target_list = io.load_targets(seed_path, self.gene_indeces)
		self.candidate_list = io.load_multiple_indeces(candidate_path, self.gene_indeces)



	def do(self):
		""" Perform the unbiased evaluation.
		
		Notes:
		Measures computed are: median of the normalized ranks, true positive rate at 5,10,30% of the rank and AUC.
		"""
		
		cs_targetpos = []
		gw_targetpos = []
		
		for i in xrange(len(self.target_list)):
			unordered_train_genes = self.seed_list[i]+self.candidate_list[i]
			unordered_train_labels = [1]*len(self.seed_list[i])+[-1]*len(self.candidate_list[i])
			train_genes = [t[0] for t in sorted(zip(unordered_train_genes, unordered_train_labels))]
			train_labels = [t[1] for t in sorted(zip(unordered_train_genes, unordered_train_labels))]
			
			test_indeces = [idx for idx in range(len(train_labels)) if train_labels[idx] == -1]
			test_genes = list(np.take(train_genes, test_indeces))
			test_labels = list(np.take(train_labels, test_indeces))
			test_labels[test_genes.index(self.target_list[i])] = 1
			
			scores = self.get_rank(train_genes, train_labels)
			scores = np.take(scores, test_indeces)
			
			cs_targetpos.append((ev.rank_pos(scores, test_labels), len(test_genes)))
			io.save_rank(self.output_path+'_'+self.target_names[i]+'_candidates', scores, test_genes, self.gene_indeces)

		# Compute results
		results = ev.evaluate(cs_targetpos)
		keys = results.keys()
		for key in keys:
			results['candidates '+key] = results.pop(key)
		

		self.ms.easy.set_Ktr_list(self.ms.kernels, True)
		
		for i in xrange(len(self.target_list)):
			train_genes = range(len(self.gene_indeces.keys()))
			train_labels = [-1]*len(self.gene_indeces.keys())
			for index in self.seed_list[i]:
				train_labels[index] = 1
			
			test_indeces = [idx for idx in range(len(train_labels)) if train_labels[idx] == -1]
			test_genes = list(np.take(train_genes, test_indeces))
			test_labels = list(np.take(train_labels, test_indeces))
			test_labels[test_genes.index(self.target_list[i])] = 1
			
			scores = self.get_rank(train_genes, train_labels)
			scores = np.take(scores, test_indeces)
			
			gw_targetpos.append((ev.rank_pos(scores, test_labels), len(test_genes)))
			io.save_rank(self.output_path+'_'+self.target_names[i], scores, test_genes, self.gene_indeces)

		results.update(ev.evaluate(gw_targetpos))
		
		io.save_results(self.output_path, results)

		return

