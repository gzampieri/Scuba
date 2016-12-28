
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
from sklearn.cross_validation import KFold
import input_output as io




class ModelSelection():
	""" Model selection on the regularization parameter(s). 
	
	Parameters and Attributes:
	strategy -- Strategy to select the regularization parameter of UEasyMKL.
			    Options: - 'grid' to choose grid search plus cross-validation
						 - 'random' to choose random search plus cross-validation
	n_ iter -- Number of parameter values to try for model selection.
	n_fold -- Number of folds for cross-validation in model selection.
	cv_measure -- Measure to evaluate cross-validation.
	train_genes -- List of training genes.
	test_genes -- List of test genes.
	train_labels -- List of lists of training labels for each round of cross-validation.
	test_labels -- List of lists of test labels for each round of cross-validation.
	kernels -- List of kernel matrices.
	easy -- Object yielding a list of scores for genes.
	lambdas_p -- List of values of lambda_p to try.
	best_lambda_p -- Value of lambda_p selected.
	"""

	def __init__(self, easy, strategy, n_iter, n_fold, cv_measure):

		self.strategy = strategy
		self.n_iter = n_iter
		self.n_fold = n_fold
		self.cv_measure = cv_measure
		self.train_genes = None
		self.test_genes = None
		self.train_labels = None
		self.test_labels = None
		self.kernels = None
		self.easy = easy
		if self.strategy == 'random':
			self.lambdas_p = [np.random.uniform(0,1) for i in range(self.n_iter)]
		elif self.strategy == 'grid':
			self.lambdas_p = [float(i)/(self.n_iter-1) for i in range(self.n_iter-1)] + [1.0]
		else:
			raise ValueError('Specify a correct model selection strategy! (grid, random)')
		self.best_lambda_p = None



	def get_best_lambda(self):
		""" Abstract method. """
		
		return



	def set_kernel_list(self, kernel_list):
		""" Set list of kernels. """
		
		self.kernels = kernel_list
		return self



	def set_folds(self, genes, labels):
		""" Generate folds for cross-validation. """
		
		self.train_genes = genes
		self.test_list = genes
		self.train_labels = []
		self.test_labels = []
		
		pos_indeces = [i for i in range(len(labels)) if labels[i] == 1]
		
		if self.n_fold == 1:
			kf = KFold(len(pos_indeces), len(pos_indeces))
		else:
			kf = KFold(len(pos_indeces), self.n_fold)
		
		for train_index, test_index in kf:
			ltrain = [-1]*len(genes)
			for i in train_index:
				ltrain[pos_indeces[i]] = 1
			self.train_labels.append(ltrain)
			ltest = [-1]*len(genes)
			for i in test_index:
				ltest[pos_indeces[i]] = 1
			self.test_labels.append(ltest)
		
		return



	def search(self):
		""" Abstract method. """
		
		return



	def fit(self):
		""" Abstract method. """
		
		return





class SingleSelection(ModelSelection):
	""" Model selection on a single regularization parameter. 
	
	Parameters and Attributes:
	strategy -- Strategy to select the regularization parameter of UEasyMKL.
			    Options: - 'grid' to choose grid search plus cross-validation
						 - 'random' to choose random search plus cross-validation
	n_ iter -- Number of parameter values to try for model selection.
	n_fold -- Number of folds for cross-validation in model selection.
	cv_measure -- Measure to evaluate cross-validation.
	train_genes -- List of training genes.
	test_genes -- List of test genes.
	train_labels -- List of lists of training labels for each round of cross-validation.
	test_labels -- List of lists of test labels for each round of cross-validation.
	kernels -- List of kernel matrices.
	easy -- Object yielding a list of scores for genes.
	lambdas_p -- List of values of lambda_p to try.
	best_lambda_p -- Value of lambda_p selected.
	"""
	
	def __init__(self, easy, strategy, n_iter, n_fold, cv_measure):
		ModelSelection.__init__(self, easy, strategy, n_iter, n_fold, cv_measure)
	


	def get_best_lambda(self):
		
		return self.best_lambda_p



	def search(self):
		""" Compute the average performance of easy over all folds of cross-validation and return the best value. 
		
		Return:
		average_performance -- Best average performance over all folds of cross-validation.
		"""
		
		score_list = []
		if len(self.train_genes) == len(self.kernels[0]):
			for fold in xrange(len(self.train_labels)):
				fold_scores = []
				for lambda_p in self.lambdas_p:
					scores, weights = self.fit(self.train_labels[fold], self.kernels, None, False, True, lambda_p)
					fold_scores.append(scores)
				score_list.append(fold_scores)
			
		else:
			train_kernel_list = io.extract_submatrices(self.train_genes, self.train_genes, self.kernels)
			self.easy.set_Ktr_list(train_kernel_list, False)
			for fold in xrange(len(self.train_labels)):
				fold_scores = []
				for lambda_p in self.lambdas_p:
					scores, weights = self.fit(self.train_labels[fold], train_kernel_list, None, False, False, lambda_p)
					fold_scores.append(scores)
				score_list.append(fold_scores)
		
		# Compute average performance for each value of lambda
		average_performance_list = np.array([])
		for l in xrange(len(self.lambdas_p)):
			performance_list = np.array([])
			for fold in xrange(self.n_fold):
				test_indeces = [i for i in range(len(self.train_labels[fold])) if self.train_labels[fold][i] == -1]
				test_scores = np.take(score_list[fold][l], test_indeces)
				test_labels = np.take(self.test_labels[fold], test_indeces)
				performance_list = np.append(performance_list, [self.cv_measure(test_labels, test_scores)])
			average_performance_list = np.append(average_performance_list, np.mean(performance_list))
		
		self.best_lambda_p = self.lambdas_p[np.argmax(average_performance_list)]
		
		return np.max(average_performance_list)



	def fit(self, train_labels, train_kernel_list, test_kernel_list = None, reset_kernels = False, large_data = False, lambda_p = 0.1):
		""" Return easy scores and kernels weights. 
		
		Parameters:
		train_labels -- List of labels for training.
		train_kernel_list -- List of kernels for training.
		test_kernel_list -- List of kernels for testing.
		reset_kernels -- Boolean variable, indicates if it required to set up easy.
		large_data -- Boolean variable, indicates if data passed has to be handled as heavy.
		lambda_p -- Easy regularization parameter.
		
		Return:
		scores -- List of scores.
		weights -- List of learned linear coefficients for kernels.
		"""
		
		self.easy.set_lambda(lambda_p)
		if reset_kernels:
			self.easy.set_Ktr_list(train_kernel_list, large_data)
		
		weights = self.easy.train(train_kernel_list, train_labels)
		scores = self.easy.rank(test_kernel_list)
		
		return scores, weights





class DoubleSelection(ModelSelection):
	""" Model selection on two regularization parameters. 
	
	Parameters and Attributes:
	strategy -- Strategy to select the regularization parameter of UEasyMKL.
			    Options: - 'grid' to choose grid search plus cross-validation
						 - 'random' to choose random search plus cross-validation
	n_ iter -- Number of parameter values to try for model selection.
	n_fold -- Number of folds for cross-validation in model selection.
	cv_measure -- Measure to evaluate cross-validation.
	train_genes -- List of training genes.
	test_genes -- List of test genes.
	train_labels -- List of lists of training labels for each round of cross-validation.
	test_labels -- List of lists of test labels for each round of cross-validation.
	kernels -- List of kernel matrices.
	easy -- Object yielding a list of scores for genes.
	lambdas_p -- List of values of lambda_p to try.
	lambdas_n -- List of values of lambda_n to try.
	best_lambda_p -- Value of lambda_p selected.
	best_lambda_n -- Value of lambda_n selected.
	"""

	def __init__(self, easy, strategy, n_iter, n_fold, cv_measure):
		ModelSelection.__init__(self, easy, strategy, n_iter, n_fold, cv_measure)
		if self.strategy == 'random':
			self.lambdas_n = [np.random.uniform(0,1) for i in range(self.n_iter)]
		elif self.strategy == 'grid':
			self.lambdas_n = [float(i)/(self.n_iter-1) for i in range(self.n_iter-1)] + [1.0]
		else:
			raise ValueError('Specify a correct model selection strategy! (grid, random)')
		self.best_lambda_n = None



	def get_best_lambda(self):
		
		return self.best_lambda_p, self.best_lambda_n



	def search(self):
		""" Compute the average performance of easy over all folds of cross-validation and return the best value. 
		
		Return:
		average_performance -- Best average performance over all folds of cross-validation.
		"""
		
		score_list = []
		if len(self.train_genes) == len(self.kernels[0]):
			for fold in xrange(len(self.train_labels)):
				fold_scores = []
				for lambda_p in self.lambdas_p:
					for lambda_n in self.lambdas_n:
						scores, weights = self.fit(self.train_labels[fold], self.kernels, None, False, True, lambda_p, lambda_n)
						fold_scores.append(scores)
				score_list.append(fold_scores)
		
		else:
			train_kernel_list = io.extract_submatrices(self.train_genes, self.train_genes, self.kernels)
			self.easy.set_Ktr_list(train_kernel_list, False)
			for fold in xrange(len(self.train_labels)):
				fold_scores = []
				for lambda_p in self.lambdas_p:
					for lambda_n in self.lambdas_n:
						scores, weights = self.fit(self.train_labels[fold], train_kernel_list, None, False, False, lambda_p, lambda_n)
						fold_scores.append(scores)
				score_list.append(fold_scores)
		
		average_performance_list = np.array([])
		for l in xrange(len(self.lambdas_p)*len(self.lambdas_n)):
			performance_list = np.array([])
			for fold in xrange(self.n_fold):
				test_indeces = [i for i in range(len(self.train_labels[fold])) if self.train_labels[fold][i] == -1]
				test_scores = np.take(score_list[fold][l], test_indeces)
				test_labels = np.take(self.test_labels[fold], test_indeces)
				performance_list = np.append(performance_list, [self.cv_measure(test_labels, test_scores)])
			average_performance_list = np.append(average_performance_list, np.mean(performance_list))

		self.best_lambda_p = self.lambdas_p[np.argmax(average_performance_list) / len(self.lambdas_n)]
		self.best_lambda_n = self.lambdas_n[np.argmax(average_performance_list) % len(self.lambdas_n)]
		
		return np.max(average_performance_list)



	def fit(self, train_labels, train_kernel_list = None, test_kernel_list = None, reset_kernels = False, large_data = False, lambda_p = 0.1, lambda_n = 0.5):
		""" Return easy scores and kernels weights. 
		
		Parameters:
		train_labels -- List of labels for training.
		train_kernel_list -- List of kernels for training.
		test_kernel_list -- List of kernels for testing.
		reset_kernels -- Boolean variable, indicates if it required to set up easy.
		large_data -- Boolean variable, indicates if data passed has to be handled as heavy.
		lambda_p -- First easy regularization parameter.
		lambda_n -- Second easy regularization parameter.
		
		Return:
		scores -- List of scores.
		weights -- List of learned linear coefficients for kernels.
		"""
		
		self.easy.set_lambda(lambda_p, lambda_n)
		if reset_kernels:
			self.easy.set_Ktr_list(train_kernel_list, large_data)
		
		weights = self.easy.train(train_kernel_list, train_labels)
		scores = self.easy.rank(test_kernel_list)
		
		return scores, weights

