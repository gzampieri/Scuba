
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


@author: Michele Donini

"""


from cvxopt import matrix, solvers
import numpy as np
import pandas as pd


class EasyMKL():
	""" EasyMKL algorithm.
	
	The parameter lambda_p has to be validated from 0 to 1.
	
	For more information:
	EasyMKL: a scalable multiple kernel learning algorithm
	by Fabio Aiolli and Michele Donini

	Paper @ http://www.math.unipd.it/~mdonini/publications.html
	
	Attributes:
	Ktr_sum -- Sum of training kernel matrices.
	q -- List of average of entries of Ktr_sum.
	large_data -- Boolean variable, indicates if training data is heavy.
	lambda_p -- Regularization parameter (float, range [0,1]).
	tracenorm -- Boolean variable, indicates whether kernels traces have to be normalized.
	labels -- List of labels of training examples.
	gammas -- List of weights for examples.
	weights -- List of weights for kernel matrices.
	r -- List of scores for training examples.
	"""

	def __init__(self, Ktr_list = None, lambda_p = 0.1, tracenorm = True, large_data = False):

		if Ktr_list == None:
			self.Ktr_sum = None
			self.q = None
			self.large_data = large_data
		else:
			self.set_Ktr_list(Ktr_list, large_data)
		self.lambda_p = lambda_p
		self.tracenorm = tracenorm
		self.labels = None
		self.gammas = None
		self.weights = None
		self.r = None



	def set_lambda(self, lambda_p):
		""" Set lambda_p. """
		
		self.lambda_p = lambda_p
		return self



	def set_Ktr_list(self, Ktr_list, large_data):
		""" Set Ktr_sum, q and large_data. """
		
		if large_data:
			self.Ktr_sum = self.sum_kernels(Ktr_list)
			self.q = self.sum_rows(self.Ktr_sum)
			self.large_data = True
		else:
			self.Ktr_sum = self.sum_kernels(self.normalize_kernels(Ktr_list))
			self.q = None
			self.large_data = False
		return



	def get_trace(self, K):
		""" Get trace of a matrix. """
		
		return sum([K.iloc[i,i] for i in range(K.shape[0])]) / K.shape[0]



	def normalize_kernels(self, K_list):
		""" Divide kernel matrices by their traces. """
		
		traces = [self.get_trace(K) for K in K_list]
		
		# Divide each matrix by its trace
		if self.tracenorm:
			for matrix_idx in range(len(K_list)):
				K_list[matrix_idx] = K_list[matrix_idx].divide(traces[matrix_idx])
		
		return K_list


	def sum_kernels(self, K_list, weights = None):
		""" Compute sum of kernels.
	
		Parameters:
		K_list -- List of kernel matrices.
		weights -- List of linear coefficients (non specified by default).
		
		Return:
		A -- Kernel matrix created by summing all the kernels.
		
		Notes:
		If weights is None, the sum is the ordinary sum.
		Otherwise, each kernel is multiplied by its corresponding weight and then summed.
		"""
		
		M = K_list[0].shape[0]
		N = K_list[0].shape[1]             
		A = pd.DataFrame(data=np.zeros((M,N)), index=K_list[0].index, columns=K_list[0].columns)
		
		if weights == None:
			for K in K_list:
				A = A.add(K)
		else:
			for w,K in zip(weights, K_list):
				A = A.add(w*K)
		
		return A



	def sum_rows(self, K):
		""" Get average of rows values. """ 
		
		q = K.sum(axis=0).divide(K.shape[0])
		return q



	def train(self, Ktr_list, labels):
		""" Train the model.
		
		Parameters:
		Ktr_list --- List of kernel matrices for training examples.
		labels --- List of labels of training examples.
		"""
		
		# Set labels as +1/-1
		set_labels = set(labels)
		if len(set_labels) != 2:			
			raise ValueError('The different labels are not 2')
		elif (-1 in set_labels and 1 in set_labels):
			self.labels = pd.Series(data=labels, index=self.Ktr_sum.index)
		else:
			poslab = np.max(list(set_labels))
			self.labels = pd.Series(data=[1 if i==poslab else -1 for i in labels], index=self.Ktr_sum.index)
		
		Xp = list(self.labels[self.labels == 1].index)
		Xn = list(self.labels[self.labels == -1].index)
		
		# Compute gammas
		self.gammas = self.komd_train(self.Ktr_sum, self.labels, Xp, Xn)
		
		# Compute weights
		self.weights = self.get_weights(Ktr_list, self.gammas, self.labels, Xp, Xn)
		
		# Compute final gammas
		Ktr_weighted_sum = self.sum_kernels(Ktr_list, self.weights)
		self.gammas = self.komd_train(Ktr_weighted_sum, self.labels, Xp, Xn)
		
		yg =  self.gammas*self.labels
		self.r = Ktr_weighted_sum.dot(yg).values
		
		return self.weights



	def komd_train(self, K, labels, Xp, Xn):
		""" Train KOMD. """
		
		YY = pd.DataFrame(data=np.diag(list(labels)), index=self.Ktr_sum.index, columns=self.Ktr_sum.columns)
		P = matrix(2*((1.0-self.lambda_p) * YY.dot(K).dot(YY).as_matrix() + np.diag([self.lambda_p]*len(labels))))
		q = matrix([0.0]*len(labels))
		G = -matrix(np.diag([1.0]*len(labels)))
		h = matrix([0.0]*len(labels),(len(labels),1))
		A = matrix([[1.0 if lab==+1 else 0 for lab in labels],[1.0 if lab2==-1 else 0 for lab2 in labels]]).T
		b = matrix([[1.0],[1.0]],(2,1))
		
		solvers.options['show_progress']=False
		sol = solvers.qp(P,q,G,h,A,b)
		
		gammas = pd.Series(data=sol['x'], index=self.Ktr_sum.index)
		
		return gammas



	def get_weights(self, Ktr_list, gammas, labels, Xp, Xn):
		""" Get weights of kernels. """
		
		weights = []
		if self.large_data:
			X = Xp + [Xn[np.random.randint(len(Xn))] for i in range(int((len(Xp)+len(Xn))/10))]
                        g = gammas.loc[X]
			l = labels.loc[X]
			yg = g*l
			for K in Ktr_list:
				k = K.loc[X,X]
				b = yg.dot(k).dot(yg)
				weights.append(b)
		else:
			yg =  gammas*labels
			for K in Ktr_list:
				b = yg.dot(K).dot(yg)
				weights.append(b)
		
		norm2 = sum([w for w in weights])
		weights = [w / norm2 for w in weights]
		
		return weights



	def rank(self, Ktest_list = None):
		""" Compute the probability distribution for positive examples.
		
		Parameters:
		Ktest_list --- List of kernel matrices of test examples.
		
		Return:
		r -- List of scores for test examples.
		"""
		
		if self.weights == None:
			raise ValueError('EasyMKL has to be trained first!')
		
		if Ktest_list == None:
			r = self.r
		else:
			yg =  self.gammas*self.labels
			Ktest_weighted_sum = self.sum_kernels(Ktest_list, self.weights)
			r = Ktest_weighted_sum.dot(yg).as_matrix()
		
		return list(r)        





class Scuba(EasyMKL):
	""" EasyMKL algorithm with unbalanced regularization.
	
	Attributes:	
	lambda_p -- Regularization parameter for positive examples (float, range [0,1]).
	"""
	
	def __init__(self, Ktr_list = None, lambda_p = 0.1, tracenorm = True, large_data = False):
		EasyMKL.__init__(self, Ktr_list, lambda_p, tracenorm, large_data)



	def komd_train(self, K, labels, Xp, Xn):
		""" Train KOMD with unbalanced regularization. """
		
		P = matrix((1.0-self.lambda_p)*K.loc[Xp,Xp].values + np.diag([self.lambda_p]*len(Xp)))
		if self.large_data:
			q = matrix(-(1.0-self.lambda_p)*self.q.loc[Xp].values)
		else:
			q = matrix(-(1.0-self.lambda_p)*K.loc[Xp,Xn].values.dot([1.0/len(Xn)]*len(Xn)))
		G = -matrix(np.diag([1.0]*len(Xp)))
		h = matrix(0.0, (len(Xp),1))
		A = matrix(1.0, (len(Xp),1)).T
		b = matrix(1.0)
		
		solvers.options['show_progress']=False
		sol = solvers.qp(P,q,G,h,A,b)
		
		gammas = pd.Series(data=[1.0/len(Xn)]*(len(Xp)+len(Xn)), index=self.Ktr_sum.index)
		for idx,v in enumerate(Xp):
			gammas[v] = sol['x'][idx]
		
		return gammas


