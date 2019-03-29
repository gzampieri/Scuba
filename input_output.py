
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


import pandas as pd
import numpy as np


def load_all_genes(kernel_path):
	""" Load the full list of genes with corresponding indeces. 
	
	Parameters:
	kernel_path_list -- List of kernel paths.
	
	Return:
	all_genes_dict -- Dictionary whose keys are genes represented by kernels and whose values are corresponding indeces.
	"""
	with open(kernel_path,'r') as f:
		kernel_path_list = [line.rstrip() for line in f.readlines()]
	
	all_genes_list = pd.DataFrame()
	for path in kernel_path_list:
		l = pd.read_table(path.split('_Matrix')[0] + '_Index.txt', sep=' ', header=None)
		all_genes_list = pd.concat([all_genes_list, l])
	
	all_genes_list = list(pd.unique(all_genes_list[0]))
        all_genes_dict = {all_genes_list[i] : i for i in xrange(len(all_genes_list))}
	
        return all_genes_dict



def load_kernels(kernel_path, all_genes_dict):
	""" Load the list of kernels. 
	
	Parameters:
	kernel_path -- Path of the file containing kernels' paths.
        all_genes_dict -- Dictionary whose keys are genes represented by kernels and whose values are corresponding indeces.
	
	Return:
	kernel_name_list -- List of kernel file names.
	kernel_list -- List of pandas data frames, each one representing a kernel matrix.

        Notes:
        If the gene set represented by the kernel matrices do not fully overlap, these are expanded to include the global set with the kernel average replacing missing values.
	"""
		
	with open(kernel_path,'r') as f:
		kernel_path_list = [line.rstrip() for line in f.readlines()]
	
        kernel_list = []
        for path in kernel_path_list:
                #Load kernel matrix
                kernel_matrix = np.load(path)
                kernel_matrix = pd.DataFrame(data=kernel_matrix)
		
                #Dictionary of indeces related to the kernel matrix and associated to gene name
                gene_index_dict = pd.read_table(path.split('_Matrix')[0] + '_Index.txt', sep=' ', names='gene_name gene_index'.split(), index_col='gene_index')
                
		# Calculate mean kernel value
		kernel_mean = kernel_matrix.sum().sum() / kernel_matrix.size
                # Add gene_name column to kernel matrix in order to perform the following reindex
                kernel_matrix['gene_name'] = gene_index_dict.gene_name
                # Reindex with gene_name
                kernel_matrix = kernel_matrix.set_index('gene_name')
                # Change name of columns with gene_name
                kernel_matrix.columns = [kernel_matrix.index[i] for i in xrange(len(kernel_matrix))]
                # 1) Reindex to all genes list (rows)
                # 2) Transpose matrix (rows and columns are inverted)
                # 3) Reindex to all genes list (columns)
                # 4) Replace NaN values with 0
		final_kernel_matrix = kernel_matrix.reindex(all_genes_dict.keys()).transpose().reindex(all_genes_dict.keys()).fillna(kernel_mean)
		
		# Normalize matrix to get unitary diagonal sum
		mean_trace = np.mean(np.diag(final_kernel_matrix))
		final_kernel_matrix = final_kernel_matrix.divide(mean_trace)
		
		# Update the list of kernels
                kernel_list.append(final_kernel_matrix)
        
	
	kernel_name_list = [path.split('/')[-1] for path in kernel_path_list]
        
        return kernel_name_list, kernel_list



def extract_submatrices(row_indeces, column_indeces, kernel_list):
	""" Get list of kernel submatrices. 
	
	Parameters:
	row_indeces -- List of row indeces.
	column_indeces -- List of column indeces.
	kernel_list -- List of pandas data frames representing kernel matrices.
	
	Return:
	subkernel_list -- List of pandas data frames extracted from input matrices.
	"""
	
	subkernel_list = []
	for K in kernel_list:
		submatrix = K.iloc[row_indeces, column_indeces]
		subkernel_list.append(submatrix)
	
	return subkernel_list



def load_multiple_indeces(path, all_genes_dict):
	""" Load multiple lists of genes from files.
	
	Parameters:
	path -- Path of the list of gene lists paths.
	all_genes_dict -- Dictionary whose keys are genes represented by kernels and whose values are corresponding indeces.
	
	Return:
	genes -- List of gene indeces lists.
	"""
	
	genes = []
	with open(path,'r') as f:
		path_list = [line.rstrip() for line in f.readlines()]
		for p in path_list:
			genes.append(load_indeces(p, all_genes_dict))
	
	return genes



def load_indeces(path, all_genes_dict):
	""" Load a list of genes from file.
	
	Parameters:
	path -- Path of the list of genes.
	all_genes_dict -- Dictionary whose keys are genes represented by kernels and whose values are corresponding indeces.
	
	Return:
	indeces -- Lists of gene indeces.
	"""
	
	with open(path,'r') as f:
		genes = [line.rstrip() for line in f.readlines()]
	indeces = []
	discarded = []
	for gene in genes:
		if gene in all_genes_dict.keys():
                        indeces.append(all_genes_dict[gene])
		else:
			discarded.append(gene)
	
	print path, ': ', len(discarded), 'genes not present in kernels'
	
	return indeces



def load_targets(path, all_genes_dict):
	""" Load targets from files.
	
	Parameters:
	path -- Path of the list of seed gene lists paths.
	all_genes_dict -- Dictionary whose keys are genes represented by kernels and whose values are corresponding indeces.
	
	Return:
	target_names -- -- List of target gene names.
	target_indeces -- List of target gene indices.
	
	Notes:
	Seeds' file names are the names of the corresponding target gene for each seed group.
	"""
	
	with open(path,'r') as f:
		path_list = [line.rstrip() for line in f.readlines()]
	target_names = [p.split('/')[-1] for p in path_list]
	target_indeces = [all_genes_dict[gene] for gene in target_names]
	
	return target_names, target_indeces



def save_rank(path, scores, genes, all_genes_dict):
	""" Save to file a gene rank.
	
	Parameters:
	scores -- List of scores.
	genes -- List of genes
	all_genes_dict -- Dictionary whose keys are genes represented by kernels and whose values are corresponding indeces.
	path -- Output path.
	"""
	
	rank = sorted(zip(scores, genes), reverse=True)
	with open(path+'_rank.txt', 'w') as f:
			for score, gene in rank:
				f.write(all_genes_dict.keys()[all_genes_dict.values().index(gene)] + "\t" + str(score) + "\n")



def save_details(path, best_lambda, kernel_names, weights):
	""" Save to file the values of learned parameters.
	
	Parameters:
	kernel_names -- List of kernels' names.
	path -- Output path.
	best_lambda -- Learned value of lambda.
	weights -- List of learned linear coefficients for kernels.
	file_name -- Label for the output files (empty string by default)
	"""
	
	with open(path+"_details.txt",'w') as f:
            f.write("Regularization hyper-parameter value:\n\t" + str(best_lambda) + "\n\n")
		f.write("Weights of kernels:\n")
		for i in range(len(weights)):
			f.write("\t" + kernel_names[i] + "\t" + str(weights[i]) + "\n")



def save_results(path, results):
	""" Save to file the overall performance.
	
	Parameters:
	path -- Output path.
	results -- Dictionary containing the results for each quantity computed.
	"""
	
	with open(path+"_results.txt",'w') as f:
		for measure in sorted(results):
			f.write(measure + "\t\t" + str(results[measure]) + "\n")




