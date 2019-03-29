
"""
    Scuba - Gene prioritization.
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


# Gene prioritization

In molecular biology, gene prioritization is the task of identifying the most promising genes to submit to experimental studies.
It is an important step to narrow down the target of wet-lab analyses such as validations and screenings, in order to optimize their results.


# Scuba

Scuba is a computational approach for gene prioritization, based on the integration of gene functional associations.
In particular, it focuses on the identification of novel putative associations between genes and genetic disorders.

Let us consider the following entities:
* a group of genes known to be related to a disorder
* a group of candidate genes to be prioritized, that can even be the whole set of human genes
* a collection of functional associations among genes, deriving from multiple data sources
The goal of Scuba is to prioritize the group of candidate genes by exploiting their functional connections with the group of known genes.
The output is a rank of the candidate genes, ordered from the most to the least promising.


# System requirements

Running this code may be a computationally intensive procedure. Validations mentioned in the documentation were performed on a computer cluster.


# Dependencies

This code was tested using Python 2.7.14, Numpy 1.14.2, Pandas 0.23.4 and Scikit-learn 0.19.1. 


# Usage

Scuba is usable from command line by typing `python launcher.py [arguments]`.

For information on arguments, type `python launcher.py --help` or see the documentation in [launcher.py](https://github.com/gzampieri/Scuba/blob/master/launcher.py).

Mandatory arguments:
	kernel_list -- Path of a text file containing the list of kernel matrices' paths, one path per line.
				   The name of each file must contain the string '_Matrix'.
				   Example:
						../kernel_folder/kernel1_Matrix
						../kernel_folder/kernel2_Matrix
						../kernel_folder/kernel3_Matrix
				   Each kernel is a square, symmetric, positive semi-definite matrix, whose entries must be separated by a single space.
				   Kernel matrices can be computed using the script compute_kernel.py.
				   In the same folder of each kernel, there must be another file containing the index of each gene in the corresponding matrix, labelled by the string '_Index' (e.g. kernel1_Index).
				   The format is: gene on the first column, index on the second column (separated by a single space).
				   Example:
						A1BG 4502
						A1CF 1735
						A2BP1 9498
						A2M 2128

                                    Data sources do not strictly need to include the same sets of genes. If the gene sets represented by the kernel matrices do not fully overlap, these are expanded to include the global set and the missing values are replaced by the kernel average.
						
	disease_genes -- In the case a single prediction is desired, path of a text file containing the list of input disease genes (seed genes), one gene per line.
					 Example:
							FLG2
							DENND2D
							BAT2
							GBX1
							WDR6
							
					 In the case of validation, path of a text file containing the list of paths for input disease genes lists, one path per line.
					 Example:
							../seeds_folder/ACAD9
							../seeds_folder/ATF7IP
							../seeds_folder/BCL3
							
					 In this case, all files in the list must have the same format as in the case of single prediction (one gene per line).
				
	output -- Output folder path and file name.
			  Example:
					../results_folder/my_results
					
			  The ouput is: - a file for each seed list, containing the ranked scores for all genes represented by kernels, except the seed genes.
			                - a second file for each seed list, containing information on the learned parameters (regularization parameter of EasyMKL and weight of each kernel)
			                - in the case of unbiased evaluation, a file containing performance of Scuba


Optional arguments:
	candidate_genes -- In the case of single prediction, path of a text file containing the list of input disease genes (seed genes), one gene per line.
				       In the case of validation, path of a text file containing the list of paths for input disease genes lists, one path per line.
					   Format must be the same as for disease-genes, in both cases.
	
	test -- Type of evaluation to perform. To select the unbiased evaluation, specify 'unbiased'.
	
	model_selection -- Strategy to select the regularization parameter.
					   Options: - 'grid' to choose grid search plus cross-validation
					            - 'random' to choose random search plus cross-validation
					   Grid search is used by default.
	
	iterations -- Number of parameter values to try in model selection (5 by default).
	
	folds -- Number of folds for cross-validation in model selection (1 by default, corresponding to leave-one-out-cross-validation).
	
	measure -- Evaluation measure used for model selection.
			   Options: - 'auroc' to choose the area under the receiver-operating-characteristic curve (AUROC)
			   AUROC is used by default.


"""


import argparse
parser = argparse.ArgumentParser(description='Gene prioritization by heterogeneous kernel integration.')
parser.add_argument('--kernel-list', required=True, help='Path to the list of kernels paths.')
parser.add_argument('--disease-genes', required=True, help='Path to the list of input disease genes.')
parser.add_argument('--output', required=True, help='Path and name of the output files.')
parser.add_argument('--candidate-genes', default=None, help='Path to the list of input candidate genes.')
parser.add_argument('-t', '--test', default=None, help='Perform tests on benchmark validation settings. Options: "unbiased" for the unbiased evaluation (ignored by default).')
parser.add_argument('-a', '--algorithm', default='scuba', help='Options: "easy" for EasyMKL"; "scuba" for Scuba, used by default.')
parser.add_argument('--model-selection', default='grid', help='Strategy for selecting the regularization hyper-parameter(s). Options: "grid" for grid search and "random" for random search (grid by default).')
parser.add_argument('-i', '--iterations', type=int, default=5, help='Number of parameter values to try in model selection (5 by default).')
parser.add_argument('-f', '--folds', type=int, default=1, help='Number of folds of cross validation in model selection (leave-one-out by default).')
parser.add_argument('-m', '--cv-measure', default='auroc', help='Evaluation measure used in model selection. Options: "auroc" for area under the ROC curve, used by default.')
args = parser.parse_args()





def main():
	""" Run Scuba. 
		
	By input, it is possible to choose among two possibilities:
	- Single prediction: gene prioritization given a list of seed genes and optionally a list of candidate genes.
	- Unbiased evaluation test: validation of Scuba by the evaluation of Bornigen et al.
	
	Reference:
	Bornigen D et al. An unbiased evaluation of gene prioritization tools. Bioinformatics. 2012;28:3081-88.
	"""
	
	if args.algorithm == 'easy':
		from prioritizer import EasyMKL
		prior = EasyMKL()
	elif args.algorithm == 'scuba':
		from prioritizer import Scuba
		prior = Scuba()
	else:
		raise ValueError('Specify a correct algorithm! (easy, scuba)')
	
	if args.cv_measure == 'auroc':
		import evaluation as ev
		cv_measure = ev.AUROC
	else:
		raise ValueError('Specify a correct evaluation measure! (auroc)')
	
	if args.algorithm == 'easy' or args.algorithm == 'scuba':
		from model_selection import SingleSelection
		ms = SingleSelection(prior, args.model_selection, args.iterations, args.folds, cv_measure)
	
	if args.test == None:
		from task import Prediction
		task = Prediction(args.kernel_list, args.disease_genes, args.candidate_genes, args.output, ms)
	elif args.test == 'unbiased':
		from task import Validation
		task = Validation(args.kernel_list, args.disease_genes, args.candidate_genes, args.output, ms)
	else:
		raise ValueError('Specify a correct validation setting! (unbiased)')
	
	task.do()




if __name__ == "__main__":
	main()

