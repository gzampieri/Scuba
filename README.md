
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

Scuba is usable from command line by typing `python launcher.py [arguments]`. For information on the arguments, type `python launcher.py --help` or see the documentation in [launcher.py](launcher.py).

Folder [Toy data](Toy_data/) contains simple inputs and corresponding outputs that can be used to check code functioning.

Script [compute_kernel.py](compute_kernel.py) allows creating kernel matrices from available datasets through a few kernel functions.
For instance, starting from a gene network adjacency matrix such as [dataset1_adjacencyMatrix.txt](Toy_data/dataset1_adjacencyMatrix.txt), run:

```python compute_kernel.py --input-matrix <path to file>/dataset1_adjacencyMatrix.txt --kernel-function MDK --output <path to file>/dataset1_Matrix_MDK3 -p 3```

to get [dataset1_Matrix_MDK3.npy](Toy_data/dataset1_Matrix_MDK3.npy). Analogously, for a numerical feature matrix such as [dataset2_featureMatrix.txt](Toy_data/dataset2_featureMatrix.txt) the *RBF* function can be used.

To prioritize the whole set of genes represented by the kernel matrices, run:

```python launcher.py --kernel-list <path to file>/my_kernels.txt --disease-genes <path to file>/my_disease_genes.txt --output <path to file>/my_output```

or, optionally, if a list of candidate genes is available:

```python launcher.py --kernel-list <path to file>/my_kernels.txt --disease-genes <path to file>/my_disease_genes.txt --output <path to file>/my_output --candidate-genes <path to file>/my_candidates.txt```

Alternatively, to perform the so-called "unbiased" ([CAFA](https://www.nature.com/articles/nmeth.2340)-like) validation described in the paper, run:

```python launcher.py --kernel-list <path to file>/my_kernels.txt --disease-genes <path to file>/seeds_unbiased_Ensembl.txt --candidate-genes <path to file>/candidates_unbiased_Ensembl.txt --output <path to file>/my_output -f 4 -i 11 --test unbiased```

Parameters f and i (and model-selection) can be changed as needed, however note that f should not be greater than 4 as there is a test case with only 4 known disease genes.

If you use Scuba, please cite:
Zampieri, G., Van Tran, D., Donini, M., Navarin, N., Aiolli, F., Sperduti, A., & Valle, G. (2018). [Scuba: scalable kernel-based gene prioritization.](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2025-5) *BMC bioinformatics*, 19(1), 23.


# License

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
    with Scuba. If not, see http://www.gnu.org/licenses/.

