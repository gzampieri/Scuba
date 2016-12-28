

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

This code was tested using Python 2.7.3, Numpy 1.11.1, Pandas 0.18.1 and Scikit-learn 0.17.1. 


# Usage

Scuba is usable from command line by typing `python launcher.py [arguments]`.

For information on arguments, type `python launcher.py --help` or see the documentation in [launcher.py](https://github.com/gzampieri/Scuba/blob/master/launcher.py).



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

