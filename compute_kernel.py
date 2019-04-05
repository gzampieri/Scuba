#!/usr/bin/env python

"""
@authors: Dinh Tran and Guido Zampieri

compute_kernel.py:
    This script computes kernel matrices of various type:
    Markov exponential diffusion kernel (MEDK), Markov diffusion kernel (MDK), regularized Laplacian kernel (RLK) and radial basis function kernel (RBF).


Usage:
    This script is usable from command line by typing `python compute_kernel.py [arguments]`.
    For information on arguments from command line, type `python compute_kernel.py --help`


Mandatory arguments:
    input_matrix -- 	Text file containing one of the following:
        			- adjacency matrix of an undirected graph, namely a squared symmetric matrix where entry A_ij represents the weight of the link between items (genes) i and j;
                   		- feature matrix, with samples (genes) along the rows and features along the columns.
	
    kernel_function -- Kernel function to use. Options: 
    				- Markov exponential diffusion kernel (MEDK);
    				- Markov diffusion kernel (MDK);
				- regularized Laplacian kernel (RLK);
                            	- radial basis function kernel (RBF).
	
    output_file -- Path and name of the output kernel matrix, which is saved as a npy file.


Optional arguments:
    kernel_parameter -- Value of the kernel parameter (float for MEDK and RLK; int for MDK).
			By default, it is equal to 0.04 for MEDK, 10 for MDK, 4.0 for RLK and 1.0 for RBF.


Dependencies:
    This code was tested using Python 2.7.14, Numpy 1.14.2 and Scipy 1.0.0.

"""



import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Computation of kernel matrices of various type: Markov exponential diffusion kernel (MEDK), Markov diffusion kernel (MDK), regularized Laplacian kernel (RLK), radial basis function kernel (RBF).')
parser.add_argument('--input-matrix', required=True, help='Path to a text file containing a graph adjacency matrix (of size num genes x num genes) or a feature matrix (of size num genes x num features).')
parser.add_argument('--kernel-function', required=True, help='Kernel function to use. Options: "MEDK", "MDK", "RLK", "RBF".')
parser.add_argument('--output-file', required=True, help='Path and name of the output kernel matrix, which is saved as a npy file.')
parser.add_argument('-p', '--kernel-parameter', type=float, help='Value of the kernel parameter (float for MEDK, RLK, and RBF; int for MDK).')
args = parser.parse_args()




def main():
    """ Save to file the kernel matrix specified in input. """
    
    A = np.loadtxt(args.input_matrix)
    
    if args.kernel_function == "MEDK":
        if args.kernel_parameter == None:
            K = get_MEDK(A)
        else:
            K = get_MEDK(A, args.kernel_parameter)
    elif args.kernel_function == "MDK":
        if args.kernel_parameter == None:
            K = get_MDK(A)
        else:
            K = get_MDK(A, int(args.kernel_parameter))
    elif args.kernel_function == "RLK":
        if args.kernel_parameter == None:
            K = get_RLK(A)
        else:
            K = get_RLK(A, args.kernel_parameter)
    elif args.kernel_function == "RBF":
        if args.kernel_parameter == None:
            K = get_RBF(A)
        else:
            K = get_RBF(A, args.kernel_parameter)
    np.save(args.output_file, K)



def get_MEDK(A, beta=0.04):
    """ Compute Markov exponential diffusion kernel.
    
    Parameters:
        A -- Adjacency matrix.
        beta -- Diffusion parameter (positive float, 0.04 by default).
        
    Return:
        MEDK -- Markov exponential diffusion kernel matrix.
    """
    
    from cvxopt import matrix
    from scipy.linalg import expm
    
    # N is the number of vertices        
    N = A.shape[0]
    for idx in range(N):
        A[idx,idx] = 0
    A = matrix(A)
    D = np.zeros((N,N))
    for idx in range(N):
        D[idx, idx] = sum(A[idx,:])
    I = np.identity(N)
    M = (beta/N)*(N*I - D + A)
    MEDK = expm(M)
    
    return MEDK



def get_MDK(A, t=10):
    """ Compute Markov diffusion kernel.
    
    Parameters:
        A -- Adjacency matrix.
        t -- Diffusion parameter (positive int, 10 by default).
        
    Return:
        MDK -- Markov diffusion kernel matrix.
    """
    
    N = A.shape[0]
    for idx in range(N):
        A[idx, idx] = 0
    
    s = np.sum(A, axis=1)
    P = np.divide(A, s[:, np.newaxis], dtype=float)
    
    Zt = P.copy()
    P_temp = P.copy()
    for i in range(t-1):
        P_temp = P_temp.dot(P)
        Zt = Zt + P_temp
    Zt = (1.0/t)*Zt
    MDK = Zt.dot(Zt.transpose())
    
    return MDK



def get_RLK(A, alpha=4.):
    """ Compute regularized Laplacian kernel.
    
    Parameters:
        A -- Adjacency matrix.
        alpha -- Diffusion parameter (positive float, 4.0 by default).
        
    Return:
        RLK -- Regularized Laplacian kernel matrix.
    """
    
    from scipy.linalg import inv
    
    # N is the number of vertices
    N = A.shape[0]
    for idx in range(N):
        A[idx, idx] = 0
    
    I = np.identity(N)
    D = np.zeros((N,N))
    for idx in range(N):
        D[idx,idx] = sum(A[idx,:])
    L = D - A
    RLK = inv(I + alpha*L)
    
    return RLK



def get_RBF(A, s=1.):
    """ Compute radial basis function kernel.
    
    Parameters:
        A -- Feature matrix.
        s -- Scale parameter (positive float, 1.0 by default).
        
    Return:
        K -- Radial basis function kernel matrix.
    """
    
    from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel
    from sklearn.preprocessing import scale
    
    A = scale(A)
    dist_matrix = euclidean_distances(A, A, None, squared=True)
    dist_vector = dist_matrix[np.nonzero(np.tril(dist_matrix))]
    dist_median = np.median(dist_vector)
    K = rbf_kernel(A, None, dist_median*s)
    
    return K



if __name__ == "__main__":
	main()
