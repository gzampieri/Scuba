
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
from sklearn.metrics import roc_auc_score



def AUROC(label_list, score_list):
	""" Compute the area under the ROC curve. 
	
	Parameters:
	label_list -- List of true binary labels in the range {0,1} or {-1,1}.
	              Positive examples are labelled by 1 and unlabelled examples are labelled either by 0 or -1.
	score_list -- List of target scores to be sorted.
	
	Return:
	roc_auc_score -- Area under the ROC curve for the rank of target scores (float, range [0,1]).
                     It expresses the probability, picking a positive and an unlabelled example randomly, that the chosen positive ranks higher than the chosen unlabelled.
																					
	Notes:
	label_list and score_list must have the same length.
	"""
	
	return roc_auc_score(label_list, score_list)



def singletarget_AUROC(pos, rank_len, threshold = 1.0):
	if pos <= int(rank_len*threshold):
		return float(int(rank_len*threshold)-pos+1)/int(rank_len*threshold)
	else:
		return 0.0



def rank_pos(scores, labels):
	
	ordered_label_list = [t[1] for t in sorted(zip(scores, labels), reverse=True)]
	return ordered_label_list.index(1)+1



def TPR(pos, threshold = 0.1):
	""" Return true positive rate above a given threshold. """
	
	return float(len([float(p[0]) for p in pos if float(p[0]) <= threshold*p[1]]))*100/len(pos)



def evaluate(pos):
	""" Compute unbiased evaluation performances. """
	
	performance = {}
	pos_ratio = [float(p[0]*100)/p[1] for p in pos]
	
	performance['median'] = np.around(np.median(pos_ratio), decimals=2)
	performance['std'] = np.around(np.std(pos_ratio), decimals=2)
	performance['lowest'] = np.around(np.amax(pos_ratio), decimals=2)
	performance['tpr01'] = np.around(TPR(pos, 0.01), decimals=1)
	performance['tpr05'] = np.around(TPR(pos, 0.05), decimals=1)
	performance['tpr10'] = np.around(TPR(pos, 0.10), decimals=1)
	performance['tpr15'] = np.around(TPR(pos, 0.15), decimals=1)
	performance['tpr20'] = np.around(TPR(pos, 0.20), decimals=1)
	performance['tpr30'] = np.around(TPR(pos, 0.30), decimals=1)
	performance['auc'] = np.around(np.sum([singletarget_AUROC(p[0], p[1]) for p in pos])/len(pos), decimals=3)
	performance['auc30'] = np.around(np.sum([singletarget_AUROC(p[0], p[1], 0.3) for p in pos])/len(pos), decimals=3)
	
	return performance


