#!/usr/bin/env python
# -*- coding: utf-8 -*-


#################
## Functions to process data
#################            
            

# general
import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd

# statistics
#import statsmodels.api as sm
from scipy.stats import multivariate_normal

# statistics which aren't all that nice in python
import rpy2.robjects as robjects



def AR_predict(ts,alphas,p):
	""" Make AR predictions for a given (set of) time series, alphas (AR coeffs) and AR order p.
	
		Keyword arguments:
		- ts = matrix with dim nr obs x nr time series: time series with the oldest value at position 0 and the newest value at position -1
		- alphas = AR coefficient for each time series, ie matrix with dim nr coeffs x nr time series
		- p = order of AR(p) process (determines whether alpha0 is present in AR coeffs or not)
		

		To do:
		- test the function for AR processes in which no alpha0 is included

	"""    
	ts_pd = pd.DataFrame(ts)

	if alphas.shape[0] == p+1:
		#print('AR process including alpha0 of order:',p)
		ar_pred = np.zeros(ts_pd.shape)
		ar_pred[:] = alphas[0]
		i=1

		while i <= p:
			ar_pred += alphas[i]*ts_pd.shift(i) #ATTENTION: I USED TO HAVE -i instead of i -> shifted in the wrong dir
                	# since I assume the oldest time slot is at position 0 and the newest at position -1
			i+=1

	if alphas.shape[0] == p:
		#print('AR process without alpha0 of order:',p)
		ar_pred = np.zeros(ts_pd.shape)
		ar_pred[:] = 0
		i=0
		while i < p:
			ar_pred += alphas[i]*ts_pd.shift((i+1)) #ATTENTION: I USED TO HAVE -i instead of i -> shifted in the wrong dir
                	# since I assume the oldest time slot is at position 0 and the newest at position -1
			i+=1
	return ar_pred

def AR1_predict(ts,alpha0,alpha1): # hardcoded for assumption that have intercept too
    """ Make AR(1) predictions for setup with intercept and alpha1 given, specific implementation of AR_predict() and hopefully slightly faster

    Keyword arguments:
        - ts =  matrix with dim nr time steps x nr gp: time series with the oldest value at position 0 and the newest value at position -1
        - alpha0 = intercept vector with dim nr gp
        - alpha1 = AR1 coeff vector with dim nr gp

    """

    ts_pd = pd.DataFrame(ts)
    
    ar_pred = np.zeros(ts_pd.shape)
    ar_pred += alpha0+alpha1*ts_pd.shift(1)
    
    return ar_pred


def compute_llh_cv(res_tr,res_cv,phi):
    """ Compute sum of log likelihood of a set of residuals based on a covariance matrix derived from a different set (of timeslots) of residuals
    
    Keyword arguments:
        - res_tr: the residual of the training run lacking a specific fold after removing the local mean response (nr ts x nr gp). Nans must be removed before
        - res_cv: the residual of a fold which was removed from the training run
        - phi: matrix to localize the covariance matrix based on a specific localisation radius and distance information (phi = output of fct gaspari_cohen(geo_dist/L))
    
    Output:
        - llh_innov_cv: sum of the log likelihood over the cross validation time slots
    
    """

    ecov_res_tr = np.cov(res_tr,rowvar=False)
    cov_res_tr=phi*ecov_res_tr
    
    mean_0 = np.zeros(phi.shape[0]) # we want the mean of the res to be 0

    llh_innov_cv=np.sum(multivariate_normal.logpdf(res_cv,mean=mean_0, cov=cov_res_tr))

    return llh_innov_cv   



def gaspari_cohn(r):
	""" Localize covariance matrix.
		
		Keyword argument:		
		- r = distance / localization radius

		Remarks    	
		- based on Gaspari-Cohn 1999, QJR  (as taken from Carrassi et al 2018, arxiv)
    
	"""
	r = np.abs(r)

    
	if r>=0 and r<1:
        	y = 1 - 5/3*r**2 + 5/8*r**3 + 1/2*r**4 - 1/4*r**5
	if r>=1 and r<2:
        	y = 4 - 5*r + 5/3*r**2 + 5/8*r**3 - 1/2*r**4 + 1/12*r**5 - 2/(3*r)
	if r>=2:
        	y = 0
        
        
	return y

