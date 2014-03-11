'''
Wraps R's version of FastICA via Rpy2.  See the README for information about possible 
differences between the python and R versions of FastICA.

Created on June 14, 2013

@author: Kevin S. Brown, University of Connecticut

This source code is provided under the BSD-3 license, duplicated as follows:

Copyright (c) 2013, Kevin S. Brown
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, 
are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this 
    list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice, this 
    list of conditions and the following disclaimer in the documentation and/or other 
    materials provided with the distribution.

    3. Neither the name of the University of Connecticut nor the names of its contributors 
    may be used to endorse or promote products derived from this software without specific 
    prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS 
OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY 
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER 
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT 
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

from numpy import array
from numpy.random import randn

try:
    import rpy2.robjects.numpy2ri
    from rpy2.robjects.packages import importr
    rpy2.robjects.numpy2ri.activate()
except ImportError:
    print "Connection to R via Rpy not available; fastica_rpy will not work."


def fastica_rpy(X,nSources,algorithm="parallel",nonlinearity="logcosh",method="C",maxIterations=500,tolerance=1.0e-05,Winit=None):
    """
    fastica_rpy wraps the fastICA package in R, with communication to R via rpy2. If rpy2 is not installed (or R is
    not installed), this will not work. For full documentation see the R package. To use the defaults, simpy call:

            A, W, S = fastica_rpy(X,nSources)

    Parameters:
    ------------
    X : numpy array, required
        data matrix
        
    nSources : integer, required
        desired number of sources to extract

    algorithm : string, optional 
       "deflation" : components extracted one at a time
       "parallel"  : components extracted all at once

    nonlinearity: string, optional 
       should be either "logcosh" or "exp"

    method : string, optional
       should be either "R" (to do the calculations in R) or
       "C" (to use the C library R is wrapping)

    maxIterations : integer, optional
       number of fixed-point iterations

    tolerance : float, optional
       convergence criterion for the unmixing matrix

    Winit : numpy array, optional
       initial guess for the unmixing matrix

    Output:
    ---------- 
    A : numpy array
        mixing matix of size X.shape[0] x nSources

    W : numpy array
        unmixing matrix of size nSources x X.shape[0]

    S : numpy array
        matrix of independent courses, size nSources x X.shape[1]
    """
    if Winit is None:
        Winit = randn(nSources,nSources)
    try:
        rica = importr('fastICA')
    except NameError:
        print 'Cannot import package \'fastICA\'; you are missing either rpy2 (python problem) or fastICA (R problem)'
    outDict = rica.fastICA(X.T,nSources,fun=nonlinearity,method=method,maxit=maxIterations,tol=tolerance,w_init=Winit.T)
    return array(outDict.rx2('A')).T, array(outDict.rx2('W')).T, array(outDict.rx2('S')).T