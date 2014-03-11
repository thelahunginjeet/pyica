from pyica import fastica as ica
from pyica import fastica_rpy as rica
import numpy as np


class TestICA:

    def setup(self):
        self.signals = np.vstack([np.sin([x/20.0 for x in xrange(1,1001)]),(1.0 + np.mod(xrange(1000),200) - 100.0)/100.0])
    	self.mixing = np.array([[0.291, 0.6557], [-0.5439, 0.5572]])
        self.X = np.dot(self.mixing,self.signals)
        self.A,self.W,self.S = ica(self.X,2,maxIterations=10000)

    def test_W_orthogonality(self):
        assert np.allclose(np.dot(self.W.T,self.W),np.eye(2),atol=1.0e-06),"python: W^TW not within 1e-06 of I"

    def test_S_recovery(self):
        from scipy.linalg import det
        assert np.allclose(1.0,np.abs(det(np.corrcoef(self.S,self.signals)[0:2,2:])),atol=1.0e-03),"python: |det(rho(ShatT,S))| not within 1e-03 of unity"


class TestRICA:

    def setup(self):
        self.signals = np.vstack([np.sin([x/20.0 for x in xrange(1,1001)]),(1.0 + np.mod(xrange(1000),200) - 100.0)/100.0])
    	self.mixing = np.array([[0.291, 0.6557], [-0.5439, 0.5572]])
        self.X = np.dot(self.mixing,self.signals)
        self.AR,self.WR,self.SR = rica(self.X,2,method="R",maxIterations=10000)
        self.AC,self.WC,self.SC = rica(self.X,2,method="C",maxIterations=10000)

    def test_R_W_orthogonality(self):
        assert np.allclose(np.dot(self.WR.T,self.WR),np.eye(2),atol=1.0e-06),"native R: W^TW not within 1.0e-06 of I"

    def test_R_S_recovery(self):
        from scipy.linalg import det
        assert np.allclose(1.0,np.abs(det(np.corrcoef(self.SR,self.signals)[0:2,2:])),atol=1.0e-03),"native R: |det(rho(ShatT,S))| not within 1e-03 of unity"

    def test_C_W_orthogonality(self):
        assert np.allclose(np.dot(self.WC.T,self.WC),np.eye(2),atol=1.0e-06),"R calling C: W^TW not within 1.0e-06 of I"

    def test_C_S_recovery(self):
        from scipy.linalg import det
        assert np.allclose(1.0,np.abs(det(np.corrcoef(self.SC,self.signals)[0:2,2:])),atol=1.0e-03),"R calling C: |det(rho(ShatT,S))| not within 1e-03 of unity"
