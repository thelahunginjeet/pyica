import pyica.fastica as ica
import pyica.rpyica as rica
import numpy as np


class TestICA:

    def setup(self):
        print "setup class: TestICA"
        self.signals = np.vstack([np.sin([x/20.0 for x in xrange(1,1001)]),(1.0 + np.mod(xrange(1000),200) - 100.0)/100.0])
    	self.mixing = np.array([[0.291, 0.6557], [-0.5439, 0.5572]])
        self.X = np.dot(self.mixing,self.signals)
        self.A,self.W,self.S = ica.fastica(self.X,2,maxIterations=1000)

    def test_W_orthogonality(self):
        assert np.allclose(np.dot(self.W.T,self.W),np.eye(2),atol=1.0e-06)


class TestRICA:

    def setup(self):
        print "setup class: TestRICA"
        self.signals = np.vstack([np.sin([x/20.0 for x in xrange(1,1001)]),(1.0 + np.mod(xrange(1000),200) - 100.0)/100.0])
    	self.mixing = np.array([[0.291, 0.6557], [-0.5439, 0.5572]])
        self.X = np.dot(self.mixing,self.signals)

    def test_R_calcs(self):
        A,W,S = rica.fastica_rpy(self.X,2,method="R",maxIterations=1000)
        assert np.allclose(np.dot(W.T,W),np.eye(2),atol=1.0e-06)

    def test_C_calcs(self):
        A,W,S = rica.fastica_rpy(self.X,2,method="C",maxIterations=1000)
        assert np.allclose(np.dot(W.T,W),np.eye(2),atol=1.0e-06)