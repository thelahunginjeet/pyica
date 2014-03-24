from numpy.random import randn,rand,randint,gamma,laplace
from numpy import pi
from numpy import tan,arctanh,sinh,arcsinh,power,sqrt,log
from numpy import shape,newaxis,vstack


def unitsources(nSources=(1,1,1),nSamples=1024,subType='dblcosh',supType='invcosh'):
    '''
    Generates a sum(nSources) x nSamples array of sources; each row is a source, and they all have
    zero mean and unit variance (up to sampling errors).
    
    Parameters:
    ----------
    nSources : tuple, optional
        nSources[0] : number of SubGaussian sources (zero is allowed)
        nSources[1] : number of Gaussian sources (zero is allowed)
        nSources[2] : number of SuperGaussian sources (zero is allowed)
        
    nSamples : number of samples (length of each row), optional
  
    subType : string, optional
	type of subgaussian distribution from which to sample; unrecognized 
	distribution types default to 'dblcosh'

    supType : string, optional
	type of supergaussian distribution from which to sample; unrecognized
	distribution types default to 'invcosh'

 
    Output:
    ----------
    S : numpy array
        first nSources[0] rows are SubGaussian (starting at 0)
        next nSources[1] rows are Gaussian
        next nSources[2] rows are SuperGaussian

    '''
    # for function dispatching
    superGaussTable = {'invcosh': sinvcosh, 'laplace': slaplace, 'logistic': slogistic,
	'exparcsinh': sexparcsinh}
    subGaussTable = {'dblcosh': sdblcosh, 'expsinh': sexpsinh, 'gg' : sgg}
    sourceList = []
    # subgaussians
    try:
        s = subGaussTable[subType](nSources[0],nSamples)
    except KeyError:
	# default to dblcosh
        s = sdblcosh(nSources[0],nSamples)
    sourceList.append(s)
    # gaussians
    sourceList.append(sgauss(nSources[1],nSamples))
    # supergaussians
    try:
        s = superGaussTable[supType](nSources[2],nSamples)
    except KeyError:
	# default to invcosh
        s = sinvcosh(nSources[2],nSamples)
    sourceList.append(s)
    return vstack(filter(lambda x : shape(x) > 0, sourceList))



""" Gaussian distribution """

def sgauss(nSources,nSamples):
    '''
    Generates an nSources x nSamples array of Gaussian sources with mean zero and unit variance.
    Simply wraps the appropriate function from numpy.rand.
    '''
    return randn(nSources,nSamples)


""" SuperGaussian distributions """

def sinvcosh(nSources,nSamples):
    '''
    Generates an nSources x nSamples array of sources distributed according to:
        
        p(x) = 1/(2*cosh(pi*x/2))
    
    This yields a set of superGaussian sources (more peaked than Gaussian), and 
    each source will have zero mean and unit variance.
    '''
    return (4/pi)*arctanh(tan((pi/2)*(rand(nSources,nSamples) - 0.5)))


def slaplace(nSources,nSamples):
    '''
    Generates an nSources x nSamples array of sources which are Laplace distributed.
    '''
    s = laplace(size=(nSources,nSamples))
    return s/s.std(axis=1)[:,newaxis]


def slogistic(nSources,nSamples):
    '''
    Generates an nSources x nSamples array of logistically distributed random 
    variates with mean zero and unit variance:

        p(x) = 1/((2*sqrt(12)/pi)*cosh^2(pi*x/sqrt(12)))
    
    '''
    return -(sqrt(3)/pi)*log(1.0/rand(nSources,nSamples) - 1.0)


def sexparcsinh(nSources,nSamples):
    '''
    Generates and nSources by nSamples array of sources distributed according to:
    
        p(x) = sqrt((1 + sinh(x)^2)/2*pi)*exp(-sinh(x)^2/2)

    This yields a set of superGaussian sources; each source is standardized to
    have unit variance (p(x) above does not). 
    '''
    s = sinh(randn(nSources,nSamples))
    return s/s.std(axis=1)[:,newaxis]


""" SubGaussian distributions """

def sdblcosh(nSources,nSamples):
    '''
    Generates an nSources x nSamples array of sources distributed according to:
    
        p(x) = exp(-x^2)*cosh(x*sqrt(2))
    
    This yields a set of subGaussian sources (flatter top that Gaussian), and
    each source will have zero mean and unit variance.
    '''
    return (.5**.5)*(randn(nSources,nSamples) + randspin(size=(nSources,nSamples)))


def sexpsinh(nSources,nSamples):
    '''
    Generates an nSources x nSamples set of sources distributed according to:

        p(x) = sqrt((1 + sinh(x)^2)/(2*pi))*exp(-sinh^2(x)/2)

    Each source is standardize to have unit variance.
    '''
    s = arcsinh(randn(nSources,nSamples))
    return s/s.std(axis=1)[:,newaxis]


def sgg(nSources,nSamples):
    '''
    Generates an nSources x nSamples set of sources distributed according to:

        p(x) = (8/(2*Gamma[1/8]))*exp(-x^8),

    i.e. a generalized Gaussian distribution with shape parameter = 8.
    '''
    gg = randspin(size=(nSources,nSamples))*power(abs(gamma(shape=1.0/8.0,scale=1.0,size=(nSources,nSamples))),1.0/8.0)
    return gg/gg.std(axis=1)[:,newaxis]


""" Accessory functions """

def randbit(size=None):
    '''
    Generates an array of shape size of random {0,1} bits.
    '''
    if size is None:
        return randint(2)
    else:
        return randint(2,size=size)

def randspin(size=None):
    '''
    Generates an array of shape size of random {-1,1} spin variables.
    '''
    return 2*randbit(size=size) - 1;
