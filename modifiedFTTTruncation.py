import torch
from functional_tt_fabrique import Extended_TensorTrain, orthpoly
from tt_fabrique import TensorTrain

from math import sqrt
from copy import deepcopy

class Threshold(object):
    def __init__(self,delta):
        self.delta = delta
    def __call__(self, u, sigma,v,  pos):
        return max([torch.sum(sigma > self.delta),1])

def roundToAcc(tensorTrain, accuracy, verbose = False):
    """performs the TT rounding procedure by Oseledets.
    """
    assert isinstance(tensorTrain, TensorTrain)

    # copy the input tensor so that it is not overwritten
    roundedTensorTrain = deepcopy(tensorTrain)

    # orthogonalize from right to left
    roundedTensorTrain.set_core(0)
    
    # note that the Frobenoius norm of the non-orthogonal core equals the frob norm of the ful tensor
    truncationParam = accuracy/sqrt(len(tensorTrain.comps)-1) * torch.linalg.norm(roundedTensorTrain.comps[0])
    rule = Threshold(truncationParam)
    
    # perform SVD sweep through the TT
    for pos in range(roundedTensorTrain.n_comps-1):
        c = roundedTensorTrain.comps[pos]
        s = c.shape
        c = c.reshape(s[0]*s[1], s[2])
        u, sigma, vt = torch.linalg.svd(c, full_matrices=False) 
        new_rank =  rule(u, sigma, vt, pos) 

        # update informations
        u, sigma, vt = u[:,:new_rank], sigma[:new_rank], vt[:new_rank,:]
        new_shape = (s[0], s[1], min(new_rank,s[2]))
        
        roundedTensorTrain.comps[pos] = u.reshape(new_shape)
        roundedTensorTrain.comps[pos+1] = torch.einsum('ir, rkl->ikl ', torch.diag(sigma) @ vt, roundedTensorTrain.comps[pos+1] ) # Stimmt das noch ?

    roundedTensorTrain.core_position = roundedTensorTrain.n_comps-1
    assert(roundedTensorTrain.comps[0].shape[0] == 1 and roundedTensorTrain.comps[-1].shape[2] == 1)
    
    # update the rank
    roundedTensorTrain.rank = [1] + [roundedTensorTrain.comps[pos].shape[2] for pos in range(roundedTensorTrain.n_comps-1)] + [1]
    if verbose:
        print('New rank is ', roundedTensorTrain.rank)

    return roundedTensorTrain

def roundToAccReverse(tensorTrain, accuracy, verbose = False):
    """performs the TT rounding procedure by Oseledets in reverse order (from right to left).
    """
    assert isinstance(tensorTrain, TensorTrain)
    dimension = len(tensorTrain.comps)

    # copy the input tensor so that it is not overwritten
    roundedTensorTrain = deepcopy(tensorTrain)

    # orthogonalize from left to right
    roundedTensorTrain.set_core(dimension-1)
    
    # note that the Frobenoius norm of the non-orthogonal core equals the frob norm of the ful tensor
    truncationParam = accuracy/sqrt(len(tensorTrain.comps)-1) * torch.linalg.norm(roundedTensorTrain.comps[dimension-1])
    rule = Threshold(truncationParam)
    
    # perform SVD sweep through the TT
    for pos in range(roundedTensorTrain.n_comps-1,0,-1):
        c = roundedTensorTrain.comps[pos]
        s = c.shape
        c = c.reshape(s[0], s[1]*s[2])
        u, sigma, vt = torch.linalg.svd(c, full_matrices=False) 
        new_rank =  rule(u, sigma, vt, pos) 

        # update informations
        u, sigma, vt = u[:,:new_rank], sigma[:new_rank], vt[:new_rank,:]
        new_shape = (min(new_rank,s[0]), s[1], s[2])
    
        roundedTensorTrain.comps[pos] = vt.reshape(new_shape)
        roundedTensorTrain.comps[pos-1] = torch.einsum('ijk, kl->ijl ', roundedTensorTrain.comps[pos-1], u@torch.diag(sigma) )

    roundedTensorTrain.core_position = 0
    assert(roundedTensorTrain.comps[0].shape[0] == 1 and roundedTensorTrain.comps[-1].shape[2] == 1)
    
    # update the rank
    roundedTensorTrain.rank = [1] + [roundedTensorTrain.comps[pos].shape[2] for pos in range(roundedTensorTrain.n_comps-1)] + [1]
    if verbose:
        print('New rank is ', roundedTensorTrain.rank)

    return roundedTensorTrain


def approximateDwithMCI(FTT,corePosition,qListLeft,qListRight,numSamples,seed=None):
    """approximate the D_i term from the coordinate flow paper with MC approximation, where i is the core position

    Args:
        FTT (_type_): _description_
        corePosition (_type_): _description_
        qListLeft (_type_): _description_
        qListRight (_type_): _description_
        numSamples (_type_): _description_
        seed (_type_, optional): _description_. Defaults to None.

    Returns:
        integralApprox (torch.tensor): MC approximation of the (d,d) matrix D_i
    """
    
    if seed is not None:
        torch.manual_seed(seed)
    samples, normalizationConst = sampleFTTDomain(FTT,numSamples)
    
    # note the unsqueeze due to evaluateQProducts being of shape (b,1) --> (b,1,1) and evaluateGradVTTtimesX being of shape (b,d,d)
    integrand = evaluateGradVTTtimesX(FTT,samples) + evaluateQProducts(FTT,corePosition,qListLeft,qListRight,samples).unsqueeze(2)
    
    # MC approximation replacing integral by sum
    integralApprox = torch.sum(integrand,dim=0) * normalizationConst # will be shape (d,d) 
    return integralApprox

def sampleFTTDomain(FTT,numSamples):
    d = len(FTT.tt.comps)
    domain = FTT.tfeatures.domain
    strip = domain[0]
    sample_points = (strip[1]-strip[0])*torch.rand((numSamples,d)) + strip[0]
    normalizationConst = 1./(strip[1]-strip[0])**d
    return sample_points, normalizationConst


def evaluateGradVTTtimesX(FTT,x):
    """evaluates grad_FTT(x) * x^T as required in the D terms from the coordinate flow paper

    Args:
        FTT (Extended_TensorTrain): FTT with number of dimensions equal to x.shape[1]
        x (torch.tensor): shape should be (batchSize, dimension)

    Returns:
        torch.tensor: shape (b,d,d) outer product of grad and x.
    """
    grad = FTT.grad(x)  # should be of shape (b,d)
    
    # compute the outer product gradFTT * xFTT^T
    return torch.einsum('bd,bj->bdj',grad,x)


def evaluateQProducts(FTT,corePosition,qListLeft,qListRight,x):
    """compute the contraction
            Q_1Q_2Q_3Q_4...Q_{i-1}Q_iQ_{i+1}...Q_d
        where i is the core position, Q_1,..., Q_{i-1} (qListLeft) are left-orthonormalized
        and Q_iQ_{i+1}...Q_d (qListRight) are right-orthonormalized

    Args:
        FTT (_type_): _description_
        corePosition (_type_): _description_
        qListLeft (_type_): _description_
        qListRight (_type_): _description_
        x (_type_): _description_

    Returns:
        (torch.tensor): shape (b,1) returns the scalar contraction of the functional Q lists for every input
    """
    d = x.shape[1]              
    assert corePosition < d and corePosition >= 0
    assert len(qListLeft) == corePosition and len(qListRight) == d-corePosition
    
    rank1obj = FTT.tfeatures(x)
    
    for pos in range(0, corePosition):
            # match of inner dimension with respective vector size
            assert(qListLeft[pos].shape[1] == rank1obj[pos].shape[1])
            # vectors must be 2d objects 
            assert(len(rank1obj[pos].shape) == 2)
    for pos in range(0, d-corePosition):
            # match of inner dimension with respective vector size
            assert(qListRight[pos].shape[1] == rank1obj[pos].shape[1])
            # vectors must be 2d objects 
            assert(len(rank1obj[pos].shape) == 2)
            
    leftContractions = [ torch.einsum('ijk, bj->ibk', c, v)  for  c,v in zip(qListLeft, rank1obj[:corePosition]) ]
    rightContractions = [ torch.einsum('ijk, bj->ibk', c, v)  for  c,v in zip(qListRight, rank1obj[corePosition:]) ]
    contractions = leftContractions + rightContractions 
    
    result = contractions[-1]
    for pos in range(d-2, -1,-1):
        # contraction w.r.t. the 3d coordinate
        result = torch.einsum('ibj, jbk -> ibk', contractions[pos], result) 
    # result is of shape b x 1
    return result.reshape(result.shape[1], result.shape[2])
        


def modifiedFTTtruncation(FTT, accuracy, numSamplesForD, verbose = False, debug = False):
    """performs the modified FTT truncation algorithm (algorithm 2 in the paper by Dektor and Venturi)
       
       Input:
        v → FTT tensor with cores 1,2,...,d
        δ → desired accuracy
        
        Output:
        vTT → truncated FTT tensor satisfying ||v - FTT||_L2(μ) ≤ δ||v||_L2(μ)
        S(vTT) → sum of all multilinear singular values of vTT
        D → left factor of the Riemannian gradient (40)
    """
    assert isinstance(FTT, Extended_TensorTrain)
    dimension = len(FTT.tt.comps)

    # copy the input tensor so that it is not overwritten
    roundedTensorTrain = deepcopy(FTT)

    # orthogonalize from left to right
    roundedTensorTrain.tt.set_core(dimension-1)
    
    # initialize the list of left Q matrices with the orthonormal cores (up to the last one, which is non-orthogonal)
    qList = deepcopy(roundedTensorTrain.tt.comps)
    if debug:
        print("qList shapes: ", [q.shape for q in qList])
    
    # initialize sum of D matrices
    D = torch.zeros((dimension,dimension))
    
    # note that the Frobenoius norm of the non-orthogonal core equals the frob norm of the ful tensor
    truncationParam = accuracy/sqrt(len(roundedTensorTrain.tt.comps)-1) * torch.linalg.norm(roundedTensorTrain.tt.comps[dimension-1])
    rule = Threshold(truncationParam)
    
    # initialize the sum of all singular values encountered during the sweep
    singularValueSum = 0
    
    # perform sweep through the TT from right to left
    for pos in range(roundedTensorTrain.tt.n_comps-1,0,-1):
        if debug:
            print('position is ', pos)
        
        # perform right unfolding of the current core
        c = roundedTensorTrain.tt.comps[pos]
        s = c.shape
        if debug:
            print('the current core shape is ', s)
        c = c.reshape(s[0], s[1]*s[2])
        
        # Perform LQ decomposition of the unfolding (cf. (60) in paper)
        QT, LT = torch.linalg.qr(torch.transpose(c,1,0)) 
        L = torch.transpose(LT, 1, 0)
        Q = torch.transpose(QT, 1, 0)
        if debug:
            print('LQ decomposition of the core yields shapes')
            print('for L: \t', L.shape)
            print('for Q: \t', Q.shape)
        
        # perform SVD of L and determine truncated rank (cf. (61) in paper)
        u, sigma, vt = torch.linalg.svd(L, full_matrices=False) 
        new_rank =  rule(u, sigma, vt, pos) 
        print('the new rank after truncation is: ', new_rank.item())

        # update SVD via rank truncation
        u, sigma, vt = u[:,:new_rank], sigma[:new_rank], vt[:new_rank,:]
        if debug:
            print('the shapes of the svd USVT of L (after truncation) are')
            print('for U: \t\t', u.shape)
            print('for S: \t\t', sigma.shape)
            print('for VT: \t', vt.shape)
        
            print('the shape of the previous core is: ')
            print(roundedTensorTrain.tt.comps[pos-1].shape)
        
        new_shape = (min(new_rank,s[0]).item(), s[1], s[2])
        previousShape = roundedTensorTrain.tt.comps[pos-1].shape
        new_shape_previous = (previousShape[0],previousShape[1],min(new_rank,s[0]).item())
        if debug:
            print('new shape of the current core should be: ', new_shape)
    
        # Update the current core and the next core (to the left) (cf. (63) and (65) in paper)
        roundedTensorTrain.tt.comps[pos] = torch.einsum('li,ij->lj', vt, roundedTensorTrain.tt.comps[pos].reshape(s[0], s[1]*s[2])).reshape(new_shape)
        
        if debug:
            print("the shape of the current core after update: ", roundedTensorTrain.tt.comps[pos].shape)
        roundedTensorTrain.tt.comps[pos-1] = torch.einsum('ik, kl->il ', roundedTensorTrain.tt.comps[pos-1]\
                                                          .reshape(previousShape[0]*previousShape[1],previousShape[2]), u@torch.diag(sigma) )\
                                                          .reshape(new_shape_previous)
        if debug:
            print("the shape of the previous core after update: ", roundedTensorTrain.tt.comps[pos-1].shape)
        
        # Update Q lists (cf. (63) in the paper. Note that we leave out the singular values here.)
        if debug:
            print('qList[pos-1] shape: ', qList[pos-1].shape)
            print('u shape: ', u.shape)
        qList[pos-1] = torch.einsum('ijk, kl->ijl ', qList[pos-1], u )
        if debug:
            print('qList[pos] shape: ', qList[pos].shape)
            print('vT shape: ', vt.shape)
        qList[pos] = torch.einsum('li,ijk->ljk', vt, qList[pos])
        
        
        # compute D matrix (cf. (64) in the paper. Note that there is a mistake in the paper! The index should be >i-1)
        Dpos = approximateDwithMCI(roundedTensorTrain,pos,qList[:pos],qList[pos:],numSamplesForD,seed=1)
        D += Dpos
        
        # add all remianing singular values to the sum
        singularValueSum += sum(sigma)

    roundedTensorTrain.tt.core_position = 0
    assert(roundedTensorTrain.tt.comps[0].shape[0] == 1 and roundedTensorTrain.tt.comps[-1].shape[2] == 1)
    
    # update the rank
    roundedTensorTrain.tt.rank = [1] + [roundedTensorTrain.tt.comps[pos].shape[2] for pos in range(roundedTensorTrain.tt.n_comps-1)] + [1]
    if verbose:
        print('New rank is ', roundedTensorTrain.tt.rank)

    return roundedTensorTrain, singularValueSum, D


def modifiedFTTtruncationOld(tensorTrain, accuracy, verbose = False):
    """performs the modified FTT truncation algorithm (algorithm 2 in the paper by Dektor and Venturi)
       Note that this algorithm takes as input a TT instead of an FTT. This is due to the algorithm assuming that
       the input TT is associated with a set of orthonormal basis functions in every dimension.
       All the FTT operations from the original algorithm can then be performed on the TT, using the orthonormality of the basis. 
       
       Input:
        v → TT tensor with cores 1,2,...,d
        δ → desired accuracy
        
        Output:
        vTT → truncated TT tensor satisfying ||v - F(TT)||_L2(μ) ≤ δ||v||_L2(μ)
        S(vTT) → sum of all multilinear singular values of vTT
        D → left factor of the Riemannian gradient (40)
    """
    assert isinstance(tensorTrain, TensorTrain)
    dimension = len(tensorTrain.comps)

    # copy the input tensor so that it is not overwritten
    roundedTensorTrain = deepcopy(tensorTrain)

    # orthogonalize from left to right
    roundedTensorTrain.set_core(dimension-1)
    
    # initialize the list of left Q matrices with the orthonormal cores (up to the last one, which is non-orthogonal)
    qListLeft = roundedTensorTrain.comps[:-1]
    
    # note that the Frobenoius norm of the non-orthogonal core equals the frob norm of the ful tensor
    truncationParam = accuracy/sqrt(len(tensorTrain.comps)-1) * torch.linalg.norm(roundedTensorTrain.comps[dimension-1])
    rule = Threshold(truncationParam)
    
    # initialize the sum of all singular values encountered during the sweep
    singularValueSum = 0
    
    # perform sweep through the TT from right to left
    for pos in range(roundedTensorTrain.n_comps-1,0,-1):
        
        # perform right unfolding of the current core
        c = roundedTensorTrain.comps[pos]
        s = c.shape
        c = c.reshape(s[0], s[1]*s[2])
        
        # Perform LQ decomposition of the unfolding
        QT, LT = torch.linalg.qr(torch.transpose(c,1,0)) 
        
        # perform SVD and determine truncated rank
        u, sigma, vt = torch.linalg.svd(c, full_matrices=False) 
        new_rank =  rule(u, sigma, vt, pos) 

        # update SVD via rank truncation
        u, sigma, vt = u[:,:new_rank], sigma[:new_rank], vt[:new_rank,:]
        new_shape = (min(new_rank,s[0]), s[1], s[2])
    
        # Update the current core and the next core (to the left)
        roundedTensorTrain.comps[pos] = vt.reshape(new_shape)
        roundedTensorTrain.comps[pos-1] = torch.einsum('ijk, kl->ijl ', roundedTensorTrain.comps[pos-1], u@torch.diag(sigma) )
        
        # add all remianing singular values to the sum
        singularValueSum += sum(sigma)

    roundedTensorTrain.core_position = 0
    assert(roundedTensorTrain.comps[0].shape[0] == 1 and roundedTensorTrain.comps[-1].shape[2] == 1)
    
    # update the rank
    roundedTensorTrain.rank = [1] + [roundedTensorTrain.comps[pos].shape[2] for pos in range(roundedTensorTrain.n_comps-1)] + [1]
    if verbose:
        print('New rank is ', roundedTensorTrain.rank)

    return roundedTensorTrain



def ttUnitTest():
    d = 5
    n = 5
    tt = TensorTrain(dims=[n]*d)

    ranks = [1] + [n]*(d-1) + [1]
    torch.manual_seed(0)
    tt.fill_random(ranks,eps=1.)
    
    for i in range(d):
        print(f'Shape of {i}-th core is {tt.comps[i].shape}.')

    roundedtt = roundToAccReverse(tt,0.01)

    for i in range(d):
        print(f'Shape of {i}-th core of rounded TT is {roundedtt.comps[i].shape}.')
    print(roundedtt.rank)
    
    
def modifiedruncationUnitTest():
    d = 5
    degrees = [4]*d
    ranks = [1] + [4]*(d-1) + [1]
    
    domain = [[-1.,1.] for _ in range(d)]

    # torch.manual_seed(1000)
    op = orthpoly(degrees,domain)
    ETT = Extended_TensorTrain(op,ranks)

    for i in range(d):
        print(f'Shape of {i}-th core is {ETT.tt.comps[i].shape}.')

    roundedTensorTrain, singularValueSum, D = modifiedFTTtruncation(ETT,1e-2,1000,verbose=False)

    for i in range(d):
        print(f'Shape of {i}-th core of rounded TT is {roundedTensorTrain.tt.comps[i].shape}.')
    print(roundedTensorTrain.tt.rank)

if __name__ == "__main__":
    
    ttUnitTest()
    modifiedruncationUnitTest()