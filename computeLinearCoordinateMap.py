
import torch
from functional_tt_fabrique import Extended_TensorTrain, orthpoly
from modifiedFTTTruncation import modifiedFTTtruncation



def Phi(v,D,stepSize):
    """Defines the right hand side of the FTT valued PDE Eq. (45) from the paper
    """
    pass

def PhiHat(Gamma,D):
    """Implements the right hand side of matrix ODE Eq. (39) from the paper, assuming that the matrix D is given as in Eq. (41)

    Args:
        Gamma (torch.tensor): current linear trafo of shape (d,d)
        D (torch.tensor): matrix of shape (d,d) defined in Eq. (41) or respecitevly (64) of the paper.

    Returns:
        -D@Gamma: matrix of shape (d,d).
    """
    return - D @ Gamma

def backwardsDifference(stencilPoints,stepSize):
    """computes the 3 point backwards finite difference quotient to approximate the derivative f'(x0).
    The stencilPoints provides should be a list [f(x0),f(x-1),f(x-2),f(x-3),...]
    The FD scheme will be computed up to p=6 stencil points (excluding the point x0 at which the derivative is evaluated) 

    Args:
        stencilPoints (list): list of function values at decreasing inputs x0, x-1, x-2, x-3,... where xi = xi-1 + stepSize
        stepSize (float): stepSize between the points xi

    Returns:
        f'(x0): finite backwards difference approximation of f'(x0)
    """
    numPoints = len(stencilPoints) - 1
    assert numPoints >= 1
    if numPoints == 1:
        return (stencilPoints[0] - stencilPoints[1]) / stepSize
    elif numPoints == 2:
        return (3./2.*stencilPoints[0] - 2*stencilPoints[1] + 1./2.*stencilPoints[-2]) / stepSize
    elif numPoints == 3:
        return (11./6.*stencilPoints[0] - 3.*stencilPoints[1] + 3./2.*stencilPoints[2] - 1./3.*stencilPoints[3]) / stepSize
    elif numPoints == 4:
        return (25./12.*stencilPoints[0] - 4.*stencilPoints[1] + 3.*stencilPoints[2]  -4./3.*stencilPoints[3] + 1./4.*stencilPoints[4]) / stepSize
    elif numPoints == 5:
        return (137./60.*stencilPoints[0] - 5.*stencilPoints[1] + 5.*stencilPoints[2]  -10./3.*stencilPoints[3] + 5./4.*stencilPoints[4] - 1./5.*stencilPoints[5]) / stepSize
    else:
        return (49./20.*stencilPoints[0] - 6.*stencilPoints[1] + 15./2.*stencilPoints[2]  -20./3.*stencilPoints[3] + 15./4.*stencilPoints[4] - 6./5.*stencilPoints[5] + 1./6.*stencilPoints[6]) / stepSize



def computeLinearCoordinateMap(initialFTT, stepSize, stoppingTolerance, maximumIterations):
    """
        Algorithm 1 from https://arxiv.org/pdf/2207.11955
        
        Input:
        uTT → initial FTT tensor,
        ∆eps → step-size for gradient descent,
        η → stopping tolerance,
        Miter → maximum number of iterations.
        
        Output:
        Γ → rank-reducing linear coordinate transformation,
        vTT → reduced rank FTT tensor on transformed coordinates vTT(x) = uTT(Γx) .
    """    
    assert isinstance(initialFTT, Extended_TensorTrain)
    
    dimension = len(initialFTT.tt.comps)
    
    v0, S0, D0 = modifiedFTTtruncation(initialFTT,accuracy=1e-2,numSamplesForD=1000)
    
    v = [v0]
    S = [S0]
    D = [D0]
    
    linearTrafos = [torch.eye(dimension)]
    
    p = 3
    Sdot = -float('inf')
    S = [0 for _ in range(p+1)]
    
    iter = 0
    while (Sdot < -stoppingTolerance and iter < maximumIterations):
        
        v.append(v[-1] + stepSize * Phi(v[-1],D[-1],stepSize))
        v1, S1, D1 = modifiedFTTtruncation(v[-1])
        
        linearTrafos.append(linearTrafos[-1] + stepSize * PhiHat(linearTrafos[-1],D[-1],stepSize))
        v.append(v1)
        S.append(S1)
        D.append(D1)
        
        # approximating the current derivative of S based on backwards FD. The inversion of the list is due to the BD function expecting its input as f(x0), f(x-1), ...
        Sdot = backwardsDifference(S[::-1],stepSize)
        
        coordinateTransform = linearTrafos[-1]
        reducedRankFTT = v[-1]
    
        iter += 1

    return coordinateTransform, reducedRankFTT


def coordinateMapUnitTest():
    d = 5
    degrees = [4]*d
    ranks = [1] + [4]*(d-1) + [1]
    
    domain = [[-1.,1.] for _ in range(d)]

    # torch.manual_seed(1000)
    op = orthpoly(degrees,domain)
    ETT = Extended_TensorTrain(op,ranks)
    
    stepSize = 1e-2
    stoppingTolerance = 1e-1
    maximumIterations = 100
    coordinateTransform, reducedRankFTT = computeLinearCoordinateMap(ETT, stepSize, stoppingTolerance, maximumIterations)


if __name__ == "__main__":
    coordinateMapUnitTest()