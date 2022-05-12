from scipy import sparse
from scipy.sparse import linalg
import numpy as np
import collections

def construct_matrix(order,n):
    '''
    Create a sparse matrix with dimension (n x n) for the given order
    order : select the order of the derivative to use for smoothing
    n     : dimension of the matrix
    '''
    if order == 0:
        diag = np.ones(n)*2
        diag[0]  = 1
        diag[-1] = 1
        diag1 = -np.ones(n-1)*2
        A1 = sparse.diags([diag,diag1],[0,1])
    elif order == 1:
        diag = np.ones(n)*6
        diag[:2] = [1,5]
        diag[-2:] = [5,1]
        diag1 = -np.ones(n-1)*8
        diag1[0] = -4
        diag1[-1] = -4
        diag2 = np.ones(n-2)*2
        A1 = sparse.diags([diag,diag1,diag2],[0,1,2])
    elif order == 2:
        diag = np.ones(n)*20
        diag[:3] = [1,10,19]
        diag[-3:] = [19,10,1]
        diag1 = -np.ones(n-1)*30
        diag1[:2] = [-6,-24]
        diag1[-2:] = [-24,-6]
        diag2 = np.ones(n-2)*12
        diag2[0] = 6
        diag2[-1] = 6
        diag3 = -np.ones(n-3)*2
        A1 = sparse.diags([diag,diag1,diag2,diag3],[0,1,2,3])      
    else:
        raise NotImplementedError("Order >2 not implemented")
    return sparse.csc_matrix(A1)

def smooth(vec,order,weight):
    '''
    Use the selected derivative to smooth the input vector vec
    vec   : input vector 
    order : select the order of the derivative to use for smoothing
    weight: select the strenght of the smoothing 
    returns smoothed vector
    '''
    n = len(vec)
    vec = np.array(vec)
    
    A0 = sparse.csc_matrix(sparse.eye(n))
    A1 = construct_matrix(order,n)
    A = A0 + weight * A1
    A = ( A + A.T )
    b = 2 * vec
    return 2*A,linalg.spsolve(A,b)


class multi_smooth:
    '''

    The classs multi_smooth saves the sparse matrix. 
    Usfull if many vectors with the same paramters are processed
    '''
    def __init__(self, N, orders, weights):
        '''
        N     : lenght of the vectors to smoooth
        order : select the order of the derivative to use for smoothing
        weight: select the strenght of the smoothing 
        '''
        if not isinstance(orders, (collections.Sequence, np.ndarray)):
            orders = [orders]
        if not isinstance(weights, (collections.Sequence, np.ndarray)):        
            weights = [weights]*len(orders)
        
        A0 = sparse.csc_matrix(sparse.eye(N))    
        for order,weight in zip(orders,weights):
            A0 += weight * construct_matrix(order,N)
        self.A = ( A0 + A0.T )
    
    def __call__(self,vec):
        '''
        vec   : input vector 
        returns smoothed vector
        '''
        b = 2 * vec
        return linalg.spsolve(self.A,b)


