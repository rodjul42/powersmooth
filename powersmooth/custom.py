from scipy import sparse
from scipy.sparse import linalg
import numpy as np
import collections
from sympy import expand


class custom_smooth:
    def __init__(self, N, diffs, weights):
        A0 = sparse.csc_matrix(sparse.eye(N))    
        if not isinstance(diffs, (collections.Sequence, np.ndarray)):
            diffs = [diffs]
        if not isinstance(weights, (collections.Sequence, np.ndarray)):        
            weights = [weights]*len(diffs)
        for weight,diff in zip(weights,diffs):
            res = expand((diff)**2)
            factork = []
            rowk = []
            colk = []
            syms = [s for s in res.atoms() if s.is_symbol ]
            for a_i,a in enumerate(syms):
                for b in syms[a_i:]:
                    factork += [res.coeff(a*b)]
                    rowk += [a.name.replace("p","+").replace("m","-")]
                    colk += [b.name.replace("p","+").replace("m","-")]
            delta_min = -np.min([eval(j,{"i":0}) for j in rowk+colk])
            delta_max = np.max([eval(j,{"i":0}) for j in rowk+colk])
            delta = delta_max + delta_min

            kl = len(factork) 

            row  = np.zeros((N,kl))
            col  = np.zeros((N,kl))
            data = np.zeros((N,kl))
            for i in range(delta_min,N-delta_max):
                row[i]  = [eval(j,{"i":i}) for j in rowk]
                col[i]  = [eval(j,{"i":i}) for j in colk]
                data[i] = factork


            for i in range(kl):
                A0 += weight*sparse.csc_matrix((data[:,i], (row[:,i], col[:,i])), shape=(N, N))
            self.A = ( A0 + A0.T )
            
    def __call__(self,vec):
        b = 2 * vec
        return linalg.spsolve(self.A,b)


