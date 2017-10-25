"""
	Implements the Direct Bayesian Model Reduction algorithm (Gerber & Horenko PNAS 2017)
	
	Simon Olsson 2017
"""
import numpy as np

def dbmr_likelihood(Lambda, Gamma, Count):
    return np.einsum('ij,kj,ik->',Count, Gamma, np.log(np.maximum(Lambda, 1e-16)) )

def estimate_dbmr(C, K=1, convergence = 1e-16, max_iter = 1000):
    """
      Estimate Lambda and Gamma matrices according to DBMR algorithm outlined in 
      "Toward a direct and scalable identification of reduced models for categorical processes" 
      by Gerber & Horenko PNAS 2017, doi: 10.1073/pnas.1612619114.

      Arguments:
      =========================================== 
      C (N,N) numpy array (count matrix of transition counts between N discrete states)
      K: int (number of states of latent process)
      convergence: float (convergence criterion for difference in log-likelihood)
      max_iter: int (maximum number of iterations)

      Returns:
      =====================
      Lambda (N,K) numpy array
      Gamma (K,N) numpy array
      lls (niter) python list of log-likelihood as a function of iteration
    """
    N = np.shape(C)[0]
    
    # initialize lambda
    Lambda = np.random.random_sample(size = N*K).reshape((N, K))
    Lambda = Lambda/Lambda.sum(axis = 0)[None,:]
    
    #initialize Gamma
    Gamma = np.zeros((K, N)) + 1e-16
    for i,s in enumerate(np.argmax(np.einsum('ij,jk->ki', C, np.log(np.maximum(Lambda, 1e-16))), axis=0)):
        Gamma[s,i] = 1
        
    #Estimation loop
    lls=[]
    delta_ll = 1

    while delta_ll>convergence:
        Lambda_new_un = np.einsum('ij,kj->ki', C, Gamma)
        Lambda = (Lambda_new_un/(Lambda_new_un.sum(axis=1)[:, None])).T
        Gamma[:, :] = 1e-16 
        for i,s in enumerate(np.argmax(np.einsum('ij,jk->ki', C, np.log(np.maximum(Lambda, 1e-16))), axis=0)):
            Gamma[s, i] = 1
        _ll = dbmr_likelihood(Lambda, Gamma, C)
        if len(lls)==0:
            delta_ll = 1
        else:
            delta_ll = np.abs(_ll-lls[-1])
        lls.append(float(_ll))
        if len(lls)>max_iter:
            print("Did not converge to a delta log-likelohood below", convergence, "(%e)"%delta_ll,"within %i iterations. Stopping."%max_iter)
            break
    return Lambda, Gamma, lls

