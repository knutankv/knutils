import numpy as np

def estimate_sixdofs(a, b, a_x, b_x, x0=None):
    # SUSPECTED AS NOT CORRECT! HAS TO BE VERIFIED/MODIFIED ...
    if x0 is None:
        x0 = np.array([0,0,0])
        
    trans_dofs = np.mean([a,b], axis=0)   
    rot_dofs = trans_dofs*0
    
    for k in range(a.shape[0]):
        rot_dofs[k,:] = np.cross(a[k, :], a_x-x0) + np.cross(b[k, :], b_x-x0)
        
    sixdofs = np.hstack([trans_dofs, rot_dofs])
    
    return sixdofs
    

def phi_by_dof(phi, n_dofs=6):

    phi_list = [None]*n_dofs
    for dof in range(0, n_dofs):
        phi_list[dof] = phi[dof::n_dofs,:]

    return phi_list


def shearframe(n, k, c, m, kg=0, relative_dampers=True):
    #
    #   ================  --> u_1
    #   |              | 
    #   |              |  k1
    #   ================  --> u_2
    #   |              |  k2
    #   :              :
    #   :              :
    #   :              :
    #   |              | 
    #   ================  --> u_n
    #   |              | k_n
    #   |              |
    # /////          /////
    #
    #

    if type(m)==int or type(m)==float:
        m = np.ones(n)*m

    if type(c)==int or type(c)==float:
        c = np.ones(n)*c

    if type(k)==int or type(k)==float:
        k = np.ones(n)*k
        
    if type(kg)==int or type(kg)==float:
        kg = np.ones(n)*kg
        
    if relative_dampers == False:
        C = np.diag(c)
    else:
        c_shift = np.insert(c[0:-1], 0, 0.0, axis=0)
        C = np.diag(c_shift + c)
        
    M = np.diag(m)

    k_shift = np.insert(k[0:-1], 0, 0.0, axis=0)
    kg_shift = np.insert(kg[0:-1], 0, 0.0, axis=0)
    
    K = np.diag(k_shift + k)
    Kg = np.diag(kg_shift + kg)
    
    for level in range(0, n-1):
        K[level, level+1] = -k[level]
        K[level+1, level] = -k[level]
        Kg[level, level+1] = -kg[level]
        Kg[level+1, level] = -kg[level]   
        
        if relative_dampers==True:
            C[level, level+1]= -c[level]
            C[level+1, level]= -c[level]

    return K, C, M, Kg