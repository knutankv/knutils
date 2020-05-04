import numpy as np

def normalize_phi(phi, include_dofs=[0,1,2,3,4,5,6], n_dofs=6):
    phi_n = phi*0

    phi_for_scaling = np.vstack([phi[dof::n_dofs, :] for dof in include_dofs])
    mode_scaling = np.max(np.abs(phi_for_scaling), axis=0)
    ix_max = np.argmax(np.abs(phi_for_scaling), axis=0)
    signs = np.sign(phi_for_scaling[ix_max, range(0, len(ix_max))])
    signs[signs==0] = 1
    mode_scaling[mode_scaling==0] = 1

    phi_n = phi/np.tile(mode_scaling[np.newaxis,:]/signs[np.newaxis,:], [phi.shape[0], 1])

    return phi_n, mode_scaling