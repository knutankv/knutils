# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 12:23:47 2020

@author: KAK
"""

import numpy as np
from .tools import print_progress

def construct_L(n_dofs, sensor_ixs, dofs_per_sensor=1):
    L_out = np.eye(n_dofs)
    n_sensors = n_dofs/dofs_per_sensor
    non_sensor_ixs = np.setdiff1d(np.arange(0, n_sensors), np.array(sensor_ixs))
    non_dof_ixs = node_ix_to_dof_ix(non_sensor_ixs, n_dofs=dofs_per_sensor)
    L_out = np.delete(L_out, non_dof_ixs, axis=0)

    return L_out


def fssp(phi, cov, dofs_per_sensor=1, n_max_sensors=None):    
    n_tot_dofs = np.shape(phi)[0]
    n_all_sensors = int(n_tot_dofs/dofs_per_sensor)
    
    if n_max_sensors is None:
        n_max_sensors = n_all_sensors + 0 #copy
    
    optimal_sensor_ix = [None]*(n_max_sensors+1)
    optimal_sensor_ix[0] = []

    L = [None]*(n_max_sensors+1)
    L[0] = np.zeros([0, n_tot_dofs])

    all_sensor_ix = np.arange(0, n_all_sensors, 1)
    current_valid_sensor_ix = all_sensor_ix*1   #make copy
    
    for add_count in range(0, n_max_sensors):
        val = [None]*(n_all_sensors-add_count)

        for check_ix, sensor_ix in enumerate(current_valid_sensor_ix):
            print_progress(add_count+check_ix/len(current_valid_sensor_ix), n_max_sensors, sym='>', postfix = '   %4i/%4i sensor positions checked for %4i/%4i sensors' %(check_ix+1,len(current_valid_sensor_ix),add_count+1,n_max_sensors))
            this_sensor_ix = np.hstack([optimal_sensor_ix[add_count], sensor_ix])
            thisL = construct_L(n_tot_dofs, this_sensor_ix, dofs_per_sensor=dofs_per_sensor)
            val[check_ix] = L_objfun(thisL, phi, cov)

        ixmax = np.argmax(val)
        optimal_sensor_ix[add_count+1] = np.sort(np.hstack([optimal_sensor_ix[add_count], current_valid_sensor_ix[ixmax]])).astype(int)
        
        L[add_count+1] = construct_L(n_tot_dofs, optimal_sensor_ix[add_count+1], dofs_per_sensor=dofs_per_sensor)    
        current_valid_sensor_ix = np.setdiff1d(all_sensor_ix, optimal_sensor_ix[add_count+1])
        
    return optimal_sensor_ix[1:], L[1:]


def bssp(phi, cov, dofs_per_sensor=1, arrange_by_sensor_number=True, n_max_sensors=None):
        
    n_tot_dofs = np.shape(phi)[0]
    n_tot_sensors = int(n_tot_dofs/dofs_per_sensor)

    optimal_sensor_ix = [None]*n_tot_sensors
    optimal_sensor_ix[0] = np.arange(0, n_tot_sensors).astype(int)
    
    L = [None]*n_tot_sensors
    L[0] = np.eye(n_tot_dofs)
    
    for del_count in range(0, n_tot_sensors-1):
        val = [None]*(n_tot_sensors-del_count)
        current_valid_sensor_ix = optimal_sensor_ix[del_count]

        for check_ix, sensor_ix in enumerate(current_valid_sensor_ix):
            print_progress(del_count, n_tot_sensors, sym='>', postfix = '   %4i/%4i sensor positions checked for %4i/%4i sensors' %(check_ix+1, len(current_valid_sensor_ix), del_count+1, n_tot_sensors))
            this_sensor_ix = np.delete(current_valid_sensor_ix, check_ix, axis=0)     

            thisL = construct_L(n_tot_dofs, this_sensor_ix, dofs_per_sensor=dofs_per_sensor)
            val[check_ix] = L_objfun(thisL, phi, cov)

        ixmax = np.argmax(val)  
        optimal_sensor_ix[del_count+1] = np.delete(optimal_sensor_ix[del_count], ixmax, axis=0)
        L[del_count+1] = construct_L(n_tot_dofs, optimal_sensor_ix[del_count+1], dofs_per_sensor=dofs_per_sensor)

    if n_max_sensors!=None:
        L = L[-n_max_sensors:]
        optimal_sensor_ix = optimal_sensor_ix[-n_max_sensors:]

    if arrange_by_sensor_number:
        L = L[::-1]
        optimal_sensor_ix = optimal_sensor_ix[::-1]

    return optimal_sensor_ix, L


def single_one(N, ix, axis=0):
    arr = np.zeros(N)
    arr[ix] = 1
    
    for __ in range(0, axis):
        arr = np.expand_dims(arr, -1)
        
    arr = np.moveaxis(arr, 0, axis)
    
    return arr


def pseudo_det(M):
    # Calculates absolute value of determinant
    d = np.linalg.svd(M, compute_uv=False)
    detM = np.prod(d)

    return detM

def L_objfun(L, phi, cov):
    # Cholesky decomposition to enable inversion. See Eq. 14 -- 16 in Vincenzi and Simonini (2017).
    # Enables skipping computation of FIM.

    if np.shape(L)[0] == 0:
        val = 0    
    else:
        C = np.linalg.cholesky(np.linalg.inv(L @ cov @ L.T))
        val = pseudo_det(C.T @ (L @ phi))

    return val


def xdistances(x, y, z):
    dx = x[np.newaxis,:] - x[np.newaxis,:].T
    dy = y[np.newaxis,:] - y[np.newaxis,:].T
    dz = z[np.newaxis,:] - z[np.newaxis,:].T
    
    sigma = np.sqrt(dx**2 + dy**2 + dz**2)
    
    return sigma
    
    
def spatial_cov(x, y, z, corr_len=1.0, phi=None, dofs_per_sensor=1):
    
    delta = xdistances(x, y, z)    
    if phi is not None:
        
        scaling = np.tile(np.linalg.norm(phi,axis=1), [phi.shape[0], 1])
        scaling = np.maximum(scaling, scaling.T)
        
        sigma_ij = (np.abs(phi) @ np.abs(phi).T)/scaling**2

        corr_len = np.amax(delta)/(np.shape(phi)[0]/dofs_per_sensor)
        
        # n_modes = np.shape(phi)[1]
        # n_dofs = np.shape(phi)[0]
        # sigma_ij = np.zeros([n_dofs, n_dofs])
        
        # for i in range(0, n_dofs):
        #     phi_i = phi[i, :]
        #     for j in range(0, n_dofs):
        #         phi_j = phi[j, :]
        #         scaling = np.max([np.linalg.norm(phi_i), np.linalg.norm(phi_j)])
        #         sigma_ij[i,j] = np.dot(np.abs(phi_i), np.abs(phi_j))/(scaling**2)

        cov = sigma_ij * np.exp(-delta/corr_len)
    else:
        cov = np.exp(-delta/corr_len)

    return cov

    
def fim(L, phi, cov):
    # Fischer information matrix. Eq. 9, Vincenzi and Simonini (2017).
    Q = (L @ phi).T @ np.linalg.inv((L @ cov @ L.T)) @ (L @ phi)

    return Q


def ie_ratio(Ls, phi, cov, Lref=None):

    if Lref is None:
        Lref = np.eye(np.shape(phi)[0])

    N_theta = np.shape(phi)[1]  

    iei = [None]*len(Ls)
    for ix,Li in enumerate(Ls):
        iei[ix] = np.exp((infoentropy(Li, phi, cov)-infoentropy(Lref,phi,cov))/N_theta)


    return iei



def infoentropy(L, phi, cov, use_pseudo_det=True, tol=None):
    # Asymptotic estimate of information entropy. Eq. 5, Vincenzi and Simonini (2017). Not used directly.
    N_theta = np.shape(phi)[1]
    Q = fim(L, phi, cov)
    
    if use_pseudo_det:
        detQ = pseudo_det(Q)   
    elif tol!=None:
        eigvals,__ = np.linalg.eig(Q)
        good_eigs = np.where(np.abs(eigvals)>tol)[0]
        detQ = np.prod(eigvals[good_eigs])
    else:
        detQ = np.linalg.det(Q)

    H = N_theta * np.log(2*np.pi)/2 - np.log(detQ)/2

    return H
    

def strike_dofs(mat, dofs, vector=False):
    mat_out = np.delete(mat, dofs, axis=0)
    if vector == False:
        mat_out = np.delete(mat_out, dofs, axis=1)
    return mat_out


def node_ix_to_dof_ix(node_ix, n_dofs=6):
    start = node_ix*n_dofs
    stop = node_ix*n_dofs+n_dofs
    dof_ix = []
    for (i,j) in zip(start,stop):
        dof_ix.append(np.arange(i,j))

    dof_ix = np.array(dof_ix).flatten()

    return dof_ix