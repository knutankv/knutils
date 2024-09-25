import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import copy as cp

def equal_3d(ax=plt.gca()):
    x_lims = np.array(ax.get_xlim())
    y_lims = np.array(ax.get_ylim())
    z_lims = np.array(ax.get_zlim())

    x_range = np.diff(x_lims)
    y_range = np.diff(y_lims)
    z_range = np.diff(z_lims)

    max_range = np.max([x_range,y_range,z_range])/2

    ax.set_xlim(np.mean(x_lims) - max_range, np.mean(x_lims) + max_range)
    ax.set_ylim(np.mean(y_lims) - max_range, np.mean(y_lims) + max_range)
    ax.set_zlim(np.mean(z_lims) - max_range, np.mean(z_lims) + max_range)
    # ax.set_aspect(1)
    
    return ax


def plot_transformation_mats(x, y, z, T, figno=None, ax=None, scaling='auto', show_legend=False):
    if ax==None:
        fig = plt.figure(figno)
        ax = fig.add_subplot(111, projection='3d')

    # ax.scatter(x,y,z,'.k')

    if scaling=='auto':
        xr = max(x)-min(x)
        yr = max(y)-min(y)
        zr = max(z)-min(z)
        r = np.sqrt(xr**2+yr**2+zr**2)
        scaling = 0.005*r

    compcolors = ['tab:red', 'tab:blue', 'tab:green']
    h = [None]*3
    for ix, Ti in enumerate(T):
        xi = x[ix]
        yi = y[ix]
        zi = z[ix]
        
        for comp in range(0,3):
            xunit = [xi, xi+Ti[comp,0]*scaling]
            yunit = [yi, yi+Ti[comp,1]*scaling]
            zunit = [zi, zi+Ti[comp,2]*scaling]

            h[comp] = plt.plot(xs=xunit, ys=yunit, zs=zunit, color=compcolors[comp])[0]

    if show_legend:
        plt.legend(h,['x', 'y', 'z'])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    equal_3d(ax)
    return ax,h


def plot_elements_from_matrices(element_matrix, node_matrix, chosen_nodes_ix=[], disp=None, node_labels=False, element_labels=False, nodes=True, elements=True, ax=None, fig=None, element_settings={}, node_settings={}, node_label_settings={}, chosen_node_settings={}, disp_settings={}, element_label_settings={}, three_d=True, transformation_mats=None, tmat_scaling='auto'):
    e_dict = {'color': 'DarkGreen', 'alpha': 1}
    e_dict.update(**element_settings)

    n_dict = {'color':'Black', 'linestyle':'', 'marker':'.', 'markersize':4, 'alpha':0.8}
    n_dict.update(**node_settings)
       
    n_chosen_dict = {'color':'GreenYellow', 'linestyle':'', 'marker':'o', 'markersize':8, 'alpha':1, 'markeredgecolor':'dimgray'}
    n_chosen_dict.update(**chosen_node_settings)
    
    disp_dict = {'color':'IndianRed', 'alpha':1}
    disp_dict.update(**disp_settings)

    l_nodes_dict = {'color':'Black', 'fontsize': 8, 'fontweight':'normal'}
    l_nodes_dict.update(**node_label_settings)
    
    l_elements_dict = {'color':'LimeGreen', 'fontsize': 8, 'fontweight':'bold', 'style':'italic'}
    l_elements_dict.update(**element_label_settings)
    
    if ax is None and fig is None:
        fig = plt.figure()
        
    if ax == None and three_d:
        ax = fig.gca(projection='3d')
    elif ax == None:
        ax = fig.gca()
    elif three_d:
        1
        # ax.set(projection='3d')  #mangler funksjonalitet her...
    
    element_handles = [None]*len(element_matrix[:,0])

    if elements:
        if transformation_mats is not None:
            xm = np.zeros([len(element_matrix[:,0]), 3])
            
        for element_ix, __ in enumerate(element_matrix[:,0]):
            node1 = element_matrix[element_ix, 1]
            node2 = element_matrix[element_ix, 2]
            nodeix1 = np.where(node_matrix[:,0]==node1)[0]
            nodeix2 = np.where(node_matrix[:,0]==node2)[0]
            x01 = node_matrix[nodeix1,1:4]
            x02 = node_matrix[nodeix2,1:4]
            x0 = np.vstack([x01,x02])

            if transformation_mats is not None:
                xm[element_ix, :] = np.mean(x0, axis=0)

            if three_d:
                element_handles[element_ix] = ax.plot(xs=x0[:,0], ys=x0[:,1], zs=x0[:,2], **e_dict)
            else:
                element_handles[element_ix] = ax.plot(x0[:,0], x0[:,1], **e_dict)
  
            if element_labels:
                xmean = np.mean(x0, axis=0)
                if three_d:
                    ax.text(xmean[0],xmean[1],xmean[2],'%i' % element_matrix[element_ix,0], **l_elements_dict)
                else:
                    ax.text(xmean[0],xmean[1],s='%i' % element_matrix[element_ix,0], **l_elements_dict)
                
            if disp is not None:
                disp_node1 = disp[nodeix1[0]*6:(nodeix1[0]*6+6)]
                disp_node2 = disp[nodeix2[0]*6:(nodeix2[0]*6+6)]
                x1 = x01+disp_node1[0:3]
                x2 = x02+disp_node2[0:3]
                x = np.vstack([x1,x2])
                
                if three_d:
                    ax.plot(xs=x[:,0], ys=x[:,1], zs=x[:,2], **disp_dict)
                else:
                    ax.plot(x[:,0], x[:,1], **disp_dict)

    if transformation_mats is not None:
        plot_transformation_mats(xm[:, 0], xm[:, 1], xm[:, 2], transformation_mats, ax=ax, scaling=tmat_scaling, show_legend=True)

    if nodes:
        if three_d:
            ax.plot(xs=node_matrix[:, 1], ys=node_matrix[:, 2], zs=node_matrix[:, 3], **n_dict)
        else:
           ax.plot(node_matrix[:, 1], node_matrix[:, 2], **n_dict)
        
        if chosen_nodes_ix != []:
            if three_d:
                ax.plot(xs=node_matrix[chosen_nodes_ix, 1], ys=node_matrix[chosen_nodes_ix, 2], zs=node_matrix[chosen_nodes_ix, 3], **n_chosen_dict)
            else:
               ax.plot(node_matrix[chosen_nodes_ix, 1], node_matrix[chosen_nodes_ix, 2], **n_chosen_dict)
        
    if node_labels:
        if three_d:
            for node_ix in range(0, np.shape(node_matrix)[0]):
                ax.text(node_matrix[node_ix, 1], node_matrix[node_ix, 2], node_matrix[node_ix, 3], '%i' % node_matrix[node_ix, 0], **l_nodes_dict)
        else:
            for node_ix in range(0, np.shape(node_matrix)[0]):
                ax.text(node_matrix[node_ix, 1], node_matrix[node_ix, 2], '%i' % node_matrix[node_ix, 0], **l_nodes_dict)
        
    if three_d:
        equal_3d(ax)
    else:
        ax.set_aspect('equal', adjustable='box')
    
    ax.grid('off')
    return ax, element_handles


def figsave(name, w=16, h=10, fig=None, maketight=True, dpi=800, fileformat='png'):

    if fig is None:
        fig = plt.gcf()

    fig.set_figwidth(w/2.54), fig.set_figheight(h/2.54)

    if maketight:
        fig.tight_layout()

    fig.savefig(f'{name}.{fileformat}', dpi=dpi)