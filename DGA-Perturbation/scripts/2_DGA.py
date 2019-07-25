import numpy as np 
from sklearn.neighbors import KNeighborsClassifier 
import joblib
import torch
from math import erf
import matplotlib.pyplot as plt

#### 1. load trajectory data and define states ####
traj=np.load("../data/10000/traj_unperturb.npy"); traj=np.vstack(traj)
define_state_knn=joblib.load("../data/module/define_state_for_MB_potential.joblib")
predict_closest_boundary=joblib.load("../data/module/predict_closest_boundary.joblib")
boundary_points=np.load("../data/module/boundary.npy")
knn_result=define_state_knn.predict(traj)  # state a =1; state b =2 ;domain =3
state_a=np.zeros(np.shape(traj)[0]); state_a[np.where(knn_result[:,0]==1)] =1
state_b=np.zeros(np.shape(traj)[0]); state_b[np.where(knn_result[:,0]==2)] =1
domain=np.zeros(np.shape(traj)[0]); domain[np.where(knn_result[:,0]==3)] =1
complement=np.zeros(np.shape(traj)[0]); complement[np.where(domain==0)]=1
# boundary is the index for the closest boundary point 
boundary=predict_closest_boundary.predict(traj)[:,0]
#### END 1 ####

#### Test 1: predict correct closest boundary points ####
#plt.scatter(boundary_points[:,0], boundary_points[:,1])
#plt.scatter(traj[5][0], traj[5][1],c='r')
#plt.scatter(boundary_points[1933][0], boundary_points[1933][1],c='r')
#plt.show()
#### END Test 1 ####

#### 2. build gaussian basis ####
def build_gaussian_basis(traj,boundary_points,num_basis):
    num_frame=np.shape(traj)[0]
    # randomly pick 100 frames and compute their mean and variance to build gaussians
    var_list=[] ;  mu_list=[] ; basis=[]; 
    for x in range(num_basis):
        random=np.random.randint(0,num_frame,size=100)
        mu=np.mean(traj[random],axis=0)
        var=np.mean(np.var(traj[random],axis=0)) # np.var computes variance for x and y separately. we then average the variances
        mu_list.append(mu)
        var_list.append(var)
        # compute gaussian x scaling function
        for frame in range(num_frame):
            g=np.exp(-1*(np.linalg.norm(traj[frame]-mu)**2)/(2*var))
            s=1
            state=int(domain[frame])
            if state==1: # apply scaling function to points in Domain
                boundary_index=int(boundary[frame])
                s_factor=10
                s=0.5*(erf((s_factor)*np.linalg.norm(traj[frame] - boundary_points[boundary_index]) - 3.5) + 1)
            basis.append(g*s)
    basis=np.asarray(basis).reshape((num_frame,num_basis),order='F')
    return basis, var_list, mu_list
#### END 2 ####

#### Test 2: boundary scaling ####
# in_domain=np.where(domain==1)
# print(result[0][in_domain][[0, 2, 5, 21, 22]])
#### END Test 2 ####


