import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#### 1. Define Muller-Brown Potential ####
# reference: https://doi.org/10.1089/cmb.2006.13.745
A=[-200,-100,-170,15];# define constants
a=[-1,-1,-6.5,0.7];
b=[0,0,11,0.6];
c=[-10,-10,-6.5,0.7];
x0=[1,0,-0.5,-1];
y0=[0,0.5,1.5,1];

def muller_brown_potential(x,y,perturb=None):  #perturb is a list [coeff_x,coeff_y]
    z=0
    for k in range(4):
        z+=A[k]*np.exp( a[k]*(x-x0[k])**2 + b[k]*(x-x0[k])*(y-y0[k])+c[k]*(y-y0[k])**2)
    if perturb != None:    #introduce 1st order perturbation
        z+=perturb[0]*x+perturb[1]*y
    return z

def muller_brown_force(x,y,perturb=None): #perturb is a list [coeff_x,coeff_y]
    dudx=0; dudy=0
    for k in range(4):
        p=A[k]*np.exp( a[k]*(x-x0[k])**2 + b[k]*(x-x0[k])*(y-y0[k])+c[k]*(y-y0[k])**2)
        x_specific=2*a[k]*(x-x0[k])  +  b[k]*(y-y0[k])
        y_specific=b[k]*(x-x0[k]) + 2*c[k]*(y-y0[k])
        dudx+=p*x_specific; dudy+=p*y_specific
    if perturb != None:    #introduce 1st order perturbation
        dudx+=perturb[0] ; dudy+=perturb[1]
    dudx=-1*dudx; dudy=-1*dudy
    return dudx, dudy
#### END 1 ####

#### 2. Define Integrator and Prepare for Simulations ####
#reference: https://doi.org/10.1080/00268978800101881
def propagate(x_n,y_n,x_n_1,y_n_1,dt,gamma=0.05,m=1,k=1,T=1,perturb=None):
    energy=muller_brown_potential(x_n,y_n,perturb)
    Fn=muller_brown_force(x_n,y_n,perturb) # systematic forces
    Fn_x=Fn[0]; Fn_y=Fn[1]
    Rn=np.random.normal(0,np.sqrt(2*m*gamma*k*T/dt)) # random forces 
    # update positions
    x_next=x_n+(x_n-x_n_1)*(1-0.5*gamma*dt)/(1+0.5*gamma*dt)+((dt**2)/m)*(Fn_x+Rn)/(1+0.5*gamma*dt)
    y_next=y_n+(y_n-y_n_1)*(1-0.5*gamma*dt)/(1+0.5*gamma*dt)+((dt**2)/m)*(Fn_y+Rn)/(1+0.5*gamma*dt)
    return x_next, y_next, energy

#reference: https://doi.org/10.1063/1.5063730
def simulate_short_trajectories(x_min_grid,x_max_grid,y_min_grid,y_max_grid,nsteps,dt,nstxout,perturb=None):
    # initiate random positions within grid square 
    x=[np.random.uniform(x_min_grid,x_max_grid),np.random.uniform(x_min_grid,x_max_grid)]
    y=[np.random.uniform(y_min_grid,y_max_grid),np.random.uniform(y_min_grid,y_max_grid)]
    energy=[muller_brown_potential(x[0],y[0],perturb), muller_brown_potential(x[1],y[1],perturb)]
    # define simulation parameters 
    nsteps=nsteps
    dt=dt 
    time=nsteps*dt
    gamma=0.05/dt
    m=1; k=1; T=1 # use reduced units 
    # run simulation
    nstep=1
    xout=[x[0]]; yout=[y[0]]; energyout=[energy[0]]
    while nstep <= nsteps-1: 
        if nstep <=10: 
            if energy[nstep] >100: # reject starting points with potential energies larger than 100 
                return 'try_again'
            else:
                pass 
        progress=propagate(x[nstep],y[nstep],x[nstep-1],y[nstep-1],dt,gamma,m,k,T,perturb)
        x.append(progress[0])
        y.append(progress[1])
        energy.append(progress[-1])
        if (nstep)%nstxout == 0: # store positions and energies every nstxout steps 
            xout.append(progress[0])
            yout.append(progress[1])
            energyout.append(progress[-1])
        nstep+=1
    return xout,yout,energyout

def run_simulate_short_trajectories(x_min,x_max,y_min,y_max,n_traj=10000,nsteps_per_trajectory=5,dt=0.001,nstxout=1,perturb=None):  
    coordinates_x=[];
    coordinates_y=[];
    energies=[];
    for traj in range(n_traj):
        if traj %100 ==0:
            print('running simulation ' + str(traj))
        out=simulate_short_trajectories(x_min,x_max,y_min,y_max,nsteps_per_trajectory,dt,nstxout,perturb)
        while out == 'try_again':
            out=simulate_short_trajectories(x_min,x_max,y_min,y_max,nsteps_per_trajectory,dt,nstxout,perturb)
        coordinates_x.append(out[0])
        coordinates_y.append(out[1])
        energies.append(out[2])
    return coordinates_x, coordinates_y, energies 

def check_state(traj): #np.shape(traj)=(n_trajectories, n_frames_per_trajectory , n_dimension_of_coordinates) 
    y=np.asarray(traj[:,:,1])
    a=np.empty(shape=np.shape(y))
    b=np.empty(shape=np.shape(y))
    for i in range(np.shape(y)[0]):
        for j in range(np.shape(y)[1]):
            if y[i,j] < 0.15: 
                a[i,j]=1; b[i,j]=0
            elif y[i,j] > 1.15: 
                b[i,j]=1; a[i,j]=0 
            else:
                a[i,j]=0; b[i,j]=0
    return a, b

#### END 2 ####



if __name__ == '__main__':
    # run simulations
    unperturb_simulation=run_simulate_short_trajectories(-2.5,1.5,-1.5,2.5,n_traj=10000,nsteps_per_trajectory=500,nstxout=100)
    perturb_simulation=run_simulate_short_trajectories(-2.5,1.5,-1.5,2.5,n_traj=10000,nsteps_per_trajectory=500,nstxout=100,perturb=[20,20])
    #output results 
    #unperturb
    coordinates_x_unperturb=unperturb_simulation[0]; coordinates_y_unperturb=unperturb_simulation[1]; energies_unperturb=unperturb_simulation[2]
    traj_unperturb=np.stack((coordinates_x_unperturb,coordinates_y_unperturb),axis=1) 
    traj_unperturb=np.swapaxes(traj_unperturb, 1, 2) # shape = (n_trajectories, n_frames_per_trajectory , n_dimension_of_coordinates)
    np.save("traj_unperturb.npy",traj_unperturb)
    np.save("energy_unperturb.npy", energies_unperturb) # shape = (n_trajectories, n_frames_per_trajectory)
    check_state_unperturb=check_state(traj_unperturb) # check if particles are in state A or B or neither 
    np.save("stateA_unperturb.npy",check_state_unperturb[0])
    np.save("stateB_unperturb.npy",check_state_unperturb[1])
    #perturb
    coordinates_x_perturb=perturb_simulation[0]; coordinates_y_perturb=perturb_simulation[1]; energies_perturb=perturb_simulation[2]
    traj_perturb=np.stack((coordinates_x_perturb,coordinates_y_perturb),axis=1) 
    traj_perturb=np.swapaxes(traj_perturb, 1, 2) # shape = (n_trajectories, n_frames_per_trajectory , n_dimension_of_coordinates)
    np.save("traj_perturb.npy",traj_perturb)
    np.save("energy_perturb.npy", energies_perturb) # shape = (n_trajectories, n_frames_per_trajectory)
    check_state_perturb=check_state(traj_perturb) # check if particles are in state A or B or neither 
    np.save("stateA_perturb.npy",check_state_perturb[0])
    np.save("stateB_perturb.npy",check_state_perturb[1])

