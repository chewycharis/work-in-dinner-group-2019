import matplotlib.pyplot as plt
import numpy as np
import pyedgar
from pyedgar.data_manipulation import tlist_to_flat, flat_to_tlist

## code taken from Erik Thiede's Committor 1d tutorial for PyEDGAR

#### 1. import data #### 
trajs_perturb = np.ndarray.tolist(np.load('/home/chuhui/data/traj_perturb.npy'))
stateA_perturb = np.ndarray.tolist(np.load('/home/chuhui/data/stateA_perturb.npy'))
stateB_perturb = np.ndarray.tolist(np.load('/home/chuhui/data/stateB_perturb.npy'))
in_domain_perturb= [1. - B_i - A_i for (A_i, B_i) in zip(stateA_perturb, stateB_perturb)]

trajs_unperturb = np.ndarray.tolist(np.load('/home/chuhui/data/traj_unperturb.npy'))
stateA_unperturb = np.ndarray.tolist(np.load('/home/chuhui/data/stateA_unperturb.npy'))
stateB_unperturb = np.ndarray.tolist(np.load('/home/chuhui/data/stateB_unperturb.npy'))
in_domain_unperturb= [1. - B_i - A_i for (A_i, B_i) in zip(stateA_unperturb, stateB_unperturb)]
#### END 1 ####

#### 2. run pyedgar ####
diff_atlas_perturb = pyedgar.basis.DiffusionAtlas.from_sklearn(alpha=0, k=64, bandwidth_type='-1/d', epsilon='bgh_generous')
diff_atlas.fit(trajs_perturb)
basis_perturb, evals_perturb = diff_atlas_perturb.make_dirichlet_basis(300, in_domain=in_domain_perturb, return_evals=True)
guess_perturb = diff_atlas_perturb.make_FK_soln(stateB_perturb, in_domain=in_domain_perturb)
g_perturb = pyedgar.galerkin.compute_committor(basis_perturb, guess_perturb, lag=1)

diff_atlas_unperturb = pyedgar.basis.DiffusionAtlas.from_sklearn(alpha=0, k=64, bandwidth_type='-1/d', epsilon='bgh_generous')
diff_atlas_unperturb.fit(trajs_unperturb)
basis_unperturb, evals_unperturb = diff_atlas_unperturb.make_dirichlet_basis(300, in_domain=in_domain_unperturb, return_evals=True)
guess_unperturb = diff_atlas_unperturb.make_FK_soln(stateB_unperturb, in_domain=in_domain_unperturb)
g_unperturb = pyedgar.galerkin.compute_committor(basis_unperturb, guess_unperturb, lag=1)
#### END 2 ####

#### 3. save output ####
np.save("g_unperturb.npy",g_unperturb)
np.save("g_perturb.npy",g_perturb)
#### END 3 ####

