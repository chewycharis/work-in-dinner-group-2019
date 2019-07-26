import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.neighbors import KNeighborsClassifier
from muller_brown_potential_simulations import muller_brown_potential

#### 1.Make KNN model for state prediction ####  
# compute muller-brown potential 
X=np.arange (-2.5,1.5,1/100) ; Y=np.arange(-1.5,2.5,1/100)
X, Y = np.meshgrid(X, Y)
Z=np.array(muller_brown_potential(np.ravel(X), np.ravel(Y)))
Z = Z.reshape(X.shape)

# define domain and complement 
dc_index=np.where(Z <-60) ; d_index=np.where(Z>=-60) # boundary : E = 60
x_dc=X[0][dc_index[1]]; y_dc=Y[:,0][dc_index[0]]
x_d=X[0][d_index[1]] ; y_d=Y[:,0][d_index[0]]
d=np.stack((x_d,y_d)).T ; dc=np.stack((x_dc,y_dc)).T

# define state A and B
a=dc[np.where(dc[:,0]<0)] 
a=a[np.where(a[:,1]>a[:,0]+np.full(np.shape(a[:,0]),1.4))]
b=dc[np.where(dc[:,0]>-0.8)]
b=b[np.where(b[:,1]<b[:,0]+np.full(np.shape(b[:,0]),1.4))]

# define training and target sets 
dc_train = np.vstack((a, b)) 
train=np.vstack((d,dc_train))
dc_target = np.vstack((np.full(np.shape(a),1), np.full(np.shape(b),2)))
target=np.vstack((np.full(np.shape(d),3),dc_target))

# define testing data 
data=np.vstack((X.ravel(),Y.ravel())).T

# produce knn model 
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(train,target)
joblib.dump(knn, 'define_state_for_MB_potential.joblib')
#### END 1 ####


#### 2. Make KNN model for finding closest boundary points ####
# use a larger data set 
X=np.arange (-2.5,1.5,1/1000) ; Y=np.arange(-1.5,2.5,1/1000)
X, Y = np.meshgrid(X, Y)
Z=np.array(muller_brown_potential(np.ravel(X), np.ravel(Y)))
Z = Z.reshape(X.shape)

#find possible boundary points 
boundary_index=[]
for x in range(4000):
    for y in range(4000):
        if Z[x,y]<-59.5 and Z[x,y] > -60.05:
            boundary_index.append([x,y])
boundary_index=np.asarray(boundary_index)
find_boundary=np.vstack((X[0][boundary_index[:,1]],Y[:,0][boundary_index[:,0]])).T

# filter boundary points such that they only exist in Domain 
result=knn.predict(find_boundary)[:,0]
boundary=find_boundary[np.where(result==3)]
np.save("boundary.npy",boundary)

# create target and training sets 
target2=np.vstack((range(np.shape(boundary)[0]),range(np.shape(boundary)[0]))).T
train2=boundary

#use KNN classifier to predict states 
knn2 = KNeighborsClassifier(n_neighbors=10) 
knn2.fit(train2,target2)
joblib.dump(knn,'predict_closest_boundary.joblib')
#### END 2 ####

