# spectral-network-identification

Identification of the Laplacian eigenvalues of a network from measurements of the dynamics on a small number of nodes.

Main codes for spectral network identification:
- linear_network_ident.m : spectral identification of a random network with linear dynamics
- nonlinear_network_ident.m : spectral identification of a random network with nonlinear (cubic) dynamics

Matlab functions:
- dmd_algo.m : Dynamic Mode Decomposition (DMD) algorithm (exctrats the Koopman spectrum from data)
- nonlinear_network_model : nonlinear dynamics used with nonlinear_network_ident.m

Reference: 
A. Mauroy and J. Hendrickx, Spectral identification of networks using sparse measurements, SIAM Journal on Applied Dynamical Systems, vol. 16, no. 1, pp. 479-513, 2017
