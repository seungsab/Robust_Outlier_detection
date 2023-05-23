clc ; close all ; clear all ; 
rng('default');

N1 = 200 ;  % The dimension of ambient space
n1 = 100 ;  % The number of inliers
n2 = 10000;  % The number of outliers
r = 5 ;     % The rank of low rank matrix
U = randn(N1,r) ; 

A = U*randn(r,n1) ; U = orth(U);
B = randn(N1,n2) ; 
D= [A  B] ;    % Given data

n = 10*3 ;     % Number of data points sampled by CoP algorithm
               % to form the recovered subspace
                
Uh = Coherence_pursuit(D , n, r) ;  


err = Uh - U*U'*Uh;

recovery_error = norm(err(:), 2)/norm(U(:),2) % Recovery error
