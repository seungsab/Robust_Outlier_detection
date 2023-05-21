%--------------------------------------------------------------------------
% This function implements the CoP (coherent pursuit) robust PCA algorithm, presented in 
% Rahmani, M., & Atia, G. (2016). Coherence pursuit: Fast, simple, and robust principal component analysis. arXiv preprint arXiv:1609.04789.

% D: The given data matrix

% n: The number of data points sampled by CoP algorithm as basis for the recovered subspace

% r: The rank of low rank component. Here it is assumed that the required rank is given. 
% If it is not given, the rank can be estimated using the data points with high coherence values. 

% U: The obtained basis for the recovered subspace

% If you use this code in your research/work, cite this paper:
% Rahmani, M., & Atia, G. (2016). Coherence pursuit: Fast, simple, and robust principal component analysis. arXiv preprint arXiv:1609.04789.

% Copyright @ Mostafa Rahmani, 2017

%--------------------------------------------------------------------------
%--------------------------------------------------------------------------

function U = Coherence_pursuit(D , n, r) 

n = fix(n) ; r = fix(r) ;

[N1,~] = size(D) ; 

T = repmat(sum(D.^2).^0.5 , N1 , 1) ;
X = D./T ; 

G = X'*X ; G = G - diag(diag(G)) ;
p = sum(G.^2) ; p = p/max(p) ; 

figure ; stem(p); title('The elements of vector p') ; grid on ; 

[~,b] = sort(p , 'descend') ;
Y = X(:, b(1:n)) ; 

[s,~,~] = svd(Y , 'econ') ; 

U = s(: , 1:r) ; 

