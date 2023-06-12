%%	What is this file?
%%
%%	Author: 
%%	Email:
%%	Date:
%%

function [U, C, Supp] = crpca(M, k, nIter, alpha)
[d, n] = size(M);
nCor = floor(alpha * n);
res = sqrt(sum(M.^2, 1));
[~, srtI] = sort(res, 'descend');
C = sparse(d, n);
C(:, srtI(1:nCor)) = M(:, srtI(1:nCor));

Mnorms = sum(M.^2, 1);

for itr = 1:nIter
    
    Mt = M - C;
    
    if d < 500
        [U, ~, ~] = lansvd(Mt * transpose(Mt), k, 'L');
    else
        [U, ~, ~] = lansvd(Mt, k, 'L');
    end
    
    projs = transpose(U) * M;
    projLs = sum(projs.^2, 1);
    resNrms = Mnorms - projLs;
    
    [~, srtI] = sort(resNrms, 'descend');
    C = sparse(d, n);
    C(:, srtI(1:nCor)) = M(:, srtI(1:nCor));
end

Mt = M - C;

if d < 500
    [U, ~, ~] = lansvd(Mt * transpose(Mt), k, 'L');
else
    [U, ~, ~] = lansvd(Mt, k, 'L');
end
    
projs = transpose(U) * M;
projLs = sum(projs.^2, 1);
resNrms = Mnorms - projLs;
[~, srtI] = sort(resNrms, 'descend');

C = sparse(d, n);
C(:, srtI(1:nCor)) = M(:, srtI(1:nCor));
Supp = zeros(n, 1);
Supp(srtI(1:nCor)) = 1;
end