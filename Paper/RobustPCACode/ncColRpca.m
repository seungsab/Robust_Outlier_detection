function [U, C, Supp, frobErr, iters, time] = ncColRpca(M, k, alpha, nItr, beta, eps, epsC, tol)
if nargin < 8, tol = 1e-1; end
if nargin < 7, epsC = 1e-3; end
if nargin < 6, eps = 1e-10; end
if nargin < 5, beta = 5; end
if nargin < 4, nItr = 51; end

threshRed = 0.9;

tic;
[~, n] = size(M);
mxThresh = floor(alpha * n);

frobErr = inf;

%{
sigI = lansvd(M, 1, 'L')
thresh = sigI * beta * sqrt(k / n)

res = sqrt(sum(M.^2, 1));
max(res)
[~, srtI] = sort(res, 'descend');
nThresh = min(nnz(res > thresh), mxThresh)
Ct = zeros(size(M));
Ct(:, srtI(1:nThresh)) = M(:, srtI(1:nThresh));

cNrms = sqrt(sum(Ct.^2, 1));
Supp = nnz(cNrms > epsC);
nnz(Supp)
plot(cumsum(Supp));
% keyboard
%}

res = sqrt(sum(M.^2, 1));
[~, srtI] = sort(res, 'descend');
Ct = zeros(size(M));
Ct(:, srtI(1:mxThresh)) = M(:, srtI(1:mxThresh));

t = 1;
frobM = norm(M, 'fro');
rnk = 1;

while t < nItr && frobErr(t) / frobM >= eps
    [Ut, St, ~] = lansvd(M - Ct, rnk + 1, 'L');
    Lt = Ut * (transpose(Ut) * M);
    Dt = M - Lt;
    
    thresh = St(rnk + 1, rnk + 1) * beta * sqrt(k / n);
    
    res = sqrt(sum(Dt.^2, 1));
    [~, srtI] = sort(res, 'descend');
    nThresh = min(nnz(res > thresh), mxThresh);
    corCols = srtI(1:nThresh);
    Ct = zeros(size(M));
    Ct(:, srtI(1:nThresh)) = M(:, corCols);
    Lt(:, corCols) = 0;

    t = t + 1;
    frobErr(t) = norm(M - (Lt + Ct), 'fro');
    
    if ((frobErr(t - 1) - frobErr(t)) / frobErr(t - 1) < tol) && rnk < k
        St = lansvd(M - Ct, k, 'L');
        
        ratSig = St(rnk + 1:end)./[St(rnk + 2:end); St(end)];
        [~, mxId] = max(ratSig);
        rnk = rnk + mxId;
    elseif ((frobErr(t - 1) - frobErr(t)) / frobErr(t - 1) < tol) && rnk == k
        beta = threshRed * beta;
    end
end

[U, ~, ~] = lansvd(M - Ct, rnk, 'L');
C = Ct;
cNrms = sqrt(sum(C.^2, 1));
C(:, cNrms < epsC) = 0;
Supp = nnz(cNrms > epsC);
iters = length(frobErr);
time = toc;
end