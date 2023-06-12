function [PCAProjs, cPCAProjs, rPCAProjs] = genProjs(feats, labels, rnks)
cFeats = feats(labels == 0, :);
crFeats = feats(labels == 1, :);

alpha = nnz(labels) / length(labels);
maxRnk = max(rnks);

fprintf('Computing SVD for All Instances\n');
[~, ~, V] = lansvd(feats, maxRnk, 'L');

fprintf('Computing SVD for Clean Instances\n');
[~, ~, cV] = lansvd(cFeats, maxRnk, 'L');

fprintf('Computing Robust SVD\n');
[rV, ~, ~] = crpca(feats', maxRnk, 20, alpha);

PCAProjs = struct([]);
cPCAProjs = struct([]);
rPCAProjs = struct([]);

nRnk = 0;

for rnk = rnks
    nRnk = nRnk + 1;
    
    clPCAProj = cFeats * V(:, 1:rnk);
    crPCAProj = crFeats * V(:, 1:rnk);
    
    PCAProjs(nRnk).clProj = sum(clPCAProj.^2, 2);
    PCAProjs(nRnk).crProj = sum(crPCAProj.^2, 2);
    
    clPCAProj = cFeats * cV(:, 1:rnk);
    crPCAProj = crFeats * cV(:, 1:rnk);
    
    cPCAProjs(nRnk).clProj = sum(clPCAProj.^2, 2);
    cPCAProjs(nRnk).crProj = sum(crPCAProj.^2, 2);
    
    clPCAProj = cFeats * rV(:, 1:rnk);
    crPCAProj = crFeats * rV(:, 1:rnk);
    
    rPCAProjs(nRnk).clProj = sum(clPCAProj.^2, 2);
    rPCAProjs(nRnk).crProj = sum(crPCAProj.^2, 2);
end
end