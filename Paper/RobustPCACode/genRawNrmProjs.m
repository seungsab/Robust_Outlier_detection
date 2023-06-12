function [rawProjs, nrmProjs] = genRawNrmProjs(feats, labels, rnks)
[n,d] = size(feats);
rawProjs = struct([]);
nrmProjs = struct([]);

featNorms = sqrt(sum(feats.^2, 2));
featNorms(featNorms == 0) = 1;

[ftR, ftC, ftV] = find(feats);
ftV = ftV ./ featNorms(ftR);

nrmFeats = sparse(ftR, ftC, ftV, n, d);

fprintf('Computing Projections for Unnormalized Features\n');
[PCAProjs, cPCAProjs, rPCAProjs] = genProjs(feats, labels, rnks);
rawProjs(1).PCAProjs = PCAProjs;
rawProjs(1).cPCAProjs = cPCAProjs;
rawProjs(1).rPCAProjs = rPCAProjs;

fprintf('Computing Projections for Normalized Features\n');
[PCAProjs, cPCAProjs, rPCAProjs] = genProjs(nrmFeats, labels, rnks);
nrmProjs(1).PCAProjs = PCAProjs;
nrmProjs(1).cPCAProjs = cPCAProjs;
nrmProjs(1).rPCAProjs = rPCAProjs;
end