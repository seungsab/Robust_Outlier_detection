function [auc, Par, Nacc] = compAucPNac(scores, labs)
nCln = nnz(labs == 0);
nAnom = nnz(labs == 1);
nInst = nCln + nAnom;

[~, srtI] = sort(scores, 'descend');
srtLabs = labs(srtI);

cln = 1 - srtLabs;
clnRev = cln(nInst:-1:1);
cumCR = cumsum(clnRev);
cumCln = cumCR(nInst:-1:1);
% cumCln = cumsum(cln, 'reverse');

auc = (transpose(srtLabs) * cumCln) / (nCln * nAnom);
pred = [ones(nAnom, 1); zeros(nCln, 1)];

Par = sum(srtLabs(1:nAnom)) / nAnom;
Nacc = nnz(pred == srtLabs) / nCln;

end