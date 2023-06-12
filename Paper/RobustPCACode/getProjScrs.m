function [scrs] = getProjScrs(V, X, normalize)
if nargin < 3
    normalize = true;
end

Xnrms = sum(X.^2, 2);

Xprjs = X * V;
XprNrms = sum(Xprjs.^2, 2);


scrs = Xnrms - XprNrms;

if normalize
    scrs = scrs ./ Xnrms;
    scrs(Xnrms == 0) = 0;
end
end