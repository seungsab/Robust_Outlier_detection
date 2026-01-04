function BIC=Bayes_Information_Criterion(X,idx,ctrs);
% http://www.cs.joensuu.fi/sipu/pub/KneePointBIC-ICTAI-2008.pdf
% Data Mining and Predictive Analytics, Larose
% m: the number of cluster
% n: the number of the samples
% sigma_i:the variance of i-th cluster
% C_i: i-th cluster center
% ni: the number of i-th cluster
% xj: the j-th point in the cluster
% d is the dimension of the sample

[n d]=size(X); m=size(ctrs,1);

BIC=0;
for i=1:m
    index_i=find(idx==i); % index of i-th clustering
    Xi=X(index_i,:); % sample of the i-th cluster
    ni=size(Xi,1); % number of samples in the i-th index
    mi=mean(Xi); % mean of the i-th cluster
    
    sigma_i=1/(ni-m)*sum((sum((Xi-repmat(mean(Xi),size(Xi,1),1)).^2,2)));
    BIC=BIC+ni*log(ni/n)-ni*d/2*log(2*pi)-ni/2*log(sigma_i)-(ni-m)/2;
end
BIC=BIC-m*log(n)/2;

if ~isreal(BIC)
    BIC
end
end