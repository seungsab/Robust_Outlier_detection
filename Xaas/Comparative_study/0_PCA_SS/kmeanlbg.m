function [x,esq,j] = kmeanlbg(d,k)
%KMEANLBG Linde-Buzo-Gray

nc=size(d,2);
[x,esq,j]=k_means(d,1);
m=1;
iter = 0;
while m<k
   iter = iter + 1;
   n=min(m,k-m);
   m=m+n;
   e=1e-4*sqrt(esq)*rand(1,nc);
   [x,esq,j]=k_means(d,m,[x(1:n,:)+e(ones(n,1),:); x(1:n,:)-e(ones(n,1),:); x(n+1:m-n,:)]);

end
