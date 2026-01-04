function [x,esq,j] = k_means(d,k,x0)
% K-means 

[n,p] = size(d);
if nargin<3
   x = d(ceil(rand(1,k)*n),:);
else
   x=x0;
end
y = x+1;
iter = 0;
while any(x(:) ~= y(:))
   iter = iter + 1;
   z = disteusq(d,x,'x');
   [m,j] = min(z,[],2);
   y = x;
   for i=1:k
      s = j==i;
      if any(s)
         x(i,:) = mean(d(s,:),1);
      else
         q=find(m~=0);
         if isempty(q) break; end
         r=q(ceil(rand*length(q)));
         x(i,:) = d(r,:);
         m(r)=0;
         y=x+1;
      end
   end
end
esq=mean(m,1);