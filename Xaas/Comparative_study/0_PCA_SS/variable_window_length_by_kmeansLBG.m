function X1=variable_window_length_by_kmeansLBG(X1,PC,SD_type);
k=PC.k_cluster;

if size(X1.f0,1)>X1.L_max
    X1.f0=X1.f0(end-X1.L(end)+1:end,:);
end
numobs=size(X1.f0,1);

X=Standardization_Data(X1.f0,X1.mu0(end,:),X1.sds0(end,:),numobs,SD_type);

IDX=[]; CENTROID={}; F=[];
for i=1:size(k,2)
    if k(i)==1
        [ctrs,esq,idx] = kmeanlbg(X,k(i));
        IDX=[IDX idx]; CENTROID{i}=ctrs;
        BIC(i)=Bayes_Information_Criterion(X,idx,ctrs);
    else
        [ctrs,esq,idx] = kmeanlbg(X,k(i));
        IDX=[IDX idx]; CENTROID{i}=ctrs;
        BIC(i)=Bayes_Information_Criterion(X,idx,ctrs);
    end
end

[A ind]=max(BIC); 
idx=IDX(:,ind); n_cluster=k(ind);

if n_cluster==1
    L=size(find(idx==idx(end)),1);
else
    L=size(find(idx==idx(end)),1);
end

X1.N_cluster=[X1.N_cluster; n_cluster]; 
X1.L=[X1.L; L]; X1.BIC=[X1.BIC; BIC];
    
if isfield(X1,'L0')
    X1.L0=[X1.L0; X1.L(end)];
end



% figure;
% plot(k,BIC,'bx-','linewidth',2); hold on
% plot(k(ind),BIC(ind),'ro','linewidth',2,'markersize',15)
% set(gca,'fontsize',15,'fontweight','bold'); set(gca,'xtick',k)
% disp(['# of cluster: ' num2str(k(ind))])