function [X0,PC,T2,Q]=MWPCA_update(X0,PC,T2,Q,alpha,SD_type,nPC_select_algorithm,target_CPV,ind_x);
%% Assign variables
[f,mu0,sds0]=deal(X0.f,X0.mu0,X0.sds0);
[loadings,n_pc,variances]=deal(PC.loadings,PC.n_pc,PC.variances);
[t2,tcrit,Qdist,distcrit]=deal(T2.t2,T2.tcrit,Q.Qdist,Q.distcrit);

%% Standardization (zero-mean)
numobs1=1; numvars=size(f,2);
X1std=Standardization_Data(f(end,:),mu0(end,:),sds0(end,:),numobs1,SD_type);

%% PC_PROJECTION
scores1=X1std*loadings(:,1:n_pc(end,:));
t2_val = sum((scores1(:,1:n_pc(end,:)).^2)./repmat(variances(1:n_pc(end,:))',numobs1,1),2);
residuals_val = (X1std - scores1(:,1:n_pc(end,:))*loadings(:,1:n_pc(end,:))');
Qdist_val = sqrt(sum(residuals_val.^2,2));

t2=[t2; t2_val]; Qdist=[Qdist; Qdist_val];

%% CHECK ANOMALITY
switch alpha
    case '95%'
        Q_lim=distcrit(end-PC.n_stall+1:end,1); T2_lim=tcrit(end-PC.n_stall+1:end,1);
    case '99%'
        Q_lim=distcrit(end-PC.n_stall+1:end,2); T2_lim=tcrit(end-PC.n_stall+1:end,2);
end

Q_check=Q.Qdist(end-PC.n_stall+1:end)<=Q_lim;
T2_check=T2.t2(end-PC.n_stall+1:end)<=T2_lim;

% Ref) 2006_Adaptive Multivariate Statistical Process Control for Monitoring Time-varying Process_ind.Eng.Chem.Res.
% If three consecutive out-of-control samples have been generated, the new
% sample is not outlier and it is a sample in abnormal condition.
% Otherwise, it is an outlier.
Normal_condition=0;
Evaluate_abnormality

if PC.update
    switch Normal_condition
        case 1 % Normal condition
            % Discard the initial sample of the current window
            f=f(2:end,:); n=size(f,1);
            
            % UPDATE MEAN (mu0) and SIGMA (sds0)
            mu0=[mu0; mean(f)];
            sds0=[sds0; std(f)];
            numobs=size(f,1);
            
            % Standardization
            X0std=Standardization_Data(f,mu0(end,:),sds0(end,:),numobs,SD_type);
            
            % UPDATE PC MODEL
            [loadings, scores, variances, tscores] = pca(X0std);
            
            % Extracting the retained PC
            n_pc1=Select_retained_nPC(variances,nPC_select_algorithm,target_CPV,size(f,2));
            n_pc=[n_pc; n_pc1(1,:)];
                        
            % Calculate Q-statistics
            residuals = (X0std - scores(:,1:n_pc(end,:))*loadings(:,1:n_pc(end,:))');
            Qdist1 = sqrt(sum(residuals.^2,2));
            m_Q=mean(Qdist1); V_Q=var(Qdist1);
            V=2*(m_Q^2)/V_Q;
            distcrit1 = V_Q/(2*m_Q)*...
                [chi2inv(0.95,V)... % 95%
                chi2inv(0.99,V)]; % 99%
            
            distcrit=[distcrit; distcrit1];
            
        case 0 % Abnormal condition
            % KEEP the initial sample of the current window and DISCARD last sample
            f=f(1:end-1,:); n=size(f,1);
            
            % KEEP the LAST MEAN (mu0) and SIGMA (sds0)
            mu0=[mu0; mean(f)]; sds0=[sds0; std(f)]; numobs=size(f,1);
            n_pc=[n_pc; n_pc(end,:)];
            
            % KEEP the  Hostelling T2 scores and Q-statistics
            tcrit=[tcrit; tcrit(end,:)];
            distcrit=[distcrit; distcrit(end,:)];
            
    end
else
    % KEEP the initial sample of the current window and DISCARD last sample
    f=f(1:end-1,:); n=size(f,1);
    
    % KEEP the LAST MEAN (mu0) and SIGMA (sds0)
    mu0=[mu0; mean(f)]; sds0=[sds0; std(f)]; numobs=size(f,1);
    n_pc=[n_pc; n_pc(end,:)];
    
    % KEEP the  Hostelling T2 scores and Q-statistics
    tcrit=[tcrit; tcrit(end,:)];
    distcrit=[distcrit; distcrit(end,:)];
    
end

%% Return variables
[X0.f,X0.mu0,X0.sds0]=deal(f,mu0,sds0);
[PC.loadings,PC.n_pc,PC.variances]=deal(loadings,n_pc,variances);
[T2.t2,T2.tcrit,Q.Qdist,Q.distcrit]=deal(t2,tcrit,Qdist,distcrit);

end