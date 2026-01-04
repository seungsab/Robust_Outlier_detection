%% Training data set
X0=f(x0,:); [numobs,numvars] = size(X0);
mu0 = mean(X0); sds0 = std(X0);

% Standardization
% Mean_centered: only zero-mean (not scaled by variance)
% Z-score: zero-mean and unit variance
X0std=Standardization_Data(X0,mu0,sds0,numobs,SD_type);

% PERFORM PCA
[loadings, scores, variances, tscores] = pca(X0std);

% Extracting the retained PC
n_pc=Select_retained_nPC(variances,nPC_select_algorithm,target_CPV,size(f,2));

%% Calculate Hostelling T2 scores and critical 95, 99 and 99.9% limits
t2 = sum((scores(:,1:n_pc).^2)./repmat(variances(1:n_pc)',numobs,1),2);
tcrit = (n_pc*(numobs^2-1)/(numobs*(numobs-n_pc)))*...
    [finv(0.95,n_pc,numobs-n_pc),... % 95%
    finv(0.99,n_pc,numobs-n_pc)]; % 99%

%% Calculate squared residuals and critical 95, 99 and 99.9% limits
residuals = (X0std - scores(:,1:n_pc)*loadings(:,1:n_pc)');
Qdist = sqrt(sum(residuals.^2,2));
m_Q=mean(Qdist); V_Q=var(Qdist);
V=2*(m_Q^2)/V_Q;
distcrit = V_Q/(2*m_Q)*...
    [chi2inv(0.95,V)... % 95%
    chi2inv(0.99,V)]; % 99%