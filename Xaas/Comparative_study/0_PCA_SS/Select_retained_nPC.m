function n_pc1=Select_retained_nPC(variances,nPC_select_algorithm,target_CPV,m)
% m: # of input parameters
%% Extracting the retained PC
switch nPC_select_algorithm
    case 'CPV' % CPV: the cumulative percent variance (CPV) method
        target_CPV1=target_CPV(1);
        % EXPLAINED VARIANCE
        for i=1:size(target_CPV,2)
            explained=variances/sum(variances)*100;
            n_pc1=find(cumsum(explained)>target_CPV1==1);
            n_pc1=n_pc1(1);
            if n_pc1==m
                try
                    target_CPV1=target_CPV(i+1);
                catch
                    n_pc1=m-1;
                end
            end
        end
        
    case 'eigengap' % Score: Threshold-based method (Score rule)
        % ref.1) The rotation of eigenvectors by a perturbation (1970)
        % ref.2) Adaptive data-derived anomaly detection in the activated... (2016)
        [Y,IX] = max(abs(diff(variances)));
        n_pc1=IX(1);
        
    case 'Score' % Score: Threshold-based method (Score rule)
        % explained (n by 1 vector): the normalized variance from SVD (PCA)
        % m (scalar): # of input parameters
        maxValue=variances(1); secValue=variances(2); minValue=variances(end);
        numEigen=size( (variances>0), 1 );
        if numEigen<m
            threshold=(sum(variances)-maxValue-secValue)/numEigen;
        else
            threshold=(sum(variances)-maxValue-minValue)/numEigen;
        end
        ind=find(variances>threshold==1);
        n_pc1=ind(end);

end