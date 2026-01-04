%% OUTPUT ONLY DATA-NORMALIZATION BY PCA
% Ref. of GOOD MATERIAL FOR PCA
% http://www.mathworks.com/matlabcentral/fileexchange/18930-multivariate-data-analysis-and-monitoring-for-the-process-industries/content/html/WebinarMSPMdemo.html

%% DEFINE THE METHODOLOGIES
clearvars -except f ts1 fn run_PCA Variable_window PCA_par L_min L_max init_train_ratio fn fn_type
SD_type=PCA_par.SD_type; alpha=PCA_par.alpha;
nPC_select_algorithm=PCA_par.nPC_select_algorithm; target_CPV=PCA_par.target_CPV;
Monitoring_statistic=PCA_par.Monitoring_statistic;
x0=PCA_par.x(1,1):PCA_par.x(1,2); x1=PCA_par.x(2,1):PCA_par.x(2,2);
x2=PCA_par.x(3,1):PCA_par.x(3,2);

%% Set training and validation sample interval
[numobs,numvars] = size(f);

%% CONSTRUCT INITIAL PC MODEL
Initial_learning_PCA

%% PERFORM PCA TO VALIDATION DATA
% USE THE FIXED PC MODEL FROM TRAINING DATA
X1=f(x1,:); [numobs1 numvar1]=size(X1);

% Standardization
X1std=Standardization_Data(X1,mu0,sds0,numobs1,SD_type);

scores1=X1std*loadings(:,1:n_pc);
t2_val = sum((scores1(:,1:n_pc).^2)./repmat(variances(1:n_pc)',numobs1,1),2);
residuals_val = (X1std - scores1(:,1:n_pc)*loadings(:,1:n_pc)');
Qdist_val = sqrt(sum(residuals_val.^2,2));
% Qdist_val = sqrt(sum(residuals_val.^2,2)./(numvar1-2));

%% PERFORM PCA TO TESTING DATA
X2=f(x2,:); [numobs2 numvar2]=size(X2);

% Standardization
X2std=Standardization_Data(X2,mu0,sds0,numobs2,SD_type);

scores2=X2std*loadings(:,1:n_pc);
t2_test = sum((scores2(:,1:n_pc).^2)./repmat(variances(1:n_pc)',numobs2,1),2);
residuals_test = (X2std - scores2(:,1:n_pc)*loadings(:,1:n_pc)');
Qdist_test = sqrt(sum(residuals_test.^2,2)./(numvar2-2));

T2.t2=[t2; t2_val; t2_test];
Q.Qdist=[Qdist; Qdist_val; Qdist_test];
T2.tcrit=repmat(tcrit,size(T2.t2,1),1);
Q.distcrit=repmat(distcrit,size(Q.Qdist,1),1);

%% PLOT MONITORING STATISTICS
if strcmp(alpha,'95%')
    Q_lim=Q.distcrit(:,1); T2_lim=T2.tcrit(:,1);
else
    Q_lim=Q.distcrit(:,2); T2_lim=T2.tcrit(:,2);
end

PC.fault=[]; PC.outlier=[];
if sum(PCA_par.Monitoring_statistic)==2
    ind_f1=find(Q.Qdist>Q_lim);
    ind_f2=find(T2.t2>T2_lim);
    PC.outlier=intersect(ind_f1,ind_f2);
elseif PCA_par.Monitoring_statistic(1)==1
    PC.outlier=find(Q.Qdist>Q_lim);
else
    PC.outlier=find(T2.t2>T2_lim);
end

T2.tcrit=[T2.tcrit; T2.tcrit(end,:)];
Q.distcrit=[Q.distcrit; Q.distcrit(end,:)];

%% Compute False alarm
for i=1:size(PCA_par.x,1)
    switch i
        case 1
            disp('%%%%% INITIAL PC %%%%%')
            n_out=sum((PC.outlier<=PCA_par.x(i,2)));
            n_tot=size(x0,2);
            
        case 2
            disp('%%%%% FALSE ALARM TEST %%%%%')
            n_out=sum((PC.outlier>=PCA_par.x(i,1)).*(PC.outlier<=PCA_par.x(i,2)));
            n_tot=size(x1,2);
            
    end
    disp(['False alarm:' num2str(n_out) '  //  Total sample:' num2str(n_tot)]);
    disp([num2str(n_out/n_tot*100) '%']);
end

disp('Done')

%% PLOT SELECTED # of PC
fn_fig=['SPCA n=' num2str(size(x0,2)) ' (#n0 = ' num2str(init_train_ratio) ')_' num2str(fn_type)];

fig_T_Q=figure;
Plot_monitoring_statistics


%% SAVE RESULTS
savefig([pwd '\Result\' fn_fig '.fig']);