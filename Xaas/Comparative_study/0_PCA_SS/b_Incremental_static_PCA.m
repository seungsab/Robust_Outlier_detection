%% INCREMENTAL_STATIC_PCA
% Ref. of GOOD MATERIAL FOR PCA
% http://www.mathworks.com/matlabcentral/fileexchange/18930-multivariate-data-analysis-and-monitoring-for-the-process-industries/content/html/WebinarMSPMdemo.html

%% DEFINE THE METHODOLOGIES
clearvars -except f ts1 fn run_PCA Variable_window PCA_par L_min L_max init_train_ratio fn fn_type
SD_type=PCA_par.SD_type; alpha=PCA_par.alpha;
nPC_select_algorithm=PCA_par.nPC_select_algorithm; target_CPV=PCA_par.target_CPV;
Monitoring_statistic=PCA_par.Monitoring_statistic;
x0=PCA_par.x(1,1):PCA_par.x(1,2); x1=PCA_par.x(2,1):PCA_par.x(2,2);
x2=PCA_par.x(3,1):PCA_par.x(3,2);
PC.n_stall=PCA_par.n_stall;
PC.Monitoring_statistic=PCA_par.Monitoring_statistic;

%% Set training and validation sample interval
[numobs,numvars] = size(f);

%% CONSTRUCT INITIAL PC MODEL
Time_effors=[];
tstart = tic;
Initial_learning_PCA
telapsed = toc(tstart);
Time_effors=[Time_effors; telapsed];

%% Validation data set
X1.f=X0; X1.mu0=repmat(mu0,size(t2,1),1); X1.sds0=repmat(sds0,size(t2,1),1);
PC.loadings=loadings; PC.scores=scores; PC.n_pc=n_pc; PC.variances=variances; PC.tscores=tscores;
PC.update=1; T2.t2=t2; T2.tcrit=repmat(tcrit,size(t2,1),1);
Q.Qdist=Qdist; Q.m_Q=m_Q; Q.V_Q=V_Q; Q.distcrit=repmat(distcrit,size(t2,1),1);
T2.tcrit=[T2.tcrit; T2.tcrit(end,:)];
Q.distcrit=[Q.distcrit; Q.distcrit(end,:)];

PC.fault=[]; PC.outlier=[]; 
if strcmp(alpha,'95%'), Q_lim=Q.distcrit(end,1); T2_lim=T2.tcrit(end,1);
else, Q_lim=Q.distcrit(end,2); T2_lim=T2.tcrit(end,2); end

if sum(PCA_par.Monitoring_statistic)==2
    ind_f1=find(Q.Qdist>Q_lim);
    ind_f2=find(T2.t2>T2_lim);
    PC.outlier=intersect(ind_f1,ind_f2);
elseif PCA_par.Monitoring_statistic(1)==1
    PC.outlier=find(Q.Qdist>Q_lim);
else
    PC.outlier=find(T2.t2>T2_lim);
end

for i=1:size(x1,2)
    X1.f=[X1.f; f(x1(i),:)];
    tstart = tic;
    [X1,PC,T2,Q]=AU_SPCA_update(X1,PC,T2,Q,alpha,SD_type,nPC_select_algorithm,target_CPV,x1(i));
    telapsed = toc(tstart);
    Time_effors=[Time_effors; telapsed];
end

for i=1:size(x2,2)
    X1.f=[X1.f; f(x2(i),:)];
    tstart = tic;
    [X1,PC,T2,Q]=AU_SPCA_update(X1,PC,T2,Q,alpha,SD_type,nPC_select_algorithm,target_CPV,x2(i));
    telapsed = toc(tstart);
    Time_effors=[Time_effors; telapsed];
end

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

%% PLOT results
fn_fig=['ISPCA n=' num2str(size(x0,2)) ' (#n0 = ' num2str(init_train_ratio) ')_' num2str(fn_type)];

%% PLOT MONITORING STATISTICS
fig_T_Q=figure;
Plot_monitoring_statistics

%% SAVE RESULTS
savefig([pwd '\Result\' fn_fig '.fig']);