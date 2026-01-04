%% Moving WINDOW PCA
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
PC.k_cluster=1:4;

%% Set training and validation sample interval
[numobs,numvars] = size(f);

%% Training data set
Time_effors=[];
tstart = tic;
Initial_learning_PCA
telapsed = toc(tstart);
Time_effors=[Time_effors; telapsed];

X1.f0=X0; X1.mu0=repmat(mu0,size(t2,1),1); X1.sds0=repmat(sds0,size(t2,1),1);
PC.loadings=loadings; PC.scores=scores; PC.n_pc=n_pc; PC.variances=variances; PC.tscores=tscores;
PC.update=1; T2.t2=t2; T2.tcrit=repmat(tcrit,size(t2,1),1);
Q.Qdist=Qdist; Q.m_Q=m_Q; Q.V_Q=V_Q; Q.distcrit=repmat(distcrit,size(t2,1),1);

%% Update PCA model with Validation and testing data
if Variable_window
    % Set the parameters in variable window length
    % Ref) 2008_Variable MWPCA for Adaptive process Monitoring
    % r: the number of the PC of the kth PCA
    % m: the number of the variables
%     r=n_pc; m=size(f,2); L_th=(r+2*r*m-r^2)/(2*m);
%     L_min=round(10*L_th); L_max=round(80*L_th);% 4~6*L_th for L_min, 60~80*L_th for L_max

    X1.L_min=L_min; X1.L_max=L_max; X1.L=[]; 
    X1.N_cluster=[]; X1.BIC=[];
    
    if size(t2,1)<X1.L_max
        X1.L=repmat(size(t2,1),size(t2,1),1); X1.L0=X1.L;
    else
        X1.L=repmat(X1.L_max,size(t2,1),1); X1.L0=X1.L;
    end
else
    X1.f=X0;
end

X1.ts1=ts1;
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


h = waitbar(0,'Please wait...Validation samples'); steps=size(x1,2);
for i=1:size(x1,2)
    waitbar(i/steps)
    if Variable_window
        X1.f=[X1.f0(end-X1.L(end)+1:end,:); f(x1(i),:)];
        
        tstart = tic;
        [X1,PC,T2,Q]=VMWPCA_update(X1,PC,T2,Q,alpha,SD_type,nPC_select_algorithm,target_CPV,x1(i));
        telapsed = toc(tstart);
        Time_effors=[Time_effors; telapsed];
    else
        X1.f=[X1.f; f(x1(i),:)];
        
        tstart = tic;
        [X1,PC,T2,Q]=MWPCA_update(X1,PC,T2,Q,alpha,SD_type,nPC_select_algorithm,target_CPV,x1(i));        
        telapsed = toc(tstart);
        Time_effors=[Time_effors; telapsed];
        
    end
    
end
close(h)

h = waitbar(0,'Please wait...Testing samples'); steps=size(x1,2);
for i=1:size(x2,2)
    waitbar(i/steps)
    if Variable_window
        X1.f=[X1.f0(end-X1.L(end)+1:end,:); f(x2(i),:)];
        
        tstart = tic;
        [X1,PC,T2,Q]=VMWPCA_update(X1,PC,T2,Q,alpha,SD_type,nPC_select_algorithm,target_CPV,x2(i));
        telapsed = toc(tstart);
        Time_effors=[Time_effors; telapsed];
    else
        X1.f=[X1.f; f(x2(i),:)];
        
        tstart = tic;
        [X1,PC,T2,Q]=MWPCA_update(X1,PC,T2,Q,alpha,SD_type,nPC_select_algorithm,target_CPV,x2(i));
        telapsed = toc(tstart);
        Time_effors=[Time_effors; telapsed];
    end
    
end
close(h)

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
if Variable_window
    fn_fig=['VMWPCA n=' num2str(size(x0,2)) ' (#n0 = ' num2str(init_train_ratio) ')_' num2str(fn_type)];
else
    fn_fig=['MWPCA n=' num2str(size(x0,2)) ' (#n0 = ' num2str(init_train_ratio) ')_' num2str(fn_type)];
end

%% PLOT MONITORING STATISTICS
fig_T_Q=figure;
Plot_monitoring_statistics

%% SAVE RESULTS
savefig([pwd '\Result\' fn_fig '.fig']);