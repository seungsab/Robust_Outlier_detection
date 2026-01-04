clear all, close all, clc;

if ~isdir([pwd '\Fig'])
    mkdir([pwd '\Fig'])
end

if ~isdir([pwd '\Result'])
    mkdir([pwd '\Result'])
end

addpath([pwd '\0_PCA_SS'])

%% PRELIMINARY SETTING FOR VARIOUS PCA
init_train_ratio=0.3;
% init_train_ratio=0.51;
% init_train_ratio=0.7;

%% Damage Senario
% CSV file format
% CG1: Displacement (spacing between Caisson #2 and #3)
% CG2: Displacement (spacing between Caisson #3 and #4)
% TT1: Slope of cap concrete of Caisson #2
% TT2: Slope of cap concrete of Caisson #4
% TC_1_Avg: Temperature of concrete member

fn = [pwd '\Data\CG_1&CG_2&TT_1_sig0_3_12.csv'];  fn_type = 1;
% fn = [pwd '\Data\CG_1&CG_2_sig0_3_12.csv'];  fn_type = 2;
% fn = [pwd '\Data\CG_1_sig1_12.csv'];  fn_type = 3;
% fn = [pwd '\Data\TT_1_sig1_12.csv'];  fn_type = 4;

data = importdata(fn);
time_stamp = data.textdata(2:10:end,1);
time_stamp = datetime(time_stamp);
data.data = data.data(1:10:end, :);

f = data.data(:, 1:4);
T_avg = data.data(:, 5);

%% PCA options
measurement_type = 'static';
PCA_par=[];
PCA_par.SD_type='Z-score';
PCA_par.alpha='99%';
PCA_par.nPC_select_algorithm='eigengap'; PCA_par.target_CPV=0;
PCA_par.Monitoring_statistic=[1 0]; % only Q-statistics

% For static data
if strcmp(measurement_type, 'static')
    PCA_par.n_stall = 120; 
    L_min=3600;
    L_max=14400;
else
    % For dynamic data (e.g., freq)
    PCA_par.n_stall = 10;
    L_min=500;
    L_max=2000;
end

%% DEFINE THE METHODOLOGIES
idx = time_stamp > datetime(2023,10,1);
idx = find(idx == 1);

indx2 = idx(1);
indx1 = int32(indx2*init_train_ratio);

PCA_par.x(1,:)=[1 indx1]; % INITIAL TRAINIG DATA
PCA_par.x(2,:)=[indx1+1 indx2]; % VALIDATION DATA (NO DAMAGE)
PCA_par.x(3,:)=[indx2+1 size(f,1)]; % TESTING DATA (DAMAGED)
PCA_par.d_indx=indx2;

disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
disp(['Filename: ' fn])
disp(['Standardization: ' PCA_par.SD_type])
disp(['Significance level for control limit: ' PCA_par.alpha])
disp(['nPC_select_algorithm: ' PCA_par.nPC_select_algorithm])

disp('Monitoring statistics: Q-statistics');
disp(['# of consecutive sample for outlier detection: ' num2str(PCA_par.n_stall)])
disp('Range :')
disp('Training data: ')
disp([datestr(time_stamp(1))  ' ~ ' datestr(time_stamp(indx1-1)) '  # of samples:' num2str(length(time_stamp(1:indx1)))])
disp('Validation data (no damaged): ')
disp([datestr(time_stamp(indx1)) ' ~ ' datestr(time_stamp(indx2-1)) '  # of samples:' num2str(length(time_stamp(indx1+1:indx2)))])
disp('Testing data (damaged): ')
disp([datestr(time_stamp(indx2)) ' ~ ' datestr(time_stamp(size(f,1))) '  # of samples:' num2str(length(time_stamp(indx2+1:size(f,1))))])

%% PLOT RAW DATA
zerIdx=[]; % Zero-crossing
for i=1:length(T_avg)-1
   if ((T_avg(i)>0 && T_avg(i+1)<0) || (T_avg(i)<0 && T_avg(i+1)>0))
      zerIdx(end+1)=i; % save index of zero-crossing
   end
end
ind=reshape(zerIdx,2,size(zerIdx,2)/2)';

x0=PCA_par.x(1,1):PCA_par.x(1,2);
x1=PCA_par.x(2,1):PCA_par.x(2,2);
x2=PCA_par.x(3,1):PCA_par.x(3,2);

ts1 = timetable(time_stamp, data.data);

figure; set(gcf,'position',[272 308 843 540]);
subplot(4,1,1:3)
plot(ts1.time_stamp(x0),f(x0,:),'bo','markersize',2,'markerfacecolor','b'); hold on; grid on
plot(ts1.time_stamp(x1),f(x1,:),'v','markerfacecolor',[0 0.498 0],'markeredgecolor',[0 0.498 0],'markersize',2);
plot(ts1.time_stamp(x2),f(x2,:),'kx','markersize',3);
YLIM=get(gca,'ylim');
ylabel('Disp. & Tilt','fontsize',15,'fontweight','bold'); grid on
set(gca,'fontsize',15,'fontweight','bold');
for i=1:size(ind,1)
    patch([ts1.time_stamp(ind(i,1)),ts1.time_stamp(ind(i,1)),ts1.time_stamp(ind(i,2)),ts1.time_stamp(ind(i,2))],...
        [YLIM(2) YLIM(1) YLIM(1) YLIM(2)],'y','EdgeColor','none','FaceAlpha',0.4);
end
patch([[ts1.time_stamp(PCA_par.d_indx), ts1.time_stamp(PCA_par.d_indx)] [ts1.time_stamp(end) ts1.time_stamp(end)]],[YLIM(2) YLIM(1) YLIM(1) YLIM(2)],...
    'r','EdgeColor','k','FaceAlpha',0.1);

axis tight

subplot(4,1,4)
plot(ts1.time_stamp(x0),T_avg(x0,:),'b.','markersize',2,'markerfacecolor','b'); hold on; grid on
plot(ts1.time_stamp(x1),T_avg(x1,:),'.','markerfacecolor',[0 0.498 0],'markeredgecolor',[0 0.498 0],'markersize',2);
plot(ts1.time_stamp(x2),T_avg(x2,:),'k.','markersize',3);
plot([ts1.time_stamp(1) ts1.time_stamp(size(T_avg,1))],[0 0],'r:')
ylabel('T (^oC)','fontsize',15,'fontweight','bold'); grid on
set(gca,'fontsize',15,'fontweight','bold'); grid on
YLIM=get(gca,'ylim');
for i=1:size(ind,1)
    patch([ts1.time_stamp(ind(i,1)),ts1.time_stamp(ind(i,1)),ts1.time_stamp(ind(i,2)),ts1.time_stamp(ind(i,2))],...
        [45 -10 -10 45],'y','EdgeColor','none','FaceAlpha',0.4);
end
axis tight
PCA_par.T_avg=T_avg; PCA_par.T0_below=ind;

run_PCA=[1 1 1 1];
%% STATIC PCA
if run_PCA(1)
    disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
    fprintf('Static PCA...')
    a_Static_PCA

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
end

%% INCREMENTAL STATIC PCA
if run_PCA(2)
    disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
    fprintf('Incremental Static PCA...')
    b_Incremental_static_PCA

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
end

%% MOVING WINDOW PCA
if run_PCA(3)
    Variable_window=0;
    disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
    fprintf('Moving Window PCA...')
    c_MWPCA

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
end

%% VARIABLE MOVING WINDOW PCA
if run_PCA(4)
    Variable_window=1;
    disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
    fprintf('Variable Moving Window PCA...')
    c_MWPCA

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
end