clear all; close all; clc;
% %%%%%%%%%%%% 비고 %%%%%%%%%%%% 
% 계측시작일: 2022년 8월 24일부터
% data.time %시간
% data.CG_1 %이격거리
% data.CG_2 %단차(침하)
% data.CG_3 %이격거리
% data.CG_4 %단차(침하)
% data.TT_1 %상부경사
% data.TT_2 %하부경사
% data.TT_3 %상부경사
% data.TT_4 %하부경사
% data.TC_1_Avg %부재온도
% data.TC_2_Avg %상온
% data.waterlevel %조위
% CG resolution: 0.01
% TT resolution: 0.001
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

temp = readtable('data_11_months.xlsx','VariableNamingRule','preserve');
% 11/1 00시 데이터: 9780 (9779: 10/31 08:00 --> 체크 필요)
% 12/1 00시 데이터: 14099 
% 1/1 00시 데이터: 18563 
% 2/1 00시 데이터: 22883
% 5/1 00시 데이터: 35032
% 6/1 00시 데이터: 39496
ind1 = 9780; %11/1~ baseline
% ind2 = 18564; %1/1부터 outlier
ind2 = 35033 ;%5/1부터 outlier
ind3 = 39497 ;%6/1부터 outlier

data = temp(ind1:ind2-1,:);
data_out1 = temp(ind2:ind3-1,:);
data_out2 = temp(ind3:end,:);

[data.time(1,:), data_out1.time(1,:), data_out2.time(1,:)]

FeatureIDX = [3,4,5,6,7,8,9,10,11,12,15];

Data = table2array(data(:,FeatureIDX));
Cal = Data(1,1:8);
Data(:,1:8) = Data(:,1:8)-Cal;
t = data.time;

Data_out1 = table2array(data_out1(:,FeatureIDX));
Data_out1(:,1:8) = Data_out1(:,1:8)-Cal;
Data_out2 = table2array(data_out2(:,FeatureIDX));
Data_out2(:,1:8) = Data_out2(:,1:8)-Cal;
t_out1 = data_out1.time;
t_out2 = data_out2.time;
% % % clear data;

TiTle = {'Spacing (mm)','Defelction (mm)','Slope_{upper} (\circ)',...
    'Slope_{lower} (\circ)','Temperature (\circC)','Waterlevel (cm)'};
TiTle_ul = {'Spacing','Defelction','Slope_{upper}',...
    'Slope_{lower}','Temperature','Waterlevel'};

% Check PDF
%%%% Displacement
figure('Position',[50 50 800 800]);
hold on; box on; grid on; grid minor;
for i = [1,2,3,4]
    pd = fitdist(Data(:,i),'Normal');
    pdnum(i,:) = pd.ParameterValues;
    x_values{i} = linspace(-10,10,1000);
    y{i} = pdf(pd,x_values{i});
    plot(x_values{i},y{i},'LineWidth',2)
end
set(gca,'XLim',[-10,10],'FontSize',16)
ylabel('PDF')
legend('Spacing','Deflection','Spacing','Deflection')
%%%% Slope
figure('Position',[50 50 800 800]);
hold on; box on; grid on; grid minor;
for i = [5,6,7,8]
    pd = fitdist(Data(:,i),'Normal');
    pdnum(i,:) = pd.ParameterValues;
    x_values{i} = linspace(-0.5,0.5,1000);
    y{i} = pdf(pd,x_values{i});
    plot(x_values{i},y{i},'LineWidth',2)
end
set(gca,'XLim',[-0.5,0.5],'FontSize',16)
ylabel('PDF')
legend('Slope_{upper}','Slope_{lower}','Slope_{upper}','Slope_{lower}')

% Make Outliers
driftAmount(1) = -pdnum(1,2)*1.5;
driftAmount(3) = -pdnum(3,2)*1.5;

driftAmount(2) = pdnum(2,2)*1.5;
driftAmount(4) = -pdnum(4,2)*1.5;

driftAmount(5) = pdnum(5,2)*3;
driftAmount(7) = pdnum(7,2)*3;

driftAmount(6) = pdnum(6,2)*3;
driftAmount(8) = pdnum(8,2)*3;

% for i = 1:8
%     if pdnum(i,1) >= 0
%         driftAmount(i) = 2*pdnum(i,2); % mu + 2*sigma
%     else
%         driftAmount(i) = -2*pdnum(i,2);
%     end
% end


FeatureIDX = [3,4,5,6,7,8,9,10,11,12,15];

% Induce Outliers
Data_out1(:,[1,2,5,6]) = Data_out1(:,[1,2,5,6])+driftAmount([1,2,5,6]);
Data_out2(:,1:8) = Data_out2(:,1:8)+driftAmount(1:8);
Data_out2(:,[1,2,5,6]) = Data_out2(:,[1,2,5,6])+driftAmount([1,2,5,6]);
driftAmount

% % % % % test용
% % % % % Data_out1(:,[1,2]) = Data_out1(:,[1,2])+driftAmount([1,2]);
% % % % % Data_out1(:,[5,6]) = Data_out1(:,[5,6])+driftAmount([5,6]);
% % % % % 
% % % % % % % Data_out2(:,1:8) = Data_out2(:,1:8)+driftAmount(1:8);
% % % % % Data_out2(:,[1,2]) = Data_out2(:,[1,2])+driftAmount([1,2]);
% % % % % Data_out2(:,[5,6]) = Data_out2(:,[5,6])+driftAmount([5,6]);
% % % % % Data_out2(:,[3,4]) = Data_out2(:,[3,4])+driftAmount([3,4]);
% % % % % Data_out2(:,[7,8]) = Data_out2(:,[7,8])+driftAmount([7,8]);
% % % % % 
% % % % % Data_out2(:,[1,2]) = Data_out2(:,[1,2])+driftAmount([1,2]);
% % % % % Data_out2(:,[5,6]) = Data_out2(:,[5,6])+driftAmount([5,6]);
% % % % % driftAmount

% Check baseline and outlier
TiTle_ul = {'Spacing','Defelction','Spacing','Defelction',...
    'Slope_{upper}','Slope_{lower}','Slope_{upper}','Slope_{lower}',...
    'Temperature','Waterlevel'};

% Check baseline and outlier
YL = [[-2.5 12];[-2.5 1.5];[-0.2 0.2];[-0.2 0.2]]; % 11월 기준, 센서 4개 1set 기준
sn = [1,2,5,6,3,4,7,8];
figure('Position',[50 50 1300 1000]);
for i = 1:4
    subplot(2,2,i); hold on; box on; grid on; grid minor;
    scatter(t,Data(:,sn(i)),'Marker','.','MarkerEdgeColor',[0 0.7 0.7]);
    scatter(t,Data(:,sn(i+4)),'Marker','.','MarkerEdgeColor',[0 0.6 0.6]);    
    scatter(t_out1,Data_out1(:,sn(i)),'Marker','.','MarkerEdgeColor',[0 0.5 0.5]);
    scatter(t_out1,Data_out1(:,sn(i+4)),'Marker','.','MarkerEdgeColor',[0 0.4 0.4]);
    scatter(t_out2,Data_out2(:,sn(i)),'Marker','.','MarkerEdgeColor',[0 0.3 0.3]);
    scatter(t_out2,Data_out2(:,sn(i+4)),'Marker','.','MarkerEdgeColor',[0 0.2 0.2]);
%     set(gca,'FontSize',16,'XLim',[t(1) t_out(end)],'YLim',YL(i,:))
%     set(gca,'FontSize',16,'XLim',[t(1) t_out(end)])
    datetick('x','mmmyy','keepticks')
    ylabel(TiTle{i})
%     legend('Baseline-1','Baseline-2','Outlier-1','Outlier-2','Location','best')
    legend('Baseline-1','Baseline-2','','','Location','NorthWest')
end

% Label
rng(1090402);
l1 = rand(length(Data),1)./2;
l2 = (rand(length(Data_out1),1)-0.5)./2 +1;
l3 = (rand(length(Data_out2),1)-0.5)./2 +2;
l4 = (rand(length(Data_out1),1))./2;
l5 = (rand(length(Data_out2),1))./2 +1;
Label_1 = [l1; l2; l3]; %0 1 2
Label_2 = [l1; l4; l5]; %0 0 1
figure
plot(Label_1); hold on; plot(Label_2);
legend('1번 함체','2번 함체')
% Label = [zeros(length(Data),1); ones(length(Data_out),1)];

% Save Datasets   
Dataset = array2table([Data ; Data_out1 ; Data_out2],...
    'VariableNames',...
    {'CG_1','CG_2','CG_3','CG4','TT_1','TT_2','TT_3','TT_4','TC_1_Avg','TC_2_Avg','Waterlevel'});

Time = [data.time; data_out1.time; data_out2.time];
Dataset = addvars(Dataset,Time,'Before',"CG_1");
Dataset = addvars(Dataset,Label_1,'After',"Waterlevel");
Dataset = addvars(Dataset,Label_2,'After',"Label_1");

writetable(Dataset,'J_Dataset_1101_0630_outlier_3sig.csv')
