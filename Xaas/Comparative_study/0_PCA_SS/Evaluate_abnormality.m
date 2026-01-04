% Q_check=cumsum(...)
if sum(PC.Monitoring_statistic)==2 % NORMAL in Q-statistic
    if Q_check(end)
        if T2_check(end) % NORMAL in T2
            Normal_condition=1;
        end
    end
    
elseif PC.Monitoring_statistic(1)==1 % NORMAL in Q-statistic
    % Check whether current sample is in control or out of control
    if Q_check(end)==0
        % outlier or fault? Anyway dont update current PC
        Normal_condition=0;
        PC.outlier=[PC.outlier; ind_x-1];
        PC.fault=[PC.fault; PC.fault(end)+1];
    else
        Normal_condition=1;
        PC.fault=[PC.fault; 0];
    end
    
    if 0
        try
            figure(findobj('tag','history'));
            subplot(4,1,1:3);
            plot(X0.ts1.Time(ind_x),X0.f0(end,:),'.'),hold on
            subplot(4,1,4);
            if X0.Z24_TP(ind_x)<0
                plot(X0.ts1.Time(ind_x),X0.Z24_TP(ind_x),'r.'),hold on
            else
                plot(X0.ts1.Time(ind_x),X0.Z24_TP(ind_x),'b.'),hold on
            end
            
        catch
            figure('tag','history');
            subplot(4,1,1:3);
            plot(X0.ts1.Time);cla
            plot(X0.ts1.Time(1:ind_x),X0.f0(1:ind_x,:),'.'),hold on
            subplot(4,1,4);
            plot(X0.ts1.Time);cla
            plot(X0.ts1.Time(1:ind_x),X0.Z24_TP(1:ind_x),'.'),hold on
        end
        
        if isempty(findobj('tag','abnormal check'));
            figure('tag','abnormal check')
        else
            figure(findobj('tag','abnormal check'));
        end
        
        stem(Qdist(end-PC.n_stall+1:end)); hold on
        plot([0 size(Q_check,1)+1],Q_lim*ones(1,2),'r:');
        xlim([0 size(Q_check,1)+1])
        cla
    end
    
else % NORMAL in T2
    % Check whether current sample is in control or out of control
    if T2_check(end)==0
        % outlier or fault? Anyway dont update current PC
        Normal_condition=0;
        PC.outlier=[PC.outlier; ind_x];
        PC.fault=[PC.fault; PC.fault(end)+1];
    else
        Normal_condition=1;
        PC.fault=[PC.fault; 0];
    end
    
end

%% If fault is detected. stop PCA update
if PC.fault(end)>PC.n_stall
    PC.update=0;
end