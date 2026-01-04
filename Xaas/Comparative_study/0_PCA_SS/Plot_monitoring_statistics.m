set(gcf,'position',[174 483 670 374],'name',fn_fig);
plot(ts1.time_stamp(x0),Q.Qdist(x0),'bo','markersize',2,'markerfacecolor','b'); hold on; grid on
plot(ts1.time_stamp(x1),Q.Qdist(x1),'v','markerfacecolor',[0 0.498 0],'markeredgecolor',[0 0.498 0],'markersize',2);
plot(ts1.time_stamp(x2),Q.Qdist(x2),'kx','markersize',3);
plot(ts1.time_stamp(PC.outlier),Q.Qdist(PC.outlier),'rs','markersize',2)
switch alpha
    case '95%'
        a = plot(ts1.time_stamp,Q.distcrit(1:end-1,1),'k-.','markersize',2,'linewidth',1.5);
    case '99%'
        a = plot(ts1.time_stamp,Q.distcrit(1:end-1,2),'k-.','markersize',2,'linewidth',1.5);
end
ylabel('Q-statistics (SPE)','fontsize',15,'fontweight','bold');
set(gca,'fontsize',15,'fontweight','bold'); axis tight
YLIM=get(gca,'ylim');
patch([[ts1.time_stamp(PCA_par.d_indx), ts1.time_stamp(PCA_par.d_indx)] [ts1.time_stamp(end) ts1.time_stamp(end)]],[YLIM(2) YLIM(1) YLIM(1) YLIM(2)],...
    'y','EdgeColor','k','FaceAlpha',0.1);
title([fn_fig ', Alpha=' alpha],'fontsize',15,'fontweight','bold');
legend(a, ['False alarm:' num2str(n_out) '  //  Total sample:' num2str(n_tot) ' => ' num2str(n_out/n_tot*100) '%'], 'FontSize', 10)

savefig(fig_T_Q,[pwd '\Fig\' fn_fig '.fig']);