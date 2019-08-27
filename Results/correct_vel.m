clc, clear, close all
dsName = 'mycar';
subType = 'none';
seq = 1;

loadData

time = cumtrapz(dt);
acc_gnd = dt.*acc_gnd;
vel_imu = cumtrapz(time, acc_gnd);

% KF
velKF = [0 0 0];
A = eye(3);
H = eye(3);
P{1} = eye(3)*10^-10;
R = [1 0 0; 0 1 0; 0 0 1]*10^-5

%airsim mr
% R = [4.4016473e-04 -1.0391317e-04 -7.1615077e-06;
%  -1.0391317e-04  3.8396441e-04  2.5702146e-05;
%  -7.1615077e-06  2.5702146e-05  1.0527541e-04]

% airsim edge
% R = [4.4269957e-05 -4.4235094e-06 -6.6785510e-06;
%  -4.4235094e-06  1.8945106e-04  2.3870592e-05;
%  -6.6785510e-06  2.3870592e-05  1.1429927e-05]


% airsim mrseg
% R = [ 2.5137659e-05  1.9389690e-06 -7.5634234e-06;
%   1.9389690e-06  4.9889759e-06 -1.6071081e-06;
%  -7.5634234e-06 -1.6071081e-06  2.8633654e-06]

% airsim bar
% R = [3.98180091e-05 -5.07786399e-06 -7.15094484e-06;
%  -5.07786399e-06  1.21690406e-04  1.15208632e-05;
%  -7.15094484e-06  1.15208632e-05  3.84481609e-06]

% airsim pin
% R = [3.1410756e-05 -6.9592256e-06 -4.3311611e-06;
%  -6.9592256e-06  1.2303056e-04  1.4642666e-05;
%  -4.3311611e-06  1.4642666e-05  4.3329587e-06]

% kitti
% R = [3.7570933e-06  1.1800409e-07 -8.0405171e-06;
%   1.1800409e-07  9.9594472e-06 -4.1721264e-06;
%  -8.0405171e-06 -4.1721264e-06  1.5025614e-04]

% euroc
% R = [3.4694625e-05  3.5820525e-05 -5.3593169e-05;
%  3.5820525e-05  7.9021651e-05  5.0613262e-06;
%  -5.3593169e-05  5.0613262e-06  3.2767610e-04]
% R =[1.4624291e-05  9.2608570e-06 -1.1775744e-05;
%   9.2608570e-06  2.1252627e-05  2.3222358e-06;
%  -1.1775744e-05  2.3222358e-06  5.2838499e-05]

% % mycar
R = [1.1310258e-04 -3.2495232e-06  8.0977925e-06;
 -3.2495232e-06  4.3885651e-05  3.6321462e-06;
 8.0977925e-06  3.6321462e-06  1.0538273e-05];


acc_gnd = mylp(acc_gnd, 0.9);

for i=1:1:N
    velKF(i+1,:) = A*velKF(i,:)' + dt(i)*acc_gnd(i,:)';
    %R = acc_Q{i};
    pp = A*P{i}*A' + R;
    
    mCov = dtr_Q_gnd{i};
    K = pp*H'*inv(H*pp*H' + mCov);
    z = pr_dtr_gnd(i,:)';
    velKF(i+1,:) = (velKF(i+1,:)' + K*(z-H*velKF(i+1,:)'))';
    P{i+1} = pp - K*H*pp;
end

for i =1:1:N
   kfcov3(i,:) = diag(P{i});
end
kfstd3 = sqrt(kfcov3);

pos_intKF = cumtrapz(velKF(1:end-1,:));
pr_pos_int = cumtrapz(pr_dtr_gnd);

%% plot
%{
subPlotRow = 8;
subPlotCol = 3;
index = 1;

% w = 1000;
% h = 1200;
% fig = figure('Renderer', 'painters', 'Position', [600 100 w h]);
% 
% axes( 'Position', [0, 0.95, 1, 0.05] ) ;
% set( gca, 'Color', 'None', 'XColor', 'None', 'YColor', 'None' ) ;
% figTitle = [dsName, ' ', subType, ' ', int2str(seq)];
% text( 0.5, 0, figTitle, 'FontSize', 20', 'FontWeight', 'Bold','HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom' ) ;
% 
% text( 0.75, 0, '- Ground Truth', 'FontSize', 10', 'Color', 'red', 'FontWeight', 'Bold','HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom' ) ;
% text( 0.737, -0.3, '- Predicted', 'FontSize', 10', 'Color', 'blue', 'FontWeight', 'Bold','HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom' ) ;

% % plot du
% for i=1:1:3
%     subplot(subPlotRow, subPlotCol, index)
%     index = mysubplot(gt_du(:,i), pr_du(:,i), index);
%     if i==1 strYLabel = 'x';
%     elseif i==2 strYLabel = 'y';
%     else strYLabel = 'z';
%     end
%     ylabel([strcat('\fontsize{14} {\Delta}u_', strYLabel, ', m')])
%     xlabel(['\fontsize{14} Data Points'])
% end
% 
% % plot du_std
% for i=1:1:3
%     subplot(subPlotRow, subPlotCol, index)
%     plot(du_std3(:,i), 'b.', 'MarkerSize',5);
%     grid on
%     index = index + 1;
%     if i==1 strYLabel = 'x';
%     elseif i==2 strYLabel = 'y';
%     else strYLabel = 'z';
%     end
%     ylabel([strcat('\fontsize{14} 1{\sigma}_{{\Delta}u_', strYLabel, '}, m')])
%     xlabel(['\fontsize{14} Data Points'])
% end
% 
% % plot dw
% for i=1:1:3
%     subplot(subPlotRow, subPlotCol, index)
%     index = mysubplot(gt_dw(:,i), pr_dw(:,i), index);
%     if i==1 strYLabel = 'x';
%     elseif i==2 strYLabel = 'y';
%     else strYLabel = 'z';
%     end
%     ylabel([strcat('\fontsize{14} {\Delta}w_', strYLabel, ', rad')])
%     xlabel(['\fontsize{14} Data Points'])
% end
% 
% % plot dw_std
% for i=1:1:3
%     subplot(subPlotRow, subPlotCol, index)
%     plot(dw_std3(:,i), 'b.', 'MarkerSize',5);
%     grid on
%     index = index + 1;
%     if i==1 strYLabel = 'x';
%     elseif i==2 strYLabel = 'y';
%     else strYLabel = 'z';
%     end
%     ylabel([strcat('\fontsize{14} 1{\sigma}_{{\Delta}w_', strYLabel, '}, rad')])
%     xlabel(['\fontsize{14} Data Points'])
% end
% 
% % plot vel_gnd_imu
% for i=1:1:3
%     subplot(subPlotRow, subPlotCol, index)
%     hold on
%     plot(gt_dtr_gnd(:,i), 'r.', 'MarkerSize',5);
%     plot(vel_imu(:,i), 'g.', 'MarkerSize',2);
%     plot(pr_dtr_gnd(:,i), 'b.', 'MarkerSize',1);
%     grid on
%     title('Vel Gnd Imu in Green')
%     index = index + 1;
%     if i==1 strYLabel = 'x';
%     elseif i==2 strYLabel = 'y';
%     else strYLabel = 'z';
%     end
%     ylabel([strcat('\fontsize{14} Vel Gnd ', strYLabel, ', m/s')])
%     xlabel(['\fontsize{14} Data Points'])
% end
% 
% % plot vel_gnd_KF
% for i=1:1:3
%     subplot(subPlotRow, subPlotCol, index)
%     hold on
%     plot(gt_dtr_gnd(:,i), 'r.', 'MarkerSize',5);
%     plot(velKF(:,i), 'g.', 'MarkerSize',2);
%     grid on
%     index = index + 1;
%     if i==1 strYLabel = 'x';
%     elseif i==2 strYLabel = 'y';
%     else strYLabel = 'z';
%     end
%     ylabel([strcat('\fontsize{14} Vel Gnd_', strYLabel, ', m/s')])
%     xlabel(['\fontsize{14} Data Points'])
% end
% 
% % plot kf std
% for i=1:1:3
%     subplot(subPlotRow, subPlotCol, index)
%     plot(kfstd3(:,i), 'b.', 'MarkerSize',5);
%     grid on
%     index = index + 1;
%     if i==1 strYLabel = 'x';
%     elseif i==2 strYLabel = 'y';
%     else strYLabel = 'z';
%     end
%     ylabel([strcat('\fontsize{14} 1{\sigma}_{kf_', strYLabel, '}, rad')])
%     xlabel(['\fontsize{14} Data Points'])
% end
% 
% for i=1:1:3
%     subplot(subPlotRow, subPlotCol, index)
%     hold on
%     plot(gt_pos(:,i), 'r.', 'MarkerSize',5);
%     plot(pos_intKF(:,i), 'g.', 'MarkerSize',2);
%     plot(pr_pos_int(:,i), 'b.', 'MarkerSize',1);
%     grid on
%     index = index + 1;
%     if i==1 strYLabel = 'x';
%     elseif i==2 strYLabel = 'y';
%     else strYLabel = 'z';
%     end
%     ylabel([strcat('\fontsize{14} Vel Gnd_', strYLabel, ', m/s')])
%     xlabel(['\fontsize{14} Data Points'])
% end
%}

w = 500;
h = 600;
fig = figure('Renderer', 'painters', 'Position', [600 100 w h]);
subplot(3,1, 1)
hold on
plot(gt_dtr_gnd(:,1),'r.-', 'MarkerSize',5);
plot(pr_dtr_gnd(:,1), 'b.-', 'MarkerSize',2);
plot(vel_imu(:,1),'cyan.-', 'MarkerSize',2);
plot(velKF(:,1), 'g.-', 'MarkerSize',2);
ylabel('X axis, m', 'fontsize', 14)
legend('GT', 'CNN', 'ACC', 'KF')

subplot(3,1, 2)
hold on
plot(gt_dtr_gnd(:,2),'r.-', 'MarkerSize',5);
plot(pr_dtr_gnd(:,2), 'b.-', 'MarkerSize',2);
plot(vel_imu(:,2),'cyan.-', 'MarkerSize',2);
plot(velKF(:,2), 'g.-', 'MarkerSize',2);
ylabel('Y axis, m', 'fontsize', 14)
legend('GT', 'CNN', 'ACC', 'KF')

subplot(3,1, 3)
hold on
plot(gt_dtr_gnd(:,3),'r.-', 'MarkerSize',5);
plot(pr_dtr_gnd(:,3), 'b.-', 'MarkerSize',2);
plot(vel_imu(:,3),'cyan.-', 'MarkerSize',2);
plot(velKF(:,3), 'g.-', 'MarkerSize',2);
ylabel('Z axis, m', 'fontsize', 16)
legend('GT', 'CNN', 'ACC', 'KF')
xlabel('Data Points', 'fontsize', 16)

w = 500;
h = 600;
fig = figure('Renderer', 'painters', 'Position', [600 100 w h]);
subplot(3,1, 1)
hold on
plot(kfstd3(:,1),'b.', 'MarkerSize',1);
ylabel('X axis, m', 'fontsize', 16)

subplot(3,1, 2)
hold on
plot(kfstd3(:,2),'b.', 'MarkerSize',1);
ylabel('Y axis, m', 'fontsize', 16)

subplot(3,1, 3)
hold on
plot(kfstd3(:,3),'b.', 'MarkerSize',1);
ylabel('Z axis, m', 'fontsize', 16)
xlabel('Data Points', 'fontsize', 16)

figure
subplot(3,1, 1)
hold on
plot(gt_pos(:,1),'r.-', 'MarkerSize',5);
plot(pr_pos(:,1),'b.-', 'MarkerSize',1);
plot(pos_intKF(:,1),'g.-', 'MarkerSize',1);
ylabel('X axis, m', 'fontsize', 16)
legend('GT', 'CNN', 'KF');
subplot(3,1, 2)
hold on
plot(gt_pos(:,2),'r.-', 'MarkerSize',5);
plot(pr_pos(:,2),'b.-', 'MarkerSize',1);
plot(pos_intKF(:,2),'g.-', 'MarkerSize',1);
ylabel('Y axis, m', 'fontsize', 16)
legend('GT', 'CNN', 'KF');
subplot(3,1, 3)
hold on
plot(gt_pos(:,3),'r.-', 'MarkerSize',5);
plot(pr_pos(:,3),'b.-', 'MarkerSize',1);
plot(pos_intKF(:,3),'g.-', 'MarkerSize',1);
ylabel('Z axis, m', 'fontsize', 16)
legend('GT', 'CNN', 'KF');
xlabel('Data Points', 'fontsize', 16)

figure
hold on
plot(gt_pos(:,1), gt_pos(:,2), 'r')
plot(pr_pos(:,1), pr_pos(:,2), 'b')
plot(pos_intKF(:,1), pos_intKF(:,2), 'g')
xlabel('Position X, m', 'fontsize', 16)
ylabel('Position Y, m', 'fontsize', 16)
legend('GT', 'CNN', 'KF');


prPath = ['Data\',getPRPath(dsName, subType, seq)];
kfName = strcat(prPath, '_dtrKF.txt')
dlmwrite(kfName, velKF(2:end,:))



function[pltIndex] = mysubplot(gt, pr, index)
    hold on
    grid on
    plot(gt, 'r.', 'MarkerSize',10)
    plot(pr, 'b.', 'MarkerSize',1)
    hold off
    pltIndex = index + 1;
end

function[pltIndex] = mysubplot2D(gt1, gt2, pr1, pr2, index)
    hold on
    grid on
    plot(gt1,gt2, 'r.', 'MarkerSize',10)
    plot(pr1,pr2, 'b.', 'MarkerSize',1)
    hold off
    pltIndex = index + 1;
end


function[res] = mylp(data, a)
    res(1,:) = data(1,:);
    for i =2:1:length(data)
        res(i,:) = a*res(i-1,:) + (1-a)*data(i,:);
    end
end

















