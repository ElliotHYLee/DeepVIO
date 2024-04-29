clc, clear, close all
for seq=0:1:10
    %seq = 10
clearvars  -except seq
close all


lie = Lie();
se3 = LieSE3();
so3 = LieSO3();

if seq>9
    fName = strcat('poses/', int2str(seq), '.txt');
    time = importdata(strcat('sequences/', int2str(seq) , '/times.txt'));
    folder = strcat('sequences/', int2str(seq) , '/');
else
    fName = strcat('poses/0', int2str(seq), '.txt');
    time = importdata(strcat('sequences/0', int2str(seq) , '/times.txt'));
    folder = strcat('sequences/0', int2str(seq) , '/');
end

data = importdata(fName);

N = length(data);
for i =1:1:N
    row = data(i,:);
    temp = (reshape(row, [4,3]))';
    T{i} = [temp;0 0 0 1];

    % change euler axis to my aae
    R_bdy2gnd{i} = T{i}(1:3,1:3);
    euler(i,:) = rotm2eul(R_bdy2gnd{i});
    noise = 1;
    euler(i,:) = euler(i,:).*(1 + rand(1,3)*noise/100);
    RRR = eul2rotm(euler(i,:), 'zyx');
    
    linR(i,:) = reshape(RRR', 1,9);

    % change pos axis to mine
    pos_gnd(i,:) = T{i}(1:3,4);
    
    % save linearlized T
    T_lin(i,:) = reshape(T{i}', [1,16]); 
end

time(1) = 0;
recon_T{1} = eye(4);

for i =1:1:length(data) - 1
    % get time infor for cumtrapz comparison
    dt(i,1) = (time(i+1) - time(i));

    % decompose T for w,u
    Tn = T{i+1};
    Tc = T{i};
    dT = Tc^-1*Tn;
    dtr(i,:) = dT(1:3,4);
    dtr_gnd(i,:) = R_bdy2gnd{i}*dtr(i,:)';
    [w,u] = se3.getLog(dT);
    dw(i,:) = w';
    du(i,:) = u';

    % recompose T using w,u
    recon_pos(i,:) = recon_T{i}(1:3,4);
    dT = se3.getExp(dw(i,:)', du(i,:)');
    recon_T{i+1,:} = recon_T{i}*dT;

end


% make fake acc signal
acc_gnd = zeros(N,3);
for i =1:1:length(du)-1
    acc_gnd(i+1,:) = (dtr_gnd(i+1,:) - dtr_gnd(i,:))/dt(i);
    acc_gnd(i+1,:) = acc_gnd(i+1,:) + randn(1,3)*0.05;
end


vel_gnd = cumtrapz(time(1:end), acc_gnd);
vel_gnd = vel_gnd + dtr_gnd(1,:);
pos_gnd_int = cumtrapz(dtr_gnd);


acc_gnd(1:end-1,:) = acc_gnd(1:end-1,:)./dt;

dw_gyro = dw + randn(size(dw))*0.001;


dlmwrite(strcat(folder, 'dt.txt'), dt)
dlmwrite(strcat(folder, 'dtrans.txt'), dtr)
dlmwrite(strcat(folder, 'du.txt'), du)
dlmwrite(strcat(folder, 'dw.txt'), dw)
dlmwrite(strcat(folder, 'dw_gyro.txt'), dw_gyro)
dlmwrite(strcat(folder, 'T_lin.txt'), T_lin)

dlmwrite(strcat(folder, 'dtrans_gnd.txt'), dtr_gnd)
dlmwrite(strcat(folder, 'pos.txt'), pos_gnd(1:end-1,:))
dlmwrite(strcat(folder, 'acc_gnd.txt'), acc_gnd(1:end-1,:))
% dlmwrite(strcat(folder, 'gt_acc_gnd.txt'), gt_acc_gnd(1:end-1,:))
linRName = strcat('linR', int2str(noise), '.txt')
dlmwrite(strcat(folder, linRName), linR(1:end-1,:))

R_bdy2gnd{100}
end