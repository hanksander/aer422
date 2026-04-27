clear; clc; close all;


%TASK!
A4 = [-0.021, 0.122,  0,    -9.81;
        -0.200, -0.512,  65.1,  0;
         0,      -0.006, -0.402, 0;
         0,      0,      1,     0];

B4 = [0.292; -1.96; -0.4; 0];    % Effect of delta_E on each state

% Augmented actuator model
A5 = [A4 B4;
      0 0 0 0 -10];

B5 = [0; 0; 0; 0; 10];   % input is delta_a

Ctheta = [0 0 0 1 0];    % output theta
Cq     = [0 0 1 0 0];    % output q
D = 0;

plant5 = ss(A5, B5, Ctheta, D);

Kq = -2.9;   % start here and tune

A_sas = A5;
A_sas(5,3) = A_sas(5,3) - 10*Kq;

B_sas = B5;  % input is now delta_c

sys_sas_theta = ss(A_sas, B_sas, Ctheta, 0);

damp(sys_sas_theta);
pole(sys_sas_theta);


%%%%%%%%%%%%%%%%%%%

s = tf('s');

a = 0.00000009;   
a=0.0001;% PI zero location, tune this
Ktheta = -4.175;   % gain, tune this

PI = Ktheta*(s + a)/s;



L = PI * sys_sas_theta;

figure;
margin(L);
grid on;

[GM, PM, Wcg, Wcp] = margin(L);

GM_dB = 20*log10(GM)
PM
Wcp

%{ 
GM_dB ≈ 20
PM > 60
Wcp ≈ 1 rad/s

1. Adjust K → get crossover ≈ 1 rad/s
2. Adjust a → fix phase margin (>60°)
3. Repeat until both satisfied
%}



T_theta = feedback(L, 1);

figure;
step(T_theta);
grid on;
title('Closed-loop pitch attitude response');


pole(T_theta)
damp(T_theta)

% After choosing Kq, Ktheta, and a

A6 = zeros(6,6);
A6(1:5,1:5) = A5;

% SAS + PI closure in actuator row:
% delta_a = -Kq*q + Ktheta*(theta_c - theta) + Ktheta*a*z1
A6(5,3) = A6(5,3) - 10*Kq;
A6(5,4) = A6(5,4) - 10*Ktheta;
A6(5,6) = A6(5,6) + 10*Ktheta*a;

% z1_dot = theta_c - theta
A6(6,4) = -1;

B6 = zeros(6,1);
B6(5) = 10*Ktheta;  % theta_c enters actuator through proportional PI part
B6(6) = 1;          % theta_c enters integrator

C6 = [0 0 0 1 0 0]; % output theta
D6 = 0;

sys_task1 = ss(A6, B6, C6, D6);

figure;
step(sys_task1);
grid on;
title('Task 1 pitch attitude closed-loop response');

pole(sys_task1)
damp(sys_task1)
