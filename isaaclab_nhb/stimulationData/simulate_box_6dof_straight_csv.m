clc; clear; close all;

%% 6-DOF rigid-body simulation for a rectangular body tracking a straight line
% Output CSV columns:
% t,
% x,y,z, qw,qx,qy,qz, vx,vy,vz, wx,wy,wz, Fx,Fy,Fz, taux,tauy,tauz
%
% The 19 data channels after time are:
% position(3) + quaternion(4) + linear velocity(3) + angular velocity(3)
% + force(3) + torque(3).
%
% Convention:
% - Inertial/world frame: N
% - Body frame: B
% - q = [qw qx qy qz]' maps body-frame vectors to inertial-frame vectors.
% - v and F are expressed in the inertial frame.
% - omega and tau are expressed in the body frame.

%% Body parameters
params.m = 0.05;      % kg, matched to Isaac Lab
params.L = 0.18;      % m, Isaac box X
params.W = 0.55;      % m, Isaac box Y
params.H = 0.16;      % m, Isaac box Z
params.g = 9.81;      % m/s^2
params.gvec = [0; 0; -params.g];

Jx = (1/12) * params.m * (params.W^2 + params.H^2);
Jy = (1/12) * params.m * (params.L^2 + params.H^2);
Jz = (1/12) * params.m * (params.L^2 + params.W^2);
params.J = diag([Jx, Jy, Jz]);

%% Reference trajectory: straight line at 0.30 m/s
% Keep this inside the frozen S1 command domain [0.25, 0.40] m/s and in
% sync with CoopG1S2CommandsCfg.
params.v_ref = 0.30;      % m/s
params.y_ref = 0.0;       % m
% Nominal world height of the midpoint between the two inner palm contacts:
% 0.82 m torso target height + 0.095234415 m contact height in torso frame.
params.z_ref = 0.915234415; % m
params.q_ref = [1; 0; 0; 0];
params.omega_ref = [0; 0; 0];

%% Controller gains
params.Kp_pos = diag([8, 8, 12]);
params.Kd_pos = diag([6, 6, 7]);

% Attitude gains are in body torque units. The quaternion vector part is
% multiplied by 2, so these values behave like approximate radian gains.
params.Kp_att = diag([0.40, 0.45, 0.50]);
params.Kd_att = diag([0.12, 0.14, 0.16]);

%% Simulation settings
dt = 0.005;
T = 20.0;
t = (0:dt:T)';
N = numel(t);

%% Initial state
p0 = [0.0; 0.0; params.z_ref];         % m
v0 = [params.v_ref; 0; 0];             % m/s
q0 = [1; 0; 0; 0];                     % identity attitude
w0 = [0; 0; 0];                        % rad/s, body frame

% State x = [p(3); v(3); q(4); omega(3)]
x = [p0; v0; q0; w0];

X = zeros(N, 13);
U = zeros(N, 6);   % [F_N(3); tau_B(3)]

%% RK4 integration
for k = 1:N
    tk = t(k);
    x(7:10) = normalizeQuat(x(7:10));

    [~, Fk, tauk] = closedLoopDynamics(tk, x, params);
    X(k,:) = x';
    U(k,:) = [Fk; tauk]';

    if k < N
        k1 = closedLoopDynamics(tk, x, params);

        x2 = x + 0.5 * dt * k1;
        x2(7:10) = normalizeQuat(x2(7:10));
        k2 = closedLoopDynamics(tk + 0.5 * dt, x2, params);

        x3 = x + 0.5 * dt * k2;
        x3(7:10) = normalizeQuat(x3(7:10));
        k3 = closedLoopDynamics(tk + 0.5 * dt, x3, params);

        x4 = x + dt * k3;
        x4(7:10) = normalizeQuat(x4(7:10));
        k4 = closedLoopDynamics(tk + dt, x4, params);

        x = x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4);
        x(7:10) = normalizeQuat(x(7:10));
    end
end

%% Save CSV
csvFile = 'box_6dof_straight_20s_dt0005.csv';

data = [t, X(:,1:3), X(:,7:10), X(:,4:6), X(:,11:13), U(:,1:3), U(:,4:6)];
headers = {'t', ...
    'x','y','z', ...
    'qw','qx','qy','qz', ...
    'vx','vy','vz', ...
    'wx','wy','wz', ...
    'Fx','Fy','Fz', ...
    'taux','tauy','tauz'};

tbl = array2table(data, 'VariableNames', headers);
writetable(tbl, csvFile);

fprintf('Saved %d rows to %s\n', N, csvFile);
fprintf('Inertia J = diag([%.8f %.8f %.8f]) kg*m^2\n', Jx, Jy, Jz);

%% Basic plots
figure;
plot3(params.v_ref*t, params.y_ref*ones(N,1), params.z_ref*ones(N,1), 'r--', 'LineWidth', 2); hold on;
plot3(X(:,1), X(:,2), X(:,3), 'b', 'LineWidth', 1.5);
grid on; axis equal;
xlabel('X / m'); ylabel('Y / m'); zlabel('Z / m');
legend('Reference', 'Rigid body center');
title('6-DOF Rectangular Body Straight-Line Tracking');

figure;
subplot(3,1,1);
plot(t, X(:,1), 'b', t, params.v_ref*t, 'r--', 'LineWidth', 1.2);
grid on; ylabel('x / m'); legend('actual', 'ref');
subplot(3,1,2);
plot(t, X(:,2), 'b', t, params.y_ref*ones(N,1), 'r--', 'LineWidth', 1.2);
grid on; ylabel('y / m');
subplot(3,1,3);
plot(t, X(:,3), 'b', t, params.z_ref*ones(N,1), 'r--', 'LineWidth', 1.2);
grid on; xlabel('t / s'); ylabel('z / m');

%% Dynamics and control
function [xdot, F, tau] = closedLoopDynamics(t, x, params)
    p = x(1:3);
    v = x(4:6);
    q = normalizeQuat(x(7:10));
    omega = x(11:13);

    [p_d, v_d, a_d, q_d, omega_d] = referenceSignal(t, params);

    e_p = p_d - p;
    e_v = v_d - v;

    a_cmd = a_d + params.Kp_pos * e_p + params.Kd_pos * e_v;

    % Translational dynamics: m * v_dot = F + m * gvec.
    % Therefore F below is the actuator/contact force needed in inertial frame.
    F = params.m * (a_cmd - params.gvec);

    q_err = quatMultiply(quatConj(q), q_d);
    if q_err(1) < 0
        q_err = -q_err;
    end

    e_q_vec = q_err(2:4);
    e_w = omega_d - omega;
    tau = params.Kp_att * (2 * e_q_vec) + params.Kd_att * e_w;

    p_dot = v;
    v_dot = F / params.m + params.gvec;
    q_dot = 0.5 * quatMultiply(q, [0; omega]);
    omega_dot = params.J \ (tau - cross(omega, params.J * omega));

    xdot = [p_dot; v_dot; q_dot; omega_dot];
end

function [p_d, v_d, a_d, q_d, omega_d] = referenceSignal(t, params)
    p_d = [params.v_ref * t; params.y_ref; params.z_ref];
    v_d = [params.v_ref; 0; 0];
    a_d = [0; 0; 0];
    q_d = params.q_ref;
    omega_d = params.omega_ref;
end

function q = normalizeQuat(q)
    q = q(:);
    q = q / norm(q);
    if q(1) < 0
        q = -q;
    end
end

function qc = quatConj(q)
    qc = [q(1); -q(2:4)];
end

function q = quatMultiply(q1, q2)
    w1 = q1(1); x1 = q1(2); y1 = q1(3); z1 = q1(4);
    w2 = q2(1); x2 = q2(2); y2 = q2(3); z2 = q2(4);

    q = [
        w1*w2 - x1*x2 - y1*y2 - z1*z2;
        w1*x2 + x1*w2 + y1*z2 - z1*y2;
        w1*y2 - x1*z2 + y1*w2 + z1*x2;
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ];
end

function q = eulZYXToQuat(yawPitchRoll)
    yaw = yawPitchRoll(1);
    pitch = yawPitchRoll(2);
    roll = yawPitchRoll(3);

    cy = cos(yaw / 2); sy = sin(yaw / 2);
    cp = cos(pitch / 2); sp = sin(pitch / 2);
    cr = cos(roll / 2); sr = sin(roll / 2);

    q = [
        cy*cp*cr + sy*sp*sr;
        cy*cp*sr - sy*sp*cr;
        sy*cp*sr + cy*sp*cr;
        sy*cp*cr - cy*sp*sr
    ];
    q = normalizeQuat(q);
end
