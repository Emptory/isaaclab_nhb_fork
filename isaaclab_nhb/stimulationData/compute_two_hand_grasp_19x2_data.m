clc; clear; close all;

%% Two-hand 19x2 output using grasp matrix
% Output CSV columns:
% t,
% left  hand: position(3), quaternion(4), velocity(3), omega(3), force(3), moment(3)
% right hand: position(3), quaternion(4), velocity(3), omega(3), force(3), moment(3)
%
% That is 19 channels per hand, 38 data channels total. Time t is an
% additional alignment column.
%
% Convention:
% - q = [qw qx qy qz] maps body-frame vectors to inertial-frame vectors.
% - Hand positions, velocities, and forces are expressed in inertial frame N.
% - Hand angular velocity and contact moment are expressed in body frame B.
% - Each hand contact is a full 6D wrench, not a point contact.

inputCsv = 'box_6dof_straight_20s_dt0005.csv';
outputCsv = 'box_6dof_two_hand_19x2.csv';

L = 0.18;
W = 0.55;
H = 0.16;

% Contact locations in body frame, matched to the Isaac box anchors.
rLeft_B  = [0.09;  0.22; -0.08];
rRight_B = [0.09; -0.22; -0.08];

WL = [eye(3), zeros(3);
      skew(rLeft_B), eye(3)];
WR = [eye(3), zeros(3);
      skew(rRight_B), eye(3)];
Wgrasp = [WL, WR];             % 6 x 12
Wpinv = pinv(Wgrasp);
Nw = eye(12) - Wpinv * Wgrasp;

% Optional internal load. Keep zero for minimum-norm/no-extra-squeeze.
c = zeros(12, 1);

T = readtable(inputCsv);
N = height(T);
out = zeros(N, 39);            % time + 38 data channels
maxWrenchErr = 0;

for k = 1:N
    p_N = [T.x(k); T.y(k); T.z(k)];
    q = [T.qw(k); T.qx(k); T.qy(k); T.qz(k)];
    v_N = [T.vx(k); T.vy(k); T.vz(k)];
    omega_B = [T.wx(k); T.wy(k); T.wz(k)];
    R_NB = quatToRotmScalarFirst(q);

    F_N = [T.Fx(k); T.Fy(k); T.Fz(k)];
    tau_B = [T.taux(k); T.tauy(k); T.tauz(k)];
    F_B = R_NB' * F_N;
    P_B = [F_B; tau_B];

    lambda = Wpinv * P_B + Nw * c;
    maxWrenchErr = max(maxWrenchErr, max(abs(Wgrasp * lambda - P_B)));

    fLeft_B = lambda(1:3);
    mLeft_B = lambda(4:6);
    fRight_B = lambda(7:9);
    mRight_B = lambda(10:12);

    pLeft_N = p_N + R_NB * rLeft_B;
    pRight_N = p_N + R_NB * rRight_B;

    vLeft_N = v_N + R_NB * cross(omega_B, rLeft_B);
    vRight_N = v_N + R_NB * cross(omega_B, rRight_B);

    fLeft_N = R_NB * fLeft_B;
    fRight_N = R_NB * fRight_B;

    left19 = [pLeft_N; q; vLeft_N; omega_B; fLeft_N; mLeft_B];
    right19 = [pRight_N; q; vRight_N; omega_B; fRight_N; mRight_B];

    out(k,:) = [T.t(k), left19', right19'];
end

headers = {'t', ...
    'left_x','left_y','left_z', ...
    'left_qw','left_qx','left_qy','left_qz', ...
    'left_vx','left_vy','left_vz', ...
    'left_wx','left_wy','left_wz', ...
    'left_Fx','left_Fy','left_Fz', ...
    'left_Mx','left_My','left_Mz', ...
    'right_x','right_y','right_z', ...
    'right_qw','right_qx','right_qy','right_qz', ...
    'right_vx','right_vy','right_vz', ...
    'right_wx','right_wy','right_wz', ...
    'right_Fx','right_Fy','right_Fz', ...
    'right_Mx','right_My','right_Mz'};

writetable(array2table(out, 'VariableNames', headers), outputCsv);

fprintf('Saved %d rows to %s\n', N, outputCsv);
fprintf('Data channels: 19 x 2 = 38, plus time column t.\n');
fprintf('rank(W) = %d, nullity(W) = %d\n', rank(Wgrasp), size(Wgrasp,2) - rank(Wgrasp));
fprintf('max wrench reconstruction error = %.3e\n', maxWrenchErr);

%% Helpers
function S = skew(r)
    S = [ 0,    -r(3),  r(2);
          r(3),  0,    -r(1);
         -r(2),  r(1),  0   ];
end

function R = quatToRotmScalarFirst(q)
    q = q(:) / norm(q);
    qw = q(1); qx = q(2); qy = q(3); qz = q(4);

    R = [
        1 - 2*(qy^2 + qz^2), 2*(qx*qy - qw*qz),     2*(qx*qz + qw*qy);
        2*(qx*qy + qw*qz),     1 - 2*(qx^2 + qz^2), 2*(qy*qz - qw*qx);
        2*(qx*qz - qw*qy),     2*(qy*qz + qw*qx),     1 - 2*(qx^2 + qy^2)
    ];
end
