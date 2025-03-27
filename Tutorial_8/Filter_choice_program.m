function [b, a] = AudioEQFilter(fc, dBgain, Fs, Q, type)
    % Convert inputs to radians
    omega = 2 * pi * fc / Fs;
    sin_omega = sin(omega);
    cos_omega = cos(omega);
    
    % Compute gain (linear scale for Peaking/Shelving filters)
    A = 10^(dBgain/40);  % amplitude
    
    % alpha factor
    alpha = sin_omega / (2 * Q);
    
    % For shelf filters
    beta = sqrt(A) / Q;

    % Initialize coefficients
    b = [0 0 0];
    a = [0 0 0];

    % Select filter type
    switch lower(type)
        case 'lpf'
            b(1) = (1 - cos_omega)/2;
            b(2) = 1 - cos_omega;
            b(3) = (1 - cos_omega)/2;
            a(1) = 1 + alpha;
            a(2) = -2 * cos_omega;
            a(3) = 1 - alpha;
            
        case 'hpf'
            b(1) = (1 + cos_omega)/2;
            b(2) = -(1 + cos_omega);
            b(3) = (1 + cos_omega)/2;
            a(1) = 1 + alpha;
            a(2) = -2 * cos_omega;
            a(3) = 1 - alpha;
            
        case 'bpf'
            b(1) = sin_omega/2;
            b(2) = 0;
            b(3) = -sin_omega/2;
            a(1) = 1 + alpha;
            a(2) = -2 * cos_omega;
            a(3) = 1 - alpha;
            
        case 'bpfq'
            alpha = sin_omega * sinh( log(2)/2 * Q * omega/sin_omega ); % Constant Q
            b(1) = sin_omega/2;
            b(2) = 0;
            b(3) = -sin_omega/2;
            a(1) = 1 + alpha;
            a(2) = -2 * cos_omega;
            a(3) = 1 - alpha;
            
        case 'notch'
            b(1) = 1;
            b(2) = -2 * cos_omega;
            b(3) = 1;
            a(1) = 1 + alpha;
            a(2) = -2 * cos_omega;
            a(3) = 1 - alpha;
            
        case 'apf'
            b(1) = 1 - alpha;
            b(2) = -2 * cos_omega;
            b(3) = 1 + alpha;
            a(1) = 1 + alpha;
            a(2) = -2 * cos_omega;
            a(3) = 1 - alpha;
            
        case 'peakingeq'
            b(1) = 1 + alpha*A;
            b(2) = -2 * cos_omega;
            b(3) = 1 - alpha*A;
            a(1) = 1 + alpha/A;
            a(2) = -2 * cos_omega;
            a(3) = 1 - alpha/A;
            
        case 'lowshelf'
            sqrtA = sqrt(A);
            b(1) =    A*( (A+1) - (A-1)*cos_omega + 2*sqrtA*alpha );
            b(2) =  2*A*( (A-1) - (A+1)*cos_omega );
            b(3) =    A*( (A+1) - (A-1)*cos_omega - 2*sqrtA*alpha );
            a(1) =        (A+1) + (A-1)*cos_omega + 2*sqrtA*alpha;
            a(2) =   -2*( (A-1) + (A+1)*cos_omega );
            a(3) =        (A+1) + (A-1)*cos_omega - 2*sqrtA*alpha;
            
        case 'highshelf'
            sqrtA = sqrt(A);
            b(1) =    A*( (A+1) + (A-1)*cos_omega + 2*sqrtA*alpha );
            b(2) = -2*A*( (A-1) + (A+1)*cos_omega );
            b(3) =    A*( (A+1) + (A-1)*cos_omega - 2*sqrtA*alpha );
            a(1) =        (A+1) - (A-1)*cos_omega + 2*sqrtA*alpha;
            a(2) =    2*( (A-1) - (A+1)*cos_omega );
            a(3) =        (A+1) - (A-1)*cos_omega - 2*sqrtA*alpha;
            
        otherwise
            error('Unknown filter type');
    end
    
    % Normalize by a0
    b = b / a(1);
    a = a / a(1);
end


fc = 1000;         % Center / cutoff frequency
dBgain = 6;       % Gain in dB (used for peaking/shelf filters)
Fs = 4000;        % Sampling rate
Q = 0.909;        % Quality factor



[b, a] = AudioEQFilter(fc, dBgain, Fs, Q, 'notch');

% Visualize:
close all;
fvtool(b, a, 'Fs', Fs);

