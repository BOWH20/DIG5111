function [b, a] = AudioEQLPF(fc, Fs, Q)
    % Bristow-Johnson low-pass filter design
    % Inputs:
    %   fc  - cutoff frequency (Hz)
    %   Fs  - sampling frequency (Hz)
    %   Q   - quality factor (dimensionless)

    % Normalize cutoff frequency (0 to 1 range for digital filter)
    omega = 2 * pi * fc / Fs;
    alpha = sin(omega) / (2 * Q);

    cos_omega = cos(omega);

    % Coefficients for Low-Pass Filter (from Bristow-Johnson)
    b0 = (1 - cos_omega)/2;
    b1 = 1 - cos_omega;
    b2 = (1 - cos_omega)/2;
    a0 = 1 + alpha;
    a1 = -2 * cos_omega;
    a2 = 1 - alpha;

    % Normalize coefficients by a0
    b = [b0 b1 b2] / a0;
    a = [1 a1/a0 a2/a0];
end


fc = 500;    % Cutoff at 500 Hz
Fs = 4000;   % Sampling frequency
Q = 0.707;   % Butterworth (standard smooth slope)

[b, a] = AudioEQLPF(fc, Fs, Q);

% Apply filter to a signal (e.g., x)
% y = filter(b, a, x);

% Visualize frequency response
fvtool(b, a, 'Fs', Fs);
title('Low-Pass Filter (Bristow-Johnson)');
