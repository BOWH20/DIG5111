% Parameters
N = 61;              % Number of filter coefficients
fc = 500;            % Cutoff frequency (Hz)
Fs = 2000;           % Sampling frequency (Hz)
n = -(N-1)/2:(N-1)/2; % Symmetric index for filter

% Normalized cutoff frequency (relative to Nyquist)
fcn = fc / (Fs/2);

% Ideal sinc filter (low-pass)
h = fcn * sinc(fcn * n);

% Apply a window (Hamming window used here to reduce side lobes)
w = hamming(N)';
h = h .* w;

% Plot frequency response
fvtool(h, 'Fs', Fs)
title('Frequency Response of Truncated Sinc Filter (500 Hz cutoff)');


