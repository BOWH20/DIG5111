% Parameters
N = 61;              % Number of filter taps
fc = 300;            % Cutoff frequency in Hz
Fs = 4000;           % Sampling frequency in Hz
n = -(N-1)/2:(N-1)/2; % Time index (centered around zero)

% Normalized cutoff frequency (relative to Nyquist frequency)
fcn = fc / (Fs/2);  % Normalize to Nyquist (Fs/2)

% Ideal sinc filter (Low-pass filter formula)
h = fcn * sinc(fcn * n);

% Optional: Apply window to reduce sidelobes (Hamming)
w = hamming(N)';
h = h .* w;

% Plot the truncated sinc function (Time domain)
figure;
stem(n, h, 'filled');
xlabel('Samples (n)');
ylabel('Amplitude');
title('Truncated Sinc Function (Impulse Response)');
grid on;

% Plot the frequency response (Spectrum)
figure;
fvtool(h, 'Fs', Fs);
title('Frequency Response (Magnitude and Phase)');

