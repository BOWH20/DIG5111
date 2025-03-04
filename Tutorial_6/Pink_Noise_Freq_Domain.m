% Parameters
Fs = 22050; % Sampling rate (Hz)
T = 2;      % Duration of noise (seconds)
N = Fs * T; % Total number of samples

% Generate white noise
whiteNoise = randn(N, 1);

% Compute the FFT of the white noise
Y = fft(whiteNoise);

% Frequency axis
f = (0:N-1) * (Fs/N); % Frequency values corresponding to FFT bins

% Create a pink noise filter in the frequency domain
% Pink noise has a -3 dB/octave slope, which corresponds to 1/f in the frequency domain
% Avoid division by zero at f=0 by starting from the second element
pinkFilter = 1 ./ sqrt(f(2:N/2+1)); % Pink noise filter (1/f)

% Apply the filter to the FFT of the white noise
Y(2:N/2+1) = Y(2:N/2+1) .* pinkFilter(:); % Ensure pinkFilter is a column vector
Y(N/2+2:end) = conj(Y(N/2:-1:2)); % Ensure the output is real-valued

% Compute the inverse FFT to get the time-domain pink noise
pinkNoise = real(ifft(Y));

% Normalize the pink noise to the range [-1, 1]
pinkNoise = pinkNoise / max(abs(pinkNoise));

% Play the pink noise
sound(pinkNoise, Fs);

signalAnalyzer(pinkNoise);