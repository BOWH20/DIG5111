% Parameters
Fs = 22050; % Sampling rate (Hz)
T = 2;      % Duration of noise (seconds)
N = Fs * T; % Total number of samples

% Generate white noise
whiteNoise = randn(N, 1);

% Design a pink noise filter (approximates -3 dB/octave slope)
B = [0.049922035, -0.095993537, 0.050612699, -0.004408786];
A = [1, -2.494956002, 2.017265875, -0.522189400];

% Apply the filter to white noise
pinkNoise = filter(B, A, whiteNoise);


% Play the pink noise
sound(pinkNoise, Fs);

signalAnalyzer(pinkNoise);