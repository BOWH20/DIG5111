% Define parameters
fs = 44100;   % Sampling frequency
f = 440;      % Frequency of sine wave (Hz)
duration = 2; % Duration in seconds
t = 0:1/fs:duration-1/fs; % Time vector

% Generate sine wave
x = sin(2 * pi * f * t); % Original sine wave

% Apply hard clipping (limit values to ±0.5)
y = x; % Copy original signal
y(y > 0.5) = 0.5; % Clip upper limit
y(y < -0.5) = -0.5; % Clip lower limit

% Select a small portion of the signal for plotting (first 10ms)
N = round(0.01 * fs); % First 10ms of data

% Plot original sine wave
figure;
subplot(2,1,1);
plot(t(1:N), x(1:N));
title('Original Sine Wave (First 10ms)');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;
xlim([t(1) t(N)]); % Zoom into first 10ms
ylim([-1 1]); % Keep full range for reference

% Plot clipped sine wave with extended y-range for comparison
subplot(2,1,2);
plot(t(1:N), y(1:N), 'r');
title('Clipped Sine Wave (First 10ms)');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;
xlim([t(1) t(N)]); % Zoom into first 10ms
ylim([-1 1]); % Extend y-axis range to show effect of clipping

% Play sound
sound(y, fs);




