% Parameters
fs = 10000; % Sampling frequency (10 kHz)
T = 1; % Duration of the signals (1 second)
t = 0:1/fs:T-1/fs; % Time vector for 1 second
N = 500; % Number of points to plot

% Generate the signals
f1 = 20; % Frequency of Signal 1 (20 Hz)
f2 = 200; % Frequency of Signal 2 (200 Hz)
f3 = 1000; % Frequency of Signal 3 (1000 Hz)
A = 0.5; % Amplitude of the signals

signal1 = A * sin(2 * pi * f1 * t); % Signal 1
signal2 = A * sin(2 * pi * f2 * t); % Signal 2
signal3 = A * sin(2 * pi * f3 * t); % Signal 3

% Plot the signals using subplot
figure;

% Signal 1: 20 Hz
subplot(3, 1, 1);
plot(t(1:N), signal1(1:N), 'b', 'LineWidth', 1.5);
title('Signal 1: 20 Hz');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

% Signal 2: 200 Hz
subplot(3, 1, 2);
plot(t(1:N), signal2(1:N), 'r', 'LineWidth', 1.5);
title('Signal 2: 200 Hz');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

% Signal 3: 1000 Hz
subplot(3, 1, 3);
plot(t(1:N), signal3(1:N), 'g', 'LineWidth', 1.5);
title('Signal 3: 1000 Hz');
xlabel('Time (s)');
ylabel('Amplitude');
grid on;

% Adjust layout for better visualization
sgtitle('Periodic Signals with Sampling Frequency 10 kHz');