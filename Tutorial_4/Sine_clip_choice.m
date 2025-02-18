% Define parameters
fs = 44100;   % Sampling frequency
f = input("Enter the frequency of the wave in Hz: "); % Frequency of sine wave (Hz)
duration = 2; % Duration in seconds
t = 0:1/fs:duration-1/fs; % Time vector

% Ask user for input
clip_high = input('Enter the upper clipping limit: ');
clip_low = input('Enter the lower clipping limit: ');

% Ensure valid input (upper limit must be greater than lower limit)
while clip_high <= clip_low
    disp('Error: Upper limit must be greater than lower limit.');
    clip_high = input('Enter the upper clipping limit: ');
    clip_low = input('Enter the lower clipping limit: ');
end


% Generate sine wave
x = sin(2 * pi * f * t); % Original sine wave

% Apply hard clipping (limit values to ±0.5)
y = x; % Copy original signal
y(y > clip_high) = clip_high; % Clip upper limit
y(y < clip_low) = clip_low; % Clip lower limit

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

signalAnalyzer(y, t);
