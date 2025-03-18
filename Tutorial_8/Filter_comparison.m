% Parameters
Fs = 4000;                 % Sampling frequency
t = 0:1/Fs:1;            % Time vector (0.1 seconds duration)

% Create input signal: sum of 150 Hz and 800 Hz sinusoids
x = sin(2*pi*150*t) + sin(2*pi*800*t);

% Plot original signal
figure;
plot(t, x);
xlabel('Time (s)');
ylabel('Amplitude');
title('Original Signal: 150 Hz + 800 Hz');
xlim([0,0.1]);
grid on;


%% Design Low-pass Sinc Filter (Same as before)
N = 61;
fc = 300;
n = -(N-1)/2:(N-1)/2;
fcn = fc / (Fs/2);
h = fcn * sinc(fcn * n);
w = hamming(N)';
h = h .* w;

% Filter the signal
y = conv(x, h, 'same');  % Use 'same' to keep output length same as input

% Plot filtered signal
figure;
plot(t, y);
xlabel('Time (s)');
ylabel('Amplitude');
title('Filtered Signal (Low-pass, 300 Hz cutoff)');
xlim([0, 0.1]);
grid on;

%% Compare in Frequency Domain
% FFT of original signal
X = abs(fft(x));
% FFT of filtered signal
Y = abs(fft(y));
% Frequency axis
f = (0:length(X)-1)*(Fs/length(X));

% Plot frequency spectra
figure;
subplot(2,1,1);
plot(f, X);
title('Spectrum of Original Signal');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
xlim([0 1000]);
grid on;

subplot(2,1,2);
plot(f, Y);
title('Spectrum of Filtered Signal');
xlabel('Frequency (Hz)');
ylabel('Magnitude');
xlim([0 1000]);
grid on;

sound(x);
pause(1);
sound(y);