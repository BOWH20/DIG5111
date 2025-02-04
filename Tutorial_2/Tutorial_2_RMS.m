% Read the piano audio signal
filename = 'Piano_middle_C.wav';
[Sig, fs] = audioread(filename);  % Read audio file (normalized by default)

% Compute the RMS of the signal
signal_rms = rms(Sig);

% Normalize signal to 0.5 times the RMS value
Sig_normalized = (Sig / signal_rms) * 0.5;

% Play the normalized signal
sound(Sig_normalized, fs);

% Plot original vs normalized signals
t = (0:length(Sig)-1) / fs; % Time vector

figure;
subplot(2,1,1);
plot(t, Sig);
xlabel('Time (seconds)');
ylabel('Amplitude');
title('Original Signal');
grid on;

subplot(2,1,2);
plot(t, Sig_normalized);
xlabel('Time (seconds)');
ylabel('Amplitude');
title('Normalized Signal (0.5 x RMS)');
grid on;
