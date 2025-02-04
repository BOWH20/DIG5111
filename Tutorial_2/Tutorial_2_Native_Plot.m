% Define the filename
filename = 'Piano_middle_C.wav';

% Read the WAV file using the 'native' argument
[Sig_native, Fs] = audioread(filename, 'native');

% Get the total duration of the audio
Duration = length(Sig_native) / Fs;
disp(['Duration: ', num2str(Duration), ' seconds']);

% Convert to time vector
Ts = 1 / Fs; % Sampling period
Time = (0:length(Sig_native)-1) * Ts; % Time vector

% Plot the audio signal
figure;
plot(Time, Sig_native);
xlabel('Time (seconds)');
ylabel('Amplitude');
title('Audio Signal in Time Domain (Native Format)');
grid on;
