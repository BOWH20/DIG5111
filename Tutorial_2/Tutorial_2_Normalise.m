% Define the filename
filename = 'Piano_middle_C.wav';

% Read the audio file in native format
[Sig_native, fs] = audioread(filename, 'native'); 

% Determine the maximum absolute value based on data type
max_val = double(intmax(class(Sig_native))); % Get max possible value for the data type

% Convert to double and normalize between -1 and 1
Sig_normalized = double(Sig_native) / max_val;

% Plot the original vs normalized signals
t = (0:length(Sig_native)-1) / fs; % Time vector

figure;
subplot(2,1,1);
plot(t, double(Sig_native)); % Convert to double for plotting
xlabel('Time (seconds)');
ylabel('Amplitude');
title('Original Native Audio Signal');
grid on;

subplot(2,1,2);
plot(t, Sig_normalized);
xlabel('Time (seconds)');
ylabel('Normalized Amplitude');
title('Normalized Audio Signal (Range: -1 to 1)');
grid on;

% Play normalized audio
sound(Sig_normalized, fs);
