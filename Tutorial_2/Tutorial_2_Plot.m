% Audio filename must be in the same directory as this m file
Filename = 'piano_middle_C.wav';

% Read the WAV file into memory
[Sig, Fs] = audioread(Filename);

% Sig stores raw audio data in a column; Fs is the sampling frequency
Duration = length(Sig) / Fs; % Total duration of the audio in seconds
disp(['Duration: ', num2str(Duration), ' seconds']);

% Create a time vector
Ts = 1 / Fs; % Sampling period
Time = (0:length(Sig)-1) * Ts; % Time vector (same length as Sig)

% Plot the audio signal
figure; % Open a new figure window
plot(Time, Sig); % Time on x-axis, Sig on y-axis
xlabel('Time (seconds)'); % Label for x-axis
ylabel('Amplitude'); % Label for y-axis
title('Audio Signal: piano\_middle\_C.wav');

% Add a grid for better readability
grid on;
