% Read the original piano audio file
filename = 'Piano_middle_C.wav';
[Sig, fs] = audioread(filename);

% Reverse the audio signal
Sig_reversed = flipud(Sig); % Flip the signal upside down (reverse order)

% Play the reversed audio
sound(Sig_reversed, fs);

% Plot original and reversed signals for comparison
t = (0:length(Sig)-1) / fs; % Time vector

figure;
subplot(2,1,1);
plot(t, Sig);
xlabel('Time (seconds)');
ylabel('Amplitude');
title('Original Audio Signal');
grid on;

subplot(2,1,2);
plot(t, Sig_reversed);
xlabel('Time (seconds)');
ylabel('Amplitude');
title('Reversed Audio Signal');
grid on;
