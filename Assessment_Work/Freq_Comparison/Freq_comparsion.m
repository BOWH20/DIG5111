
% Key MATLAB functions used:
%   - uigetfile: Opens a file selection dialog.
%   - audioread: Reads audio files into MATLAB.
%   - fft: Computes the Fast Fourier Transform (FFT) to convert signals from
%          the time domain to the frequency domain.
%   - semilogx: Plots data with a logarithmic scale on the x-axis.
%   - inputdlg: Creates an input dialog box to take user parameters.
%   - spectrogram: Computes and displays the time-frequency representation of a signal.

%% Select and Load the First Audio File
% Ask the user to choose the first audio file (wav, mp3, flac).
[file1, path1] = uigetfile({'*.wav;*.mp3;*.flac', 'Audio Files (*.wav, *.mp3, *.flac)'}, ...
    'Select the First Audio File');
if isequal(file1, 0)
    % If you cancel, a message is shown and the script quits.
    disp('No file selected for the first audio. Exiting script.');
    return;
else
    % Build the full path and load the audio.
    fullFilePath1 = fullfile(path1, file1);
    [y1, fs1] = audioread(fullFilePath1);
    fprintf('Loaded %s with a sample rate of %d Hz.\n', file1, fs1);
end

% If the audio file is stereo, convert it to mono by averaging the channels.
if size(y1, 2) > 1
    y1 = mean(y1, 2);
end

%% Select and Load the Second Audio File
% Do the same for the second audio file.
[file2, path2] = uigetfile({'*.wav;*.mp3;*.flac', 'Audio Files (*.wav, *.mp3, *.flac)'}, ...
    'Select the Second Audio File');
if isequal(file2, 0)
    disp('No file selected for the second audio. Exiting script.');
    return;
else
    fullFilePath2 = fullfile(path2, file2);
    [y2, fs2] = audioread(fullFilePath2);
    fprintf('Loaded %s with a sample rate of %d Hz.\n', file2, fs2);
end

% Convert the second file to mono if it's stereo.
if size(y2, 2) > 1
    y2 = mean(y2, 2);
end

%% Compute FFT and Magnitude Spectrum for the First Audio File
% Get the number of samples.
N1 = length(y1);
% Calculate the FFT which turns the time signal into frequency components.
Y1 = fft(y1);
% Only need half of the FFT output since it's symmetric.
halfN1 = floor(N1 / 2);
% Get the magnitude (absolute value) of the FFT components.
magY1 = abs(Y1(1:halfN1));
% Create a frequency vector from 0 Hz up to almost fs1/2.
f1 = (0:halfN1-1) * (fs1 / N1);

%% Compute FFT and Magnitude Spectrum for the Second Audio File
N2 = length(y2);
Y2 = fft(y2);
halfN2 = floor(N2 / 2);
magY2 = abs(Y2(1:halfN2));
f2 = (0:halfN2-1) * (fs2 / N2);

%% Prepare Data for a Logarithmic Frequency Plot
% Remove the DC component (0 Hz) to avoid problems with taking the log.
f1_log = f1(2:end);
% Convert the magnitude to decibels using 20*log10.
magY1_db = 20 * log10(magY1(2:end));
f2_log = f2(2:end);
magY2_db = 20 * log10(magY2(2:end));

%% Determine the Maximum Common Frequency for Comparison
% Use the lower Nyquist frequency (half the sampling rate) of the two files.
common_max_freq = min(fs1, fs2) / 2;

%% Prompt the User to Choose the Frequency Range to Display
% Ask the user to enter the lower and upper frequency limits.
prompt = {'Enter the lower frequency limit (Hz):', 'Enter the upper frequency limit (Hz):'};
dlgtitle = 'Specify Frequency Range for Display';
dims = [1 50];
% Set default values: lower limit at 20 Hz (or first available) and upper limit as the common max.
defaultLower = num2str(max(20, f1_log(1)));
defaultUpper = num2str(common_max_freq);
definput = {defaultLower, defaultUpper};
answer = inputdlg(prompt, dlgtitle, dims, definput);

if isempty(answer)
    % If you cancel, just use the full common frequency range.
    freq_lower = f1_log(1);
    freq_upper = common_max_freq;
else
    % Convert the answers to numeric values.
    freq_lower = str2double(answer{1});
    freq_upper = str2double(answer{2});
    
    % Make sure the inputs are valid. If not, show a warning and use default range.
    if isnan(freq_lower) || isnan(freq_upper) || freq_lower < f1_log(1) || ...
            freq_upper > common_max_freq || freq_lower >= freq_upper
        warning('Invalid frequency range input. Using default range from %.2f Hz to %.2f Hz.', ...
            f1_log(1), common_max_freq);
        freq_lower = f1_log(1);
        freq_upper = common_max_freq;
    end
end

%% Plot the Magnitude Spectra Using a Semilogarithmic Frequency Axis
figure;  % Open a new window for the plot.
% Plot the first audio file's magnitude spectrum in blue.
semilogx(f1_log, magY1_db, 'b', 'LineWidth', 1.2, 'DisplayName', file1);
hold on;
% Plot the second audio file's magnitude spectrum in red.
semilogx(f2_log, magY2_db, 'r', 'LineWidth', 1.2, 'DisplayName', file2);
hold off;
title('Magnitude Spectrum (dB) Comparison');
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
legend('show');  % Add legend to label the lines.
grid on;       % Turn on the grid for easier reading.
xlim([freq_lower, freq_upper]);  % Set the x-axis limits based on user input.
ax = gca;      % Get current axes to adjust settings.
ax.XAxis.Exponent = 0;  % Show full numbers on x-axis instead of exponential form.
movegui(gcf, "north"); % Move the figure window to the top of the screen.

%% Compute Phase Spectra (unwrapped, excluding the DC component)
% Calculate the phase angles from the FFT data. Start at the second element to skip 0 Hz.
phaseY1 = unwrap(angle(Y1(2:halfN1)));
phaseY2 = unwrap(angle(Y2(2:halfN2)));

%% Plot Phase Spectra in a New Figure
figure;  % Open a new window for the phase plot.
% Plot the phase for the first audio file.
semilogx(f1_log, phaseY1, 'b', 'LineWidth', 1.2, 'DisplayName', file1);
hold on;
% Plot the phase for the second audio file.
semilogx(f2_log, phaseY2, 'r', 'LineWidth', 1.2, 'DisplayName', file2);
hold off;
title('Phase Spectrum Comparison');
xlabel('Frequency (Hz)');
ylabel('Phase (radians)');
legend('show');
grid on;
xlim([freq_lower, freq_upper]);  % Use the user-specified frequency limits.
ax = gca;
ax.XAxis.Exponent = 0;
movegui(gcf, "northwest");  % Move this figure to the top left.

%% Time-Frequency Analysis: Spectrogram for Each Signal
% Set up parameters for the spectrogram:
%   window_length: sets time resolution (approx. 50 ms).
%   noverlap: overlapping points (approx. 25 ms).
%   nfft: number of points for the FFT (controls frequency resolution).

% For the first audio file:
window_length1 = round(0.05 * fs1);   % About 50 ms.
noverlap1 = round(0.025 * fs1);         % About 25 ms overlap.
nfft1 = 1024;                         % FFT points.

% For the second audio file:
window_length2 = round(0.05 * fs2);
noverlap2 = round(0.025 * fs2);
nfft2 = 1024;

figure;  % New window for spectrograms.

% Plot the spectrogram for the first audio file.
subplot(2,1,1);  % Divide the window into two rows; use the first row.
spectrogram(y1, window_length1, noverlap1, nfft1, fs1, 'yaxis');
title(['Spectrogram of ', file1]);
xlabel('Time (s)');
ylabel('Frequency (kHz)');
colorbar;  % Add a color bar to show the power scale.

% Plot the spectrogram for the second audio file.
subplot(2,1,2);  % Use the second row of the window.
spectrogram(y2, window_length2, noverlap2, nfft2, fs2, 'yaxis');
title(['Spectrogram of ', file2]);
xlabel('Time (s)');
ylabel('Frequency (kHz)');
colorbar;  % Add a color bar.
movegui(gcf, "northeast");  % Position the window at the top right.
