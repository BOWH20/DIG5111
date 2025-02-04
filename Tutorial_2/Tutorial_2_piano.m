% Audio filename must be in the same directory as this m file
Filename = 'piano_middle_C.wav';


filename = 'Piano_middle_C.wav';  % File name
info = audioinfo(filename);       % Get audio file info

fs = info.SampleRate;             % Sampling frequency
t1 = 0.5;                         % Start time in seconds
t2 = 1.0;                         % End time in seconds

% Read only the required portion of the signal
samples = [round(t1 * fs), round(t2 * fs)];  
sig3 = audioread(filename, samples);

% Store in workspace method 2
assignin('base', 'sig3', sig3);

% Define output filename
outputFile = 'Piano_middle_C_0.5_1.0.wav';

% Write the extracted audio to a new file
audiowrite(outputFile, sig3, fs);

% Play the newly saved audio file
[y, fs] = audioread(outputFile); % Read the saved file
sound(y, fs); % Play the sound



filename = 'Piano_middle_C.wav';  % File name
[Sig, fs] = audioread(filename);  % Read full signal

t1 = 0.5;                         % Start time in seconds
t2 = 1.0;                         % End time in seconds

% Convert time to sample indices
startSample = round(t1 * fs);
endSample = round(t2 * fs);

% Extract the portion from 0.5s to 1.0s
sig2 = Sig(startSample:endSample, :);

% Store in workspace
assignin('base', 'Sig', Sig);
assignin('base', 'sig2', sig2);
