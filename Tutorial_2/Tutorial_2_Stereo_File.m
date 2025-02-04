% Read the mono audio file
[monoSig, fs] = audioread(filename);  % Read full signal

% Create a stereo signal by duplicating the mono signal
stereoSig = [monoSig, monoSig];  % Create two identical channels

% Define output file name
outputFile = 'Piano_middle_C_stereo.wav';

% Write to a new stereo audio file
audiowrite(outputFile, stereoSig, fs);

% Confirm completion
disp('Stereo audio file has been created successfully.');

audioinfo("Piano_middle_C_stereo.wav")

stereoPiano = "Piano_middle_C_stereo.wav";  % File name (string)
[audioData, fs] = audioread(stereoPiano);  % Read audio file
sound(audioData, fs);  % Play the sound
