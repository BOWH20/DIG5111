% Parameters
NoiseFs = 22050;
NoiseDur = 2;
Noise = randn(NoiseFs * NoiseDur, 1); % Single-channel noise


b = filt.Numerator;
a = 1;

% Filtering
newnoise1 = conv(Noise, b); % Convolution
newnoise2 = filter(b, a, Noise); % Direct filtering

% Play sounds
sound(Noise, NoiseFs);
pause(3);
sound(newnoise1, NoiseFs);
pause(2.5);
sound(newnoise2, NoiseFs);

% Analyze signals
signalAnalyzer(Noise, newnoise1, newnoise2);

% Display lengths
len1 = length(newnoise1);
len2 = length(newnoise2);
disp(len1);
disp(len2);