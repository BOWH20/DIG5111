function y = tanhSoftClip(x, gain)
    % Soft clipping using the hyperbolic tangent function
    % x: Input signal
    % gain: Gain applied before clipping (controls the amount of distortion)

    % Apply gain and tanh saturation
    y = tanh(gain * x);
end

fs = 44100; % Sampling frequency
ts = 1/fs; % Sampling period
dur = 1; % Duration of the signal in seconds
NumberOfSteps = 10; % Number of steps for quantization

% Time vector
t = 0:ts:dur;

% Define a sinusoidal input signal
f = 440; % Frequency of the sine wave (440 Hz = A4 note)
z = sin(2*pi*f*t); % Sinusoidal signal

% Apply step quantization to the sinusoidal signal
y = z * NumberOfSteps; % Scale the signal by the number of steps
y = round(y); % Round to the nearest step
y = y * (1/NumberOfSteps); % Normalize the signal back to the original range

% Define the 10 ms time period
timePeriod = 0.01; % 10 ms
samplesToDisplay = round(timePeriod * fs); % Number of samples corresponding to 10 ms
t_10ms = t(1:samplesToDisplay); % Time vector for 10 ms
z_10ms = z(1:samplesToDisplay); % Original signal for 10 ms
y_10ms = y(1:samplesToDisplay); % Quantized signal for 10 ms

% Plot the original and quantized signals for 10 ms
subplot(211);
plot(t_10ms, z_10ms);
title('Original Sinusoidal Signal (10 ms)');
xlabel('Time (seconds)');
ylabel('Amplitude');
grid on;
hold on;

plot(t_10ms, y_10ms, '--');
legend('Original', 'Quantized');
grid on;

% Play the original sinusoidal signal
disp('Playing original sinusoidal signal...');
sound(z, fs);
pause(dur + 0.5); % Wait for the sound to finish playing

% Play the quantized sinusoidal signal
disp('Playing quantized sinusoidal signal...');
sound(y, fs);

% Optional: Plot the quantized signal separately for 10 ms
subplot(212);
plot(t_10ms, y_10ms);
title('Quantized Sinusoidal Signal (10 ms)');
xlabel('Time (seconds)');
ylabel('Amplitude');
grid on;

