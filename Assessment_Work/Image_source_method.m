%% MATLAB Starter Code for Image Source Method (ISM)
% This script generates a room impulse response using the Image Source Method.
% It is intended as a starting point for your DSP Assessment.
%
% Key Features:
%   1. Define a custom cuboid room (length, width, height).
%   2. Specify positions for the sound source and receiver.
%   3. Compute reflections up to a user-defined reflection order.
%   4. Include placeholders to incorporate absorption coefficients for each room surface.
%   5. Plot the resulting impulse response.
%   6. Provide a basic framework for later convolution with an audio signal.
%
% You are encouraged to expand the code (e.g., to include higher-order reflections
% or frequency-dependent absorption coefficients) by modifying the indicated sections.

%% 1. Parameter Definition
% Room dimensions (meters)
room_length = 10;    % Length in x-direction
room_width  = 7;     % Width in y-direction
room_height = 3;     % Height in z-direction

% Sound source and receiver (listener) positions [x, y, z] in meters
src_pos = [3, 4, 1.5];   % Source position
rec_pos = [7, 2, 1.5];   % Receiver position

% Sampling parameters and speed of sound
fs = 44100;          % Sampling frequency in Hz
c  = 343;            % Speed of sound in m/s

% Maximum reflection order (i.e. include reflections from -max_order to max_order in each dimension)
max_order = 50;

% Absorption coefficients for surfaces (values between 0 and 1)
% These represent the fraction of energy absorbed at each surface.
% For now, we use a uniform absorption coefficient.
% --- Students: Replace the following with surface-specific coefficients as desired.
absorption_wall   = 0.1;   % Example for left/right and front/back walls
absorption_floor  = 0.1;   % Example for floor
absorption_ceiling= 0.2;   % Example for ceiling
%
% For simplicity, we use a single uniform absorption value.
uniform_absorption = 0.05;  % (Placeholder value)
% Reflection coefficient per reflection (simple model):
reflection_coeff_single = 1 - uniform_absorption; 
% Note: In a more detailed model, you might compute the reflection coefficient 
% as the square root of (1 - absorption) to account for energy conservation.

% Impulse response duration (seconds)
ir_duration = 3.0;  
N = round(fs * ir_duration);    % Number of samples in the IR
h = zeros(N, 1);                % Preallocate the impulse response vector

%% 2. Image Source Computation and IR Construction
% Loop over reflection orders in each spatial dimension.
% The image source method creates virtual sources by “mirroring” the actual source.
% For each axis, the image source coordinate is computed as follows:
%   - If the reflection order (n) is even:  img_coord = n*room_dim + src_coord
%   - If odd:                         img_coord = n*room_dim + (room_dim - src_coord)
%
% The total number of reflections for a given image is the sum of the absolute
% values of the reflection orders in x, y, and z. This count is used to scale
% the amplitude (via the reflection coefficients) and to mimic energy loss.

for nx = -max_order:max_order
    % Compute image source x-coordinate
    if mod(nx,2) == 0
        img_x = src_pos(1) + nx * room_length;
    else
        img_x = (room_length - src_pos(1)) + nx * room_length;
    end
    
    for ny = -max_order:max_order
        % Compute image source y-coordinate
        if mod(ny,2) == 0
            img_y = src_pos(2) + ny * room_width;
        else
            img_y = (room_width - src_pos(2)) + ny * room_width;
        end
        
        for nz = -max_order:max_order
            % Compute image source z-coordinate
            if mod(nz,2) == 0
                img_z = src_pos(3) + nz * room_height;
            else
                img_z = (room_height - src_pos(3)) + nz * room_height;
            end
            
            % Compute the Euclidean distance from the image source to the receiver
            distance = sqrt((img_x - rec_pos(1))^2 + (img_y - rec_pos(2))^2 + (img_z - rec_pos(3))^2);
            
            % Compute the propagation delay in seconds and convert to a sample index
            time_delay = distance / c;
            sample_delay = round(time_delay * fs) + 1;  % +1 for MATLAB 1-indexing
            
            % Count the total number of reflections from all three dimensions
            num_reflections = abs(nx) + abs(ny) + abs(nz);
            
            % Compute the overall reflection coefficient for this image source.
            % --- Students: Modify this section to apply surface-specific absorption.
            % For example, you could assign different reflection coefficients depending on
            % which wall (left/right, front/back, floor/ceiling) was involved in the reflection.
            total_reflection_coeff = reflection_coeff_single^num_reflections;
            
            % Attenuation due to spherical spreading (inverse of distance)
            attenuation = 1 / distance;
            
            % Add the contribution from this image source to the impulse response,
            % if the computed sample delay is within the impulse response length.
            if sample_delay <= N
                h(sample_delay) = h(sample_delay) + total_reflection_coeff * attenuation;
            end
        end
    end
end

%% 3. Plot the Impulse Response
time_axis = (0:N-1) / fs;  % Time vector in seconds
figure;
stem(time_axis, h, 'Marker', 'none');
xlabel('Time (s)');
ylabel('Amplitude');
title('Room Impulse Response using Image Source Method');
grid on;

%% 4. Using the Impulse Response for Convolution Reverb
% The generated impulse response (vector h) can now be used to apply convolution
% reverb to an audio signal.
% Example (uncomment and modify as needed):
%
%   [audio, fs_audio] = audioread('your_audio_file.wav');
%   audio_reverberated = conv(audio, h);
%   soundsc(audio_reverberated, fs);
%
% Students can expand this section to include additional processing, such as
% normalizing the impulse response, using frequency-dependent absorption, or
% implementing higher-order reflections.

%% End of Script