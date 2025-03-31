%% Room Impulse Response with Surface-Specific Materials
clear all; close all; clc;

%% 1. Room and Simulation Parameters
% Room dimensions (meters)
room_length = 10;    % Length in x-direction
room_width  = 7;     % Width in y-direction
room_height = 3;     % Height in z-direction

% Sound source and receiver positions [x, y, z] in meters
src_pos = [3, 4, 1.5];   % Source position
rec_pos = [7, 2, 1.5];   % Receiver position

% Sampling parameters
fs = 44100;          % Sampling frequency in Hz
c  = 343;            % Speed of sound in m/s
max_order = 10;      % Maximum reflection order
ir_duration = 3.0;   % Impulse response duration (seconds)
N = round(fs * ir_duration); % Number of samples

%% 2. Material Definitions
% Frequency bands: [125Hz, 250Hz, 500Hz, 1kHz, 2kHz, 4kHz]
materials = struct();
materials.concrete = [0.01, 0.01, 0.02, 0.02, 0.03, 0.03];
materials.glass = [0.10, 0.06, 0.04, 0.03, 0.02, 0.02];
materials.carpet = [0.08, 0.24, 0.57, 0.69, 0.71, 0.73];
materials.wood = [0.15, 0.11, 0.10, 0.07, 0.06, 0.07];
materials.fabric = [0.14, 0.35, 0.55, 0.70, 0.70, 0.65];
materials.plaster = [0.14, 0.10, 0.06, 0.05, 0.04, 0.03];

%% 3. User Material Assignment
surface_materials = struct();
surface_materials.left_wall = 'concrete';
surface_materials.right_wall = 'concrete';
surface_materials.front_wall = 'glass';
surface_materials.back_wall = 'wood';
surface_materials.floor = 'carpet';
surface_materials.ceiling = 'plaster';

 %Optional interactive material selection (uncomment to use)
 fprintf('Assign materials to surfaces:\n');
 surfaces = {'left_wall', 'right_wall', 'front_wall', 'back_wall', 'floor', 'ceiling'};
 material_names = fieldnames(materials);
 
 for i = 1:length(surfaces)
     fprintf('\nSurface: %s\n', surfaces{i});
     fprintf('Available materials:\n');
     for j = 1:length(material_names)
         fprintf('%d. %s\n', j, material_names{j});
     end
     choice = input(sprintf('Choose material for %s (1-%d): ', surfaces{i}, length(material_names)));
     surface_materials.(surfaces{i}) = material_names{choice};
 end

% Extract absorption coefficients
alpha_left  = materials.(surface_materials.left_wall);
alpha_right = materials.(surface_materials.right_wall);
alpha_front = materials.(surface_materials.front_wall);
alpha_back  = materials.(surface_materials.back_wall);
alpha_floor = materials.(surface_materials.floor);
alpha_ceiling = materials.(surface_materials.ceiling);

%% 4. Frequency-Dependent ISM Calculation
freq_bands = [125, 250, 500, 1000, 2000, 4000]; % Hz
h_freq = zeros(N, length(freq_bands)); % Frequency-dependent IRs

for f = 1:length(freq_bands)
    h = zeros(N, 1); % Initialize impulse response for this frequency
    
    for nx = -max_order:max_order
        % Compute image source x-coordinate
        if mod(nx, 2) == 0
            img_x = src_pos(1) + nx * room_length;
        else
            img_x = (room_length - src_pos(1)) + nx * room_length;
        end
        
        for ny = -max_order:max_order
            % Compute image source y-coordinate
            if mod(ny, 2) == 0
                img_y = src_pos(2) + ny * room_width;
            else
                img_y = (room_width - src_pos(2)) + ny * room_width;
            end
            
            for nz = -max_order:max_order
                % Compute image source z-coordinate
                if mod(nz, 2) == 0
                    img_z = src_pos(3) + nz * room_height;
                else
                    img_z = (room_height - src_pos(3)) + nz * room_height;
                end
                
                % Distance and time delay
                distance = sqrt((img_x-rec_pos(1))^2 + (img_y-rec_pos(2))^2 + (img_z-rec_pos(3))^2);
                sample_delay = round((distance/c) * fs) + 1;
                
                % Surface-dependent reflection coefficient
                reflection_coeff = ...
                    (1 - alpha_left(f))^abs(nx) * ...    % Left wall
                    (1 - alpha_right(f))^abs(nx) * ...   % Right wall
                    (1 - alpha_front(f))^abs(ny) * ...   % Front wall
                    (1 - alpha_back(f))^abs(ny) * ...    % Back wall
                    (1 - alpha_floor(f))^max(0,nz) * ... % Floor
                    (1 - alpha_ceiling(f))^max(0,-nz);  % Ceiling
                
                % Add to impulse response
                if sample_delay <= N
                    h(sample_delay) = h(sample_delay) + reflection_coeff * (1/distance);
                end
            end
        end
    end
    
    h_freq(:,f) = h; % Store frequency band IR
end

%% 5. Combine Frequency Bands
% Simple weighted sum (replace with proper filtering for better accuracy)
h_combined = sum(h_freq, 2);

%% 6. Normalize and Plot
h_combined = h_combined/max(abs(h_combined)); % Normalize

% Plot impulse response
figure;
t = (0:N-1)/fs;
plot(t, h_combined);
xlabel('Time (s)');
ylabel('Amplitude');
title('Room Impulse Response with Material-Specific Absorption');
grid on;

% Plot frequency response
[H, freq] = freqz(h_combined, 1, 2048, fs);
figure;
semilogx(freq, 20*log10(abs(H)));
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('Frequency Response of Room IR');
grid on;
xlim([20 20000]);

%% 7. Save IR (optional)
% audiowrite('room_ir.wav', h_combined, fs);