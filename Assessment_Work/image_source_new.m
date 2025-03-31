%% MATLAB Room Impulse Response with Material Properties
% Enhanced version with user-defined materials and frequency-dependent absorption

%% 1. Parameter Definition and Material Database
clear; clc; close all;

% Room dimensions (meters)
room_length = 10;    % Length in x-direction
room_width  = 7;     % Width in y-direction
room_height = 3;     % Height in z-direction

% Sound source and receiver positions [x, y, z] in meters
src_pos = [3, 4, 1.5];   % Source position
rec_pos = [7, 2, 1.5];   % Receiver position

% Sampling parameters and speed of sound
fs = 44100;          % Sampling frequency in Hz
c  = 343;            % Speed of sound in m/s

% Maximum reflection order
max_order = 75;      % Reduced for faster computation (increase if needed)

% Impulse response duration (seconds)
% Calculate maximum possible distance for given reflection order
max_distance = sqrt(...
    (room_length*(max_order+1))^2 + ...  % x-direction
    (room_width*(max_order+1))^2 + ...   % y-direction
    (room_height*(max_order+1))^2);      % z-direction

% Calculate maximum possible time delay
max_delay = max_distance / c;

% Set impulse response duration with 10% padding
ir_duration = max_delay * 1.1;
N = round(fs * ir_duration);    % Number of samples in the IR   % Number of samples in the IR

%% Material Database (Frequency-Dependent Absorption Coefficients)
% Define common materials with their absorption coefficients at different frequencies
% Frequencies: 125Hz, 250Hz, 500Hz, 1kHz, 2kHz, 4kHz
materials = struct();

% Example materials (values from common acoustic measurements)
materials.concrete = struct('name', 'Concrete', 'absorption', [0.01, 0.02, 0.02, 0.03, 0.03, 0.04]);
materials.glass = struct('name', 'Glass', 'absorption', [0.03, 0.03, 0.02, 0.02, 0.02, 0.02]);
materials.wood = struct('name', 'Wood', 'absorption', [0.10, 0.10, 0.07, 0.06, 0.06, 0.07]);
materials.plaster = struct('name', 'Plaster', 'absorption', [0.10, 0.06, 0.04, 0.04, 0.05, 0.05]);
materials.carpet = struct('name', 'Carpet', 'absorption', [0.08, 0.24, 0.57, 0.69, 0.71, 0.73]);
materials.acoustic_panel = struct('name', 'Acoustic Panel', 'absorption', [0.25, 0.45, 0.85, 0.90, 0.85, 0.80]);
materials.curtains = struct('name', 'Heavy Curtains', 'absorption', [0.07, 0.31, 0.49, 0.75, 0.70, 0.60]);

% Frequency bands for absorption coefficients
freq_bands = [125, 250, 500, 1000, 2000, 4000]; % Hz

%% User Input for Surface Materials
% Display available materials
disp('Available materials:');
material_names = fieldnames(materials);
for i = 1:length(material_names)
    fprintf('%d: %s\n', i, materials.(material_names{i}).name);
end

% Get user input for each surface
fprintf('\nSelect materials for each surface (enter number 1-%d):\n', length(material_names));
left_wall_mat = input('Left wall material: ');
right_wall_mat = input('Right wall material: ');
front_wall_mat = input('Front wall material: ');
back_wall_mat = input('Back wall material: ');
floor_mat = input('Floor material: ');
ceiling_mat = input('Ceiling material: ');

% Validate input
inputs = [left_wall_mat, right_wall_mat, front_wall_mat, back_wall_mat, floor_mat, ceiling_mat];
if any(inputs < 1) || any(inputs > length(material_names))
    error('Invalid material selection. Please choose numbers between 1 and %d', length(material_names));
end

% Store material selections
surface_materials = {
    materials.(material_names{left_wall_mat}),   % Left wall
    materials.(material_names{right_wall_mat}),  % Right wall
    materials.(material_names{front_wall_mat}),  % Front wall
    materials.(material_names{back_wall_mat}),   % Back wall
    materials.(material_names{floor_mat}),       % Floor
    materials.(material_names{ceiling_mat})      % Ceiling
};

%% Frequency-Dependent Impulse Response Calculation
% Initialize multi-band impulse response
h_bands = zeros(N, length(freq_bands));

% Display progress
fprintf('\nComputing impulse response...\n');
total_iterations = (2*max_order+1)^3;
progress_interval = floor(total_iterations/10); % Update progress every 10%
count = 0;

for nx = -max_order:max_order
    % Compute image source x-coordinate
    if mod(nx,2) == 0
        img_x = src_pos(1) + nx * room_length;
    else
        img_x = (room_length - src_pos(1)) + nx * room_length;
    end
    
    % Determine x-reflection coefficient (left/right walls)
    if nx ~= 0
        if nx > 0
            % Right wall reflections
            x_reflection_coeff = sqrt(1 - surface_materials{2}.absorption);
        else
            % Left wall reflections
            x_reflection_coeff = sqrt(1 - surface_materials{1}.absorption);
        end
    else
        x_reflection_coeff = ones(1, length(freq_bands));
    end
    
    for ny = -max_order:max_order
        % Compute image source y-coordinate
        if mod(ny,2) == 0
            img_y = src_pos(2) + ny * room_width;
        else
            img_y = (room_width - src_pos(2)) + ny * room_width;
        end
        
        % Determine y-reflection coefficient (front/back walls)
        if ny ~= 0
            if ny > 0
                % Back wall reflections
                y_reflection_coeff = sqrt(1 - surface_materials{4}.absorption);
            else
                % Front wall reflections
                y_reflection_coeff = sqrt(1 - surface_materials{3}.absorption);
            end
        else
            y_reflection_coeff = ones(1, length(freq_bands));
        end
        
        for nz = -max_order:max_order
            % Compute image source z-coordinate
            if mod(nz,2) == 0
                img_z = src_pos(3) + nz * room_height;
            else
                img_z = (room_height - src_pos(3)) + nz * room_height;
            end
            
            % Determine z-reflection coefficient (floor/ceiling)
            if nz ~= 0
                if nz > 0
                    % Ceiling reflections
                    z_reflection_coeff = sqrt(1 - surface_materials{6}.absorption);
                else
                    % Floor reflections
                    z_reflection_coeff = sqrt(1 - surface_materials{5}.absorption);
                end
            else
                z_reflection_coeff = ones(1, length(freq_bands));
            end
            
            % Compute distance and delay
            distance = sqrt((img_x - rec_pos(1))^2 + (img_y - rec_pos(2))^2 + (img_z - rec_pos(3))^2);
            time_delay = distance / c;
            sample_delay = round(time_delay * fs) + 1;
            
            % Count reflections per surface type
            x_reflections = abs(nx);
            y_reflections = abs(ny);
            z_reflections = abs(nz);
            
            % Calculate total reflection coefficient for each frequency band
            total_reflection_coeff = (x_reflection_coeff.^x_reflections) .* ...
                                   (y_reflection_coeff.^y_reflections) .* ...
                                   (z_reflection_coeff.^z_reflections);
            
            % Attenuation due to spherical spreading
            attenuation = 1 / distance;
            
            % Add to the impulse response for each frequency band
            if sample_delay <= N
                h_bands(sample_delay, :) = h_bands(sample_delay, :) + ...
                    total_reflection_coeff * attenuation;
            end
            
            % Progress update
            count = count + 1;
            if mod(count, progress_interval) == 0
                fprintf('Progress: %d%%\n', round(count/total_iterations*100));
            end
        end
    end
end

% Combine frequency bands (simple average - more advanced combination possible)
h = mean(h_bands, 2);

%% Normalize the impulse response
h = h / max(abs(h));

%% Plotting Results
time_axis = (0:N-1) / fs;  % Time vector in seconds

% Plot impulse response
figure('Position', [100, 100, 900, 600]);
subplot(2,1,1);
plot(time_axis, h);
xlabel('Time (s)');
ylabel('Amplitude');
title(sprintf('Room Impulse Response\nMaterials: %s (Left), %s (Right), %s (Front), %s (Back), %s (Floor), %s (Ceiling)', ...
    surface_materials{1}.name, surface_materials{2}.name, surface_materials{3}.name, ...
    surface_materials{4}.name, surface_materials{5}.name, surface_materials{6}.name));
grid on;
xlim([0, ir_duration]);

% Plot absorption coefficients
subplot(2,1,2);
hold on;
colors = lines(length(surface_materials));
legend_entries = cell(1, length(surface_materials));
for i = 1:length(surface_materials)
    plot(freq_bands, surface_materials{i}.absorption, 'o-', ...
        'Color', colors(i,:), 'LineWidth', 1.5);
    legend_entries{i} = surface_materials{i}.name;
end
set(gca, 'XScale', 'log');
xticks(freq_bands);
xticklabels(arrayfun(@num2str, freq_bands, 'UniformOutput', false));
xlabel('Frequency (Hz)');
ylabel('Absorption Coefficient');
title('Surface Absorption Coefficients');
legend(legend_entries, 'Location', 'eastoutside');
grid on;

fprintf('\nImpulse response calculation complete.\n');