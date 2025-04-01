%% MATLAB Room Impulse Response with Material Properties
% Enhanced version with user-defined materials and frequency-dependent absorption

%% 1. Parameter Definition and Material Database
clear; clc; close all;

% Get room dimensions from user
disp('Enter room dimensions in meters:');
room_length = input('Room length (x-direction): ');
room_width = input('Room width (y-direction): ');
room_height = input('Room height (z-direction): ');

% Validate room dimensions
if any([room_length, room_width, room_height] <= 0)
    error('Room dimensions must be positive values.');
end

% Get source and receiver positions from user
disp('Enter positions in meters (must be within room dimensions):');
src_pos = input('Source position [x, y, z] (e.g., [3, 4, 1.5]): ');
rec_pos = input('Receiver position [x, y, z] (e.g., [7, 2, 1.5]): ');
    
% Validate positions
if src_pos(1) < 0 || src_pos(1) > room_length || ...
   src_pos(2) < 0 || src_pos(2) > room_width || ...
   src_pos(3) < 0 || src_pos(3) > room_height
    error('Source position is outside the room boundaries.');
end

if rec_pos(1) < 0 || rec_pos(1) > room_length || ...
   rec_pos(2) < 0 || rec_pos(2) > room_width || ...
   rec_pos(3) < 0 || rec_pos(3) > room_height
    error('Receiver position is outside the room boundaries.');
end

% Sampling parameters and speed of sound
fs = 44100;          % Sampling frequency in Hz
c = 343;             % Speed of sound in m/s

% Maximum reflection order
max_order = input('Enter maximum reflection order (recommended 50-100): ');
if max_order <= 0
    error('Reflection order must be a positive integer.');
end

% (The rest of your existing code continues here...)

% Impulse response duration (seconds)
% Calculate maximum possible distance for given reflection order
max_distance = sqrt(...
    (room_length*(max_order+1))^2 + ...  % x-direction
    (room_width*(max_order+1))^2 + ...   % y-direction
    (room_height*(max_order+1))^2);      % z-direction

% Calculate maximum possible time delay
max_delay = max_distance / c;

% Set impulse response duration with 10% padding
ir_duration = max_delay * 1.5;
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

new_freq_bands = linspace(min(freq_bands), max(freq_bands), 100); 

% Interpolated absorption coefficients for each material
materials.concrete.absorption_interp = interp1(freq_bands, materials.concrete.absorption, new_freq_bands, 'linear');
materials.glass.absorption_interp = interp1(freq_bands, materials.glass.absorption, new_freq_bands, 'linear');
materials.wood.absorption_interp = interp1(freq_bands, materials.wood.absorption, new_freq_bands, 'linear');
materials.plaster.absorption_interp = interp1(freq_bands, materials.plaster.absorption, new_freq_bands, 'linear');
materials.carpet.absorption_interp = interp1(freq_bands, materials.carpet.absorption, new_freq_bands, 'linear');
materials.acoustic_panel.absorption_interp = interp1(freq_bands, materials.acoustic_panel.absorption, new_freq_bands, 'linear');
materials.curtains.absorption_interp = interp1(freq_bands, materials.curtains.absorption, new_freq_bands, 'linear');


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
h_bands = zeros(N, length(new_freq_bands));

fprintf('\nComputing impulse response...\n');
total_iterations = (2*max_order+1)^3 * length(new_freq_bands);  % Account for frequency bands
progress_interval = floor(total_iterations / 100); % Update progress every 1%
count = 0;
for f_idx = 1:length(new_freq_bands)
    current_freq = new_freq_bands(f_idx); 

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
            x_reflection_coeff = sqrt(1 - interp1(freq_bands,surface_materials{2}.absorption, current_freq,"linear",0));
        else
            % Left wall reflections
            x_reflection_coeff = sqrt(1 - interp1(freq_bands, surface_materials{1}.absorption, current_freq, 'linear', 0));
        end
    else
        x_reflection_coeff = 1;
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
                y_reflection_coeff = sqrt(1 - interp1(freq_bands, surface_materials{4}.absorption, current_freq, 'linear', 0));
            else
                % Front wall reflections
                y_reflection_coeff = sqrt(1 - interp1(freq_bands, surface_materials{3}.absorption, current_freq, 'linear', 0));
            end
        else
            y_reflection_coeff = 1;
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
                    z_reflection_coeff = sqrt(1 - interp1(freq_bands, surface_materials{6}.absorption, current_freq, 'linear', 0));
                else
                    % Floor reflections
                    z_reflection_coeff = sqrt(1 - interp1(freq_bands, surface_materials{5}.absorption, current_freq, 'linear', 0));
                end
            else
                z_reflection_coeff = 1;
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
end

% Combine frequency bands (simple average - more advanced combination possible)
h = mean(h_bands, 2);

%% Normalize the impulse response
h = h / max(abs(h));


time_axis = (0:N-1) / fs;  % Time vector in seconds

%% Then add the Wet/Dry Mix section like this:

fprintf('\n=== Wet/Dry Mix Configuration ===\n');

% Get user preferences
mix_ratio = input('Enter wet/dry mix ratio (0=dry, 1=wet, 0.5=equal): ');
test_signal_type = input('Choose test signal [1=Impulse, 2=White noise, 3=Sine sweep, 4=Custom audio]: ');

% Create dry signal based on user choice
switch test_signal_type
    case 1 % Impulse
        dry_signal = zeros(N,1);
        dry_signal(1) = 1;
        signal_name = 'Impulse';
        
    case 2 % White noise
        dry_signal = randn(N,1);
        dry_signal = dry_signal/max(abs(dry_signal));
        signal_name = 'White noise';
        
    case 3 % Sine sweep
        dry_signal = chirp(time_axis, 20, time_axis(end), 20000);
        dry_signal = dry_signal/max(abs(dry_signal));
        signal_name = 'Sine sweep';
        
    case 4 % Custom audio
        [file, path] = uigetfile({'*.wav;*.mp3;*.ogg;*.flac','Audio Files'});
        if isequal(file,0)
            disp('Using impulse as fallback');
            dry_signal = zeros(N,1);
            dry_signal(1) = 1;
            signal_name = 'Impulse (fallback)';
        else
            [dry_signal, fs_audio] = audioread(fullfile(path,file));
            if fs_audio ~= fs
                dry_signal = resample(dry_signal, fs, fs_audio);
            end
            dry_signal = mean(dry_signal,2); % Convert to mono
            % Match length to impulse response
            if length(dry_signal) > N
                dry_signal = dry_signal(1:N);
            else
                dry_signal(end+1:N) = 0;
            end
            dry_signal = dry_signal/max(abs(dry_signal));
            [~,name] = fileparts(file);
            signal_name = ['Audio: ' name];
        end
end

h = [h; zeros(length(dry_signal) - length(h), 1)];
h = h / max(abs(h));  % Normalize the impulse response
if length(h) < length(dry_signal)
    h = [h; zeros(length(dry_signal) - length(h), 1)];
end


wet_signal = conv(dry_signal, h, 'full');  % Full convolution for richer tail
wet_signal = wet_signal(1:length(dry_signal)); % Trim to match length
wet_signal = wet_signal / max(abs(wet_signal) + 1e-6); % Normalize





% Level matching (important for natural sounding mix)
wet_signal = wet_signal / (max(abs(wet_signal)) + 1e-6);  % Normalize
wet_rms = rms(wet_signal);
wet_signal = lowpass(wet_signal, 18000, fs);  % Remove unnatural high frequencies

dry_rms = rms(dry_signal);
wet_gain = mix_ratio * (dry_rms / (wet_rms + 1e-6)) ^ 0.5;  % Less aggressive scaling




mixed_signal = dry_signal * (1 - mix_ratio) + wet_signal * wet_gain;
if max(abs(mixed_signal)) > 1
    warning('Mixed signal is clipping! Reducing gain...');
    mixed_signal = mixed_signal / max(abs(mixed_signal));
else
    mixed_signal = mixed_signal / rms(mixed_signal);
end


    


%% Enhanced Plotting
figure('Position', [100, 100, 900, 900]);
plot_duration = N/fs;
% Room impulse response plot
subplot(3,1,1);
plot(time_axis, h);
xlabel('Time (s)');
ylabel('Amplitude');
title(sprintf('Room Impulse Response\n%s walls, %s ceiling', ...
    surface_materials{1}.name, surface_materials{6}.name));
grid on;
xlim([0, plot_duration]); % Show first second only

% Dry vs Wet comparison
subplot(3,1,2);
hold on;
plot(time_axis, dry_signal, 'b', 'DisplayName', 'Dry');
plot(time_axis, wet_signal, 'r', 'DisplayName', 'Wet');
xlabel('Time (s)');
ylabel('Amplitude');
title(sprintf('Dry vs Wet Signals (%s)', signal_name));
legend('Location','northeast');
grid on;
xlim([0, plot_duration]); % Show first second only

% Mixed signal
subplot(3,1,3);
plot(time_axis, mixed_signal, 'Color', [0 0.5 0]);
xlabel('Time (s)');
ylabel('Amplitude');
title(sprintf('Mixed Output (%.0f%% Dry, %.0f%% Wet)', 100*(1-mix_ratio), 100*mix_ratio));
grid on;
xlim([0, plot_duration]); % Show first second only

%% Audio Playback
play_audio = input('Play audio? [1=Yes, 0=No]: ');
if play_audio == 1
    fprintf('Playing mixed audio...\n');
    soundsc(mixed_signal, fs);
    
    % Optional: Save to file
    save_file = input('Save to WAV file? [1=Yes, 0=No]: ');
    if save_file == 1
        filename = sprintf('room_sim_L%.1fxW%.1fxH%.1f_%s.wav', ...
            room_length, room_width, room_height, ...
            datestr(now,'yyyy-mm-dd_HH-MM-SS'));
        audiowrite(filename, mixed_signal, fs);
        fprintf('Saved as %s\n', filename);
    end
end



fprintf('\nProcessing complete.\n');
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
