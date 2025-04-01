%% MATLAB Room Impulse Response with Material Properties
% Enhanced version with user-defined materials, frequency-dependent absorption, and wet/dry mixing

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

% Impulse response duration calculation
max_distance = sqrt(...
    (room_length*(max_order+1))^2 + ...  % x-direction
    (room_width*(max_order+1))^2 + ...   % y-direction
    (room_height*(max_order+1))^2);      % z-direction

max_delay = max_distance / c;
ir_duration = max_delay * 1.1;  % Add 10% padding
N = round(fs * ir_duration);    % Number of samples in the IR
time_axis = (0:N-1) / fs;       % Time vector in seconds

%% Material Database (Frequency-Dependent Absorption Coefficients)
materials = struct();
materials.concrete = struct('name', 'Concrete', 'absorption', [0.01, 0.02, 0.02, 0.03, 0.03, 0.04]);
materials.glass = struct('name', 'Glass', 'absorption', [0.03, 0.03, 0.02, 0.02, 0.02, 0.02]);
materials.wood = struct('name', 'Wood', 'absorption', [0.10, 0.10, 0.07, 0.06, 0.06, 0.07]);
materials.plaster = struct('name', 'Plaster', 'absorption', [0.10, 0.06, 0.04, 0.04, 0.05, 0.05]);
materials.carpet = struct('name', 'Carpet', 'absorption', [0.08, 0.24, 0.57, 0.69, 0.71, 0.73]);
materials.acoustic_panel = struct('name', 'Acoustic Panel', 'absorption', [0.25, 0.45, 0.85, 0.90, 0.85, 0.80]);
materials.curtains = struct('name', 'Heavy Curtains', 'absorption', [0.07, 0.31, 0.49, 0.75, 0.70, 0.60]);

freq_bands = [125, 250, 500, 1000, 2000, 4000]; % Hz

%% User Input for Surface Materials
disp('Available materials:');
material_names = fieldnames(materials);
for i = 1:length(material_names)
    fprintf('%d: %s\n', i, materials.(material_names{i}).name);
end

fprintf('\nSelect materials for each surface (enter number 1-%d):\n', length(material_names));
left_wall_mat = input('Left wall material: ');
right_wall_mat = input('Right wall material: ');
front_wall_mat = input('Front wall material: ');
back_wall_mat = input('Back wall material: ');
floor_mat = input('Floor material: ');
ceiling_mat = input('Ceiling material: ');

inputs = [left_wall_mat, right_wall_mat, front_wall_mat, back_wall_mat, floor_mat, ceiling_mat];
if any(inputs < 1) || any(inputs > length(material_names))
    error('Invalid material selection. Please choose numbers between 1 and %d', length(material_names));
end

surface_materials = {
    materials.(material_names{left_wall_mat}),
    materials.(material_names{right_wall_mat}),
    materials.(material_names{front_wall_mat}),
    materials.(material_names{back_wall_mat}),
    materials.(material_names{floor_mat}),
    materials.(material_names{ceiling_mat})
};

%% Frequency-Dependent Impulse Response Calculation
h_bands = zeros(N, length(freq_bands));
fprintf('\nComputing impulse response...\n');
total_iterations = (2*max_order+1)^3;
progress_interval = floor(total_iterations/10);
count = 0;

for nx = -max_order:max_order
    img_x = src_pos(1) + nx * room_length;
    if mod(nx,2) ~= 0
        img_x = (room_length - src_pos(1)) + nx * room_length;
    end
    
    x_reflection_coeff = ones(1, length(freq_bands));
    if nx ~= 0
        if nx > 0
            x_reflection_coeff = sqrt(1 - surface_materials{2}.absorption);
        else
            x_reflection_coeff = sqrt(1 - surface_materials{1}.absorption);
        end
    end
    
    for ny = -max_order:max_order
        img_y = src_pos(2) + ny * room_width;
        if mod(ny,2) ~= 0
            img_y = (room_width - src_pos(2)) + ny * room_width;
        end
        
        y_reflection_coeff = ones(1, length(freq_bands));
        if ny ~= 0
            if ny > 0
                y_reflection_coeff = sqrt(1 - surface_materials{4}.absorption);
            else
                y_reflection_coeff = sqrt(1 - surface_materials{3}.absorption);
            end
        end
        
        for nz = -max_order:max_order
            img_z = src_pos(3) + nz * room_height;
            if mod(nz,2) ~= 0
                img_z = (room_height - src_pos(3)) + nz * room_height;
            end
            
            z_reflection_coeff = ones(1, length(freq_bands));
            if nz ~= 0
                if nz > 0
                    z_reflection_coeff = sqrt(1 - surface_materials{6}.absorption);
                else
                    z_reflection_coeff = sqrt(1 - surface_materials{5}.absorption);
                end
            end
            
            distance = sqrt((img_x-rec_pos(1))^2 + (img_y-rec_pos(2))^2 + (img_z-rec_pos(3))^2);
            time_delay = distance / c;
            sample_delay = round(time_delay * fs) + 1;
            
            x_reflections = abs(nx);
            y_reflections = abs(ny);
            z_reflections = abs(nz);
            
            total_reflection_coeff = (x_reflection_coeff.^x_reflections) .* ...
                                   (y_reflection_coeff.^y_reflections) .* ...
                                   (z_reflection_coeff.^z_reflections);
            
            attenuation = 1 / distance;
            
            if sample_delay <= N
                h_bands(sample_delay, :) = h_bands(sample_delay, :) + ...
                    total_reflection_coeff * attenuation;
            end
            
            count = count + 1;
            if mod(count, progress_interval) == 0
                fprintf('Progress: %d%%\n', round(count/total_iterations*100));
            end
        end
    end
end

h = mean(h_bands, 2);
h = h / max(abs(h)); % Normalize impulse response

%% Wet/Dry Mix Processing
fprintf('\n=== Wet/Dry Mix Configuration ===\n');
mix_ratio = input('Enter wet/dry mix ratio (0=dry, 1=wet, 0.5=equal): ');
test_signal_type = input('Choose test signal [1=Impulse, 2=White noise, 3=Sine sweep, 4=Custom audio]: ');

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
            dry_signal = zeros(N,1);
            dry_signal(1) = 1;
            signal_name = 'Impulse (fallback)';
        else
            [dry_signal, fs_audio] = audioread(fullfile(path,file));
            if fs_audio ~= fs
                dry_signal = resample(dry_signal, fs, fs_audio);
            end
            dry_signal = mean(dry_signal,2);
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

% Process through room
wet_signal = fftfilt(h, dry_signal);
wet_signal = wet_signal(1:N);

% Level matching
target_gain = 10^(-3/20); % -3dB crossover point
dry_gain = (1-mix_ratio);
wet_gain = mix_ratio * target_gain * (rms(dry_signal)/rms(wet_signal));

% Apply mixing
mixed_signal = dry_gain*dry_signal + wet_gain*wet_signal;
mixed_signal = mixed_signal/max(abs(mixed_signal));

%% Visualization
% Room impulse response plot
figure('Position', [100, 100, 900, 600]);
subplot(2,1,1);
plot(time_axis, h);
xlabel('Time (s)'); ylabel('Amplitude');
title(sprintf('Room Impulse Response\n%s walls, %s ceiling', ...
    surface_materials{1}.name, surface_materials{6}.name));
grid on; xlim([0, min(ir_duration,1)]);

% Absorption coefficients
subplot(2,1,2); hold on;
colors = lines(length(surface_materials));
for i = 1:length(surface_materials)
    plot(freq_bands, surface_materials{i}.absorption, 'o-', ...
        'Color', colors(i,:), 'LineWidth', 1.5);
end
set(gca, 'XScale', 'log'); xticks(freq_bands);
xticklabels(arrayfun(@num2str, freq_bands, 'UniformOutput', false));
xlabel('Frequency (Hz)'); ylabel('Absorption Coefficient');
title('Surface Absorption Coefficients');
legend({surface_materials{1:6}.name}, 'Location', 'eastoutside');
grid on;

% Wet/dry comparison
figure('Position', [100, 100, 900, 900]);
subplot(3,1,1);
plot(time_axis, dry_signal);
title(['Dry Signal (' signal_name ')']); xlim([0 min(ir_duration,1)]);
grid on;

subplot(3,1,2);
plot(time_axis, wet_signal, 'r');
title('Wet Signal (Room Processed)'); xlim([0 min(ir_duration,1)]);
grid on;

subplot(3,1,3);
plot(time_axis, mixed_signal, 'Color', [0 0.5 0]);
title(sprintf('Mixed Output (%.0f%% Dry, %.0f%% Wet)', 100*(1-mix_ratio), 100*mix_ratio));
xlim([0 min(ir_duration,1)]); grid on;

%% Audio Playback and Export
play_choice = input('\nPlay audio comparison? [1=Dry, 2=Wet, 3=Mixed, 0=None]: ');
if play_choice > 0
    fprintf('Playing... (Press any key to stop)\n');
    switch play_choice
        case 1, soundsc(dry_signal, fs);
        case 2, soundsc(wet_signal, fs);
        case 3, soundsc(mixed_signal, fs);
    end
    pause;
end

save_choice = input('Save mixed output as WAV? [1=Yes, 0=No]: ');
if save_choice == 1
    fname = sprintf('RoomIR_%dx%dx%d_%s.wav',...
        room_length, room_width, room_height,...
        datestr(now,'yyyymmdd_HHMMSS'));
    audiowrite(fname, mixed_signal, fs);
    fprintf('Saved as %s\n', fname);
end

fprintf('\nProcessing complete.\n');