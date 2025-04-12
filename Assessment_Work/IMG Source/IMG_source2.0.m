%% MATLAB Room Impulse Response with Material Properties (Revised)
% This version uses fractional delay interpolation (with proper vector shaping)
% to mitigate phase errors in high-order reflections and reduce frequency sweep artifacts.

%% 1. Parameter Definition and Material Database
clear; clc; close all;
% At start:
last_update = tic;
update_interval = 5; % seconds

% Get room dimensions from user
disp('Enter room dimensions in meters:');
room_length = input('Room length (x-direction): ');
room_width = input('Room width (y-direction): ');
room_height = input('Room height (z-direction): ');

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

% Maximum reflection order (controls the number of image sources)
max_order = input('Enter maximum reflection order (recommended 50-100): ');
if max_order <= 0
    error('Reflection order must be a positive integer.');
end

% Impulse response duration (seconds)
max_distance = sqrt( (room_length*(max_order+1))^2 + (room_width*(max_order+1))^2 + (room_height*(max_order+1))^2 );
max_delay = max_distance / c;
ir_duration = max_delay * 1.5;   % 50% extra padding
N = round(fs * ir_duration);     % Length in samples

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
new_freq_bands = linspace(min(freq_bands), max(freq_bands), 100); 
material_names = fieldnames(materials);
for i = 1:length(material_names)
    materials.(material_names{i}).absorption_interp = interp1(freq_bands, ...
        materials.(material_names{i}).absorption, new_freq_bands, 'linear');
end

%% User Input for Surface Materials
disp('Available materials:');
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

surface_materials = { ...
    materials.(material_names{left_wall_mat}), ...   % Left wall
    materials.(material_names{right_wall_mat}), ...    % Right wall
    materials.(material_names{front_wall_mat}), ...    % Front wall
    materials.(material_names{back_wall_mat}), ...     % Back wall
    materials.(material_names{floor_mat}), ...         % Floor
    materials.(material_names{ceiling_mat}) ...        % Ceiling
};

%% Optimized Frequency-Dependent Impulse Response Calculation
fprintf('\nComputing impulse response (optimized)...\n');
tic;

% Use fewer frequency bands (logarithmically spaced) for efficiency
new_freq_bands = logspace(log10(min(freq_bands)), log10(max(freq_bands)), 25)';

% Preallocate impulse response matrix for each frequency band contribution
h_bands = zeros(N, length(new_freq_bands));

% Pre-calculate image source positions (for the entire room)
nx_values = -max_order:max_order;
ny_values = -max_order:max_order;
nz_values = -max_order:max_order;
[all_nx, all_ny, all_nz] = ndgrid(nx_values, ny_values, nz_values);

% Compute image source positions using the mirror image method
img_x = src_pos(1) + all_nx * room_length;
odd_x = mod(all_nx,2)~=0;
img_x(odd_x) = (room_length - src_pos(1)) + img_x(odd_x);

img_y = src_pos(2) + all_ny * room_width;
odd_y = mod(all_ny,2)~=0;
img_y(odd_y) = (room_width - src_pos(2)) + img_y(odd_y);

img_z = src_pos(3) + all_nz * room_height;
odd_z = mod(all_nz,2)~=0;
img_z(odd_z) = (room_height - src_pos(3)) + img_z(odd_z);

% Compute distances from image sources to receiver
distances = sqrt((img_x - rec_pos(1)).^2 + (img_y - rec_pos(2)).^2 + (img_z - rec_pos(3)).^2);

% Reflection counts (absolute number of reflections on each axis)
x_reflections = abs(all_nx);
y_reflections = abs(all_ny);
z_reflections = abs(all_nz);

% Start a parallel pool if none is running
if isempty(gcp('nocreate'))
    parpool;
end

% For each frequency band, compute the corresponding impulse response component
parfor f_idx = 1:length(new_freq_bands)
    current_h = zeros(N, 1);
    current_freq = new_freq_bands(f_idx);
    
    % Get frequency-dependent absorption values for each surface
    alpha_left    = interp1(freq_bands, surface_materials{1}.absorption, current_freq, 'linear', 0);
    alpha_right   = interp1(freq_bands, surface_materials{2}.absorption, current_freq, 'linear', 0);
    alpha_front   = interp1(freq_bands, surface_materials{3}.absorption, current_freq, 'linear', 0);
    alpha_back    = interp1(freq_bands, surface_materials{4}.absorption, current_freq, 'linear', 0);
    alpha_floor   = interp1(freq_bands, surface_materials{5}.absorption, current_freq, 'linear', 0);
    alpha_ceiling = interp1(freq_bands, surface_materials{6}.absorption, current_freq, 'linear', 0);
    
    % Calculate reflection coefficients (square root of non-absorption)
    R_left    = sqrt(1 - alpha_left);
    R_right   = sqrt(1 - alpha_right);
    R_front   = sqrt(1 - alpha_front);
    R_back    = sqrt(1 - alpha_back);
    R_floor   = sqrt(1 - alpha_floor);
    R_ceiling = sqrt(1 - alpha_ceiling);
    
    % Process image sources in manageable chunks
    chunk_size = 5000;
    num_sources = numel(all_nx);
    for chunk_start = 1:chunk_size:num_sources
        chunk_end = min(chunk_start + chunk_size - 1, num_sources);
        chunk_idx = chunk_start:chunk_end;
        
        % Calculate total reflection coefficient for image sources in this chunk
        R_total = (R_left.^( x_reflections(chunk_idx).*(all_nx(chunk_idx) < 0) )) .* ...
                  (R_right.^(x_reflections(chunk_idx).*(all_nx(chunk_idx) > 0) )) .* ...
                  (R_front.^(y_reflections(chunk_idx).*(all_ny(chunk_idx) < 0) )) .* ...
                  (R_back.^( y_reflections(chunk_idx).*(all_ny(chunk_idx) > 0) )) .* ...
                  (R_floor.^(z_reflections(chunk_idx).*(all_nz(chunk_idx) < 0) )) .* ...
                  (R_ceiling.^(z_reflections(chunk_idx).*(all_nz(chunk_idx) > 0) ));
        
        % Skip negligible reflections
        valid = R_total > 0.0001;
        if ~any(valid)
            continue;
        end
        
        % Calculate time delays and attenuations for valid contributions
        chunk_distances = distances(chunk_idx(valid));
        time_delays = chunk_distances / c;
        % Compute the exact (fractional) sample delays
        sample_delays_exact = time_delays * fs + 1;  % +1 for MATLAB indexing
        attenuations = 1 ./ chunk_distances;
        contributions = R_total(valid) .* attenuations;
        
        % Apply air absorption (frequency-dependent damping)
        air_absorption_db = 0.005 * (current_freq/1000).^1.5;
        air_absorption_linear = 10.^(-air_absorption_db .* chunk_distances / 20);
        contributions = contributions .* air_absorption_linear;
        
        % --- Fractional Delay Interpolation ---
        % Compute lower and upper sample indices and fractional weights
        sample_lower = floor(sample_delays_exact);
        fraction = sample_delays_exact - sample_lower;
        sample_upper = sample_lower + 1;
        
        % Initialize temporary accumulator for the current chunk
        temp_h = zeros(N, 1);
        
        % Lower index accumulation (force column vectors)
        valid_lower = sample_lower >= 1 & sample_lower <= N;
        if any(valid_lower)
            idx_lower = sample_lower(valid_lower);
            weights_lower = (1 - fraction(valid_lower)) .* contributions(valid_lower);
            idx_lower = idx_lower(:);
            weights_lower = weights_lower(:);
            temp_h = temp_h + accumarray(idx_lower, weights_lower, [N, 1]);
        end
        
        % Upper index accumulation (force column vectors)
        valid_upper = sample_upper >= 1 & sample_upper <= N;
        if any(valid_upper)
            idx_upper = sample_upper(valid_upper);
            weights_upper = fraction(valid_upper) .* contributions(valid_upper);
            idx_upper = idx_upper(:);
            weights_upper = weights_upper(:);
            temp_h = temp_h + accumarray(idx_upper, weights_upper, [N, 1]);
        end
        
        % Add contributions from this chunk into the current frequency band's IR
        current_h = current_h + temp_h;
    end
    % Store the computed IR for this frequency band
    h_bands(:, f_idx) = current_h;
    
    % (Optional) Progress update inside parfor
    if mod(f_idx, 5) == 0
        fprintf('Completed %d/%d frequency bands (%.1f%%)\n', f_idx, length(new_freq_bands), 100 * f_idx / length(new_freq_bands));
    end
end

% Combine the frequency bands (averaging)
h = mean(h_bands, 2);
% Normalize the impulse response (adjust scaling as needed)
h = h / (max(abs(h)) * 5);

fprintf('Calculation completed in %.2f seconds\n', toc);

%% Wet/Dry Mix Configuration
fprintf('\n=== Wet/Dry Mix Configuration ===\n');
time_axis = (0:N-1) / fs;  % Time vector in seconds

% Get user preferences for mix ratio and test signal type
mix_ratio = input('Enter wet/dry mix ratio (0=dry, 1=wet, 0.5=equal): ');
test_signal_type = input('Choose test signal [1=Impulse, 2=White noise, 3=Sine sweep, 4=Custom audio]: ');

% Create dry signal based on user choice
switch test_signal_type
    case 1 % Impulse
        dry_signal = zeros(N,1); dry_signal(1) = 1;
        signal_name = 'Impulse';
    case 2 % White noise
        dry_signal = randn(N,1); dry_signal = dry_signal / max(abs(dry_signal));
        signal_name = 'White noise';
    case 3 % Sine sweep
        dry_signal = chirp(time_axis, 20, time_axis(end), 20000);
        dry_signal = dry_signal / max(abs(dry_signal));
        signal_name = 'Sine sweep';
    case 4 % Custom audio
        [file, path] = uigetfile({'*.wav;*.mp3;*.ogg;*.flac','Audio Files'});
        if isequal(file,0)
            disp('Using impulse as fallback');
            dry_signal = zeros(N,1); dry_signal(1) = 1;
            signal_name = 'Impulse (fallback)';
        else
            [dry_signal, fs_audio] = audioread(fullfile(path,file));
            if fs_audio ~= fs
                dry_signal = resample(dry_signal, fs, fs_audio);
            end
            dry_signal = mean(dry_signal,2); % Convert to mono
            if length(dry_signal) > N
                dry_signal = dry_signal(1:N);
            else
                dry_signal(end+1:N) = 0;
            end
            dry_signal = dry_signal / max(abs(dry_signal));
            [~, name] = fileparts(file);
            signal_name = ['Audio: ' name];
        end
end

% Ensure h and dry_signal have compatible lengths
if length(h) < length(dry_signal)
    h = [h; zeros(length(dry_signal) - length(h), 1)];
end

% Convolve the dry signal with the computed impulse response
wet_signal = fft_conv(dry_signal, h);
wet_signal = wet_signal(1:length(dry_signal));
wet_signal = wet_signal / (max(abs(wet_signal)) + 1e-6);

% Level matching
wet_rms = rms(wet_signal);
dry_rms = rms(dry_signal);
wet_gain = mix_ratio * (dry_rms / (wet_rms + 1e-6))^0.8;

% Define crossfade region (50 ms)
xfade_duration = 0.05;
xfade_samples = round(fs * xfade_duration);
fade_in = linspace(0, 1, xfade_samples)';
fade_out = linspace(1, 0, xfade_samples)';

% Apply crossfade between dry and wet signals
dry_signal(end - xfade_samples + 1:end) = dry_signal(end - xfade_samples + 1:end) .* fade_out;
wet_signal(1:xfade_samples) = wet_signal(1:xfade_samples) .* fade_in;

% Mix signals
mixed_signal = dry_signal * (1 - mix_ratio) + wet_signal * wet_gain;
if max(abs(mixed_signal)) > 1
    warning('Mixed signal is clipping! Reducing gain...');
    mixed_signal = mixed_signal / max(abs(mixed_signal));
end

%% Plotting
figure('Position', [100, 100, 900, 900]);
plot_duration = min(1, N/fs);
subplot(3,1,1);
plot(time_axis, h);
xlabel('Time (s)'); ylabel('Amplitude');
title(sprintf('Room Impulse Response\n%s walls, %s ceiling', surface_materials{1}.name, surface_materials{6}.name));
grid on; xlim([0, plot_duration]);

subplot(3,1,2);
hold on;
plot(time_axis, dry_signal, 'b', 'DisplayName', 'Dry');
plot(time_axis, wet_signal, 'r', 'DisplayName', 'Wet');
xlabel('Time (s)'); ylabel('Amplitude');
title(sprintf('Dry vs Wet Signals (%s)', signal_name));
legend('Location','northeast'); grid on; xlim([0, plot_duration]);

subplot(3,1,3);
plot(time_axis, mixed_signal, 'Color', [0 0.5 0]);
xlabel('Time (s)'); ylabel('Amplitude');
title(sprintf('Mixed Output (%.0f%% Dry, %.0f%% Wet)', 100*(1-mix_ratio), 100*mix_ratio));
grid on; xlim([0, plot_duration]);

%% Audio Playback (with tail handling)
play_audio = input('Play audio? [1=Yes, 0=No]: ');
if play_audio == 1
    wet_signal = fft_conv(dry_signal, h);
    dry_signal_extended = [dry_signal; zeros(length(wet_signal)-length(dry_signal), 1)];
    if length(dry_signal_extended) ~= length(wet_signal)
        error('Size mismatch: dry=%d, wet=%d', length(dry_signal_extended), length(wet_signal));
    end
    peak_dry = max(abs(dry_signal_extended)) + eps;
    peak_wet = max(abs(wet_signal)) + eps;
    max_peak = 0.5; % -6 dBFS headroom
    wet_gain = mix_ratio * (max_peak/peak_wet);
    dry_gain = (1-mix_ratio) * (max_peak/peak_dry);
    mixed_signal = dry_signal_extended * dry_gain + wet_signal * wet_gain;
    mixed_signal = mixed_signal / max(abs(mixed_signal));
    fprintf('Playing mixed audio (%.1f seconds)...\n', length(mixed_signal)/fs);
    sound(mixed_signal, fs);
    save_file = input('Save to WAV file? [1=Yes, 0=No]: ');
    if save_file == 1
        filename = sprintf('room_reverb_L%.1fxW%.1fxH%.1f_%s.wav',...
            room_length, room_width, room_height, datestr(now,'yyyy-mm-dd_HH-MM-SS'));
        audiowrite(filename, mixed_signal, fs);
        fprintf('Saved as %s\n', filename);
    end
end

%% Additional Plots
figure('Position', [100, 100, 900, 600]);
subplot(2,1,1);
plot(time_axis, h);
xlabel('Time (s)'); ylabel('Amplitude');
title(sprintf('Room Impulse Response\nMaterials: %s (Left), %s (Right), %s (Front), %s (Back), %s (Floor), %s (Ceiling)', ...
    surface_materials{1}.name, surface_materials{2}.name, surface_materials{3}.name, ...
    surface_materials{4}.name, surface_materials{5}.name, surface_materials{6}.name));
grid on; xlim([0, ir_duration]);

subplot(2,1,2);
hold on;
colors = lines(length(surface_materials));
legend_entries = cell(1, length(surface_materials));
for i = 1:length(surface_materials)
    plot(freq_bands, surface_materials{i}.absorption, 'o-', 'Color', colors(i,:), 'LineWidth', 1.5);
    legend_entries{i} = surface_materials{i}.name;
end
set(gca, 'XScale', 'log');
xticks(freq_bands);
xticklabels(arrayfun(@num2str, freq_bands, 'UniformOutput', false));
xlabel('Frequency (Hz)'); ylabel('Absorption Coefficient');
title('Surface Absorption Coefficients');
legend(legend_entries, 'Location', 'eastoutside');
grid on;

fprintf('\nProcessing complete.\n');

%% Helper Function: FFT Convolution (No Windowing)
function y = fft_conv(x, h)
    N = length(x);
    M = length(h);
    L = N + M - 1;
    x_padded = [x; zeros(L - N, 1)];
    h_padded = [h; zeros(L - M, 1)];
    X = fft(x_padded);
    H = fft(h_padded);
    Y = X .* H;
    y = ifft(Y, 'symmetric');
    y = y(1:L);
end
