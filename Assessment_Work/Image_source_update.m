%% Complete Corrected Image Source Method with Material Selection
clear all; close all; clc;

%% 1. Physical Constants and Parameters
c = 343;                  % Speed of sound (m/s)
fs = 44100;               % Sampling frequency (Hz)
ir_duration = 1.0;        % Impulse response duration (s)
max_order = 10;           % Maximum reflection order
N_fft = 2^nextpow2(fs * ir_duration); % FFT size
freq_vector = linspace(0, fs/2, N_fft/2+1)';

%% 2. Room Configuration
room_dim = [8 5 3];       % [length, width, height] in meters
src_pos = [2 2 1.5];      % Source position [x,y,z]
rec_pos = [6 3 1.5];      % Receiver position [x,y,z]

%% 3. Material Database Setup
materials = struct();

% Define materials with frequency-dependent absorption
material_data = {
    'curtain', 'Heavy Curtains', [125 0.05; 250 0.15; 500 0.35; 1000 0.55; 2000 0.65; 4000 0.65];
    'brick', 'Brick Wall', [125 0.01; 250 0.02; 500 0.03; 1000 0.04; 2000 0.05; 4000 0.05];
    'wood', 'Wood Panel', [125 0.10; 250 0.15; 500 0.25; 1000 0.30; 2000 0.35; 4000 0.35];
    'glass', 'Glass Window', [125 0.04; 250 0.06; 500 0.08; 1000 0.10; 2000 0.12; 4000 0.12];
    'carpet', 'Carpet Floor', [125 0.08; 250 0.24; 500 0.57; 1000 0.69; 2000 0.71; 4000 0.71];
    'ceiling', 'Acoustic Ceiling', [125 0.50; 250 0.65; 500 0.75; 1000 0.80; 2000 0.85; 4000 0.85]
};

for i = 1:size(material_data,1)
    materials.(material_data{i,1}) = struct(...
        'name', material_data{i,2}, ...
        'absorption', material_data{i,3});
end

%% 4. User Material Selection
material_names = fieldnames(materials);
disp('Available materials:');
for i = 1:length(material_names)
    fprintf('%d. %s\n', i, materials.(material_names{i}).name);
end

surface_names = {'Left Wall', 'Right Wall', 'Front Wall', 'Back Wall', 'Floor', 'Ceiling'};
selected_indices = zeros(1,6);

for i = 1:6
    while true
        fprintf('\nSelect material for %s (1-%d): ', surface_names{i}, length(material_names));
        choice = input('');
        if isscalar(choice) && choice >= 1 && choice <= length(material_names)
            selected_indices(i) = choice;
            break;
        else
            disp('Invalid selection. Please try again.');
        end
    end
end

%% 5. Prepare Surface Reflection Coefficients
surface_reflection = cell(6,1);
for i = 1:6
    abs_data = materials.(material_names{selected_indices(i)}).absorption;
    absorption = interp1(abs_data(:,1), abs_data(:,2), freq_vector, 'pchip', 'extrap');
    absorption = max(0, min(1, absorption));
    surface_reflection{i} = sqrt(1 - absorption);
end

%% 6. Image Source Method Calculation
N = round(fs * ir_duration);
h = zeros(N,1);

for nx = -max_order:max_order
    img_x = compute_image_coord(src_pos(1), room_dim(1), nx);
    x_hits = mod(1:abs(nx), 2) + 1; % 1=left, 2=right
    
    for ny = -max_order:max_order
        img_y = compute_image_coord(src_pos(2), room_dim(2), ny);
        y_hits = mod(1:abs(ny), 2) + 3; % 3=front, 4=back
        
        for nz = -max_order:max_order
            img_z = compute_image_coord(src_pos(3), room_dim(3), nz);
            z_hits = mod(1:abs(nz), 2) + 5; % 5=floor, 6=ceiling
            
            distance = sqrt((img_x-rec_pos(1))^2 + (img_y-rec_pos(2))^2 + (img_z-rec_pos(3))^2);
            sample_delay = round(distance/c * fs) + 1;
            
            if sample_delay <= N
                % Create and transform impulse
                temp_ir = zeros(N,1);
                temp_ir(sample_delay) = 1;
                temp_ir_fft = fft(temp_ir, N_fft);
                
                % Calculate combined reflection factor
                refl_factor = ones(N_fft,1);
                for r = x_hits
                    refl_factor = refl_factor .* surface_reflection{r};
                end
                for r = y_hits
                    refl_factor = refl_factor .* surface_reflection{r};
                end
                for r = z_hits
                    refl_factor = refl_factor .* surface_reflection{r};
                end
                
                % Apply to frequency domain
                refl_factor(end:-1:end-N_fft/2+1) = refl_factor(2:N_fft/2+1);
                temp_ir_fft = temp_ir_fft .* refl_factor / distance;
                
                % Add to impulse response
                h = h + real(ifft(temp_ir_fft, N_fft))(1:N);
            end
        end
    end
end

%% 7. Helper Function
function coord = compute_image_coord(src_coord, room_dim, order)
    if mod(order, 2) == 0
        coord = src_coord + order * room_dim;
    else
        coord = (room_dim - src_coord) + order * room_dim;
    end
end

%% 8. Results Visualization
h = h / max(abs(h));

figure;
subplot(2,1,1);
plot((0:N-1)/fs, h);
xlabel('Time (s)'); ylabel('Amplitude');
title('Room Impulse Response');

subplot(2,1,2);
H = fft(h, N_fft);
semilogx(freq_vector, 20*log10(abs(H(1:N_fft/2+1))));
xlabel('Frequency (Hz)'); ylabel('Magnitude (dB)');
title('Frequency Response');
xlim([20 fs/2]); grid on;

audiowrite('room_ir.wav', h, fs);
disp('Impulse response saved as room_ir.wav');