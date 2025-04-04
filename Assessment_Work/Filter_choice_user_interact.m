function interactiveAudioEQ()
    % Initialize variables that need to be shared with nested functions
    Fs = [];
    fc = [];
    b = [];
    a = [];
    
    % Display available filter types
    fprintf('Available filter types:\n');
    fprintf('1. Low Pass Filter (lpf)\n');
    fprintf('2. High Pass Filter (hpf)\n');
    fprintf('3. Band Pass Filter (bpf)\n');
    fprintf('4. Band Pass Filter (Constant Q) (bpfq)\n');
    fprintf('5. Notch Filter (notch)\n');
    fprintf('6. All Pass Filter (apf)\n');
    fprintf('7. Peaking EQ (peakingeq)\n');
    fprintf('8. Low Shelf (lowshelf)\n');
    fprintf('9. High Shelf (highshelf)\n\n');
    
    % Get user input for filter type
    valid_types = {'lpf', 'hpf', 'bpf', 'bpfq', 'notch', 'apf', 'peakingeq', 'lowshelf', 'highshelf'};
    while true
        type = input('Enter filter type (e.g., ''lpf'', ''hpf'', etc.): ', 's');
        if any(strcmpi(type, valid_types))
            break;
        else
            fprintf('Invalid filter type. Please try again.\n');
        end
    end
    
    % Get user input for center/cutoff frequency
    while true
        fc = input('Enter center/cutoff frequency (Hz): ');
        if fc > 0
            break;
        else
            fprintf('Frequency must be positive. Please try again.\n');
        end
    end
    
    % Get user input for gain (only needed for some filters)
    if any(strcmpi(type, {'peakingeq', 'lowshelf', 'highshelf'}))
        while true
            dBgain = input('Enter gain in dB (e.g., 6 for boost, -6 for cut): ');
            if isscalar(dBgain)
                break;
            else
                fprintf('Gain must be a scalar value. Please try again.\n');
            end
        end
    else
        dBgain = 0; % Default gain for filters that don't use it
    end
    
    % Get user input for sampling rate
    while true
        Fs = input('Enter sampling rate (Hz): ');
        if Fs > 0 && Fs > 2*fc
            break;
        else
            fprintf('Sampling rate must be positive and greater than twice the cutoff frequency. Please try again.\n');
        end
    end
    
    % Get user input for Q factor
    while true
        Q = input('Enter Q factor (e.g., 0.707 for Butterworth): ');
        if Q > 0
            break;
        else
            fprintf('Q factor must be positive. Please try again.\n');
        end
    end
    
    % Design the filter
    [b, a] = AudioEQFilter(fc, dBgain, Fs, Q, type);
    
    % Display filter coefficients
    fprintf('\nFilter coefficients:\n');
    fprintf('Numerator (b): [%f, %f, %f]\n', b(1), b(2), b(3));
    fprintf('Denominator (a): [%f, %f, %f]\n\n', a(1), a(2), a(3));
    
    % Visualize the filter response
    fvtool(b, a, 'Fs', Fs);
    
    % Optionally save coefficients to workspace
    assignin('base', 'b', b);
    assignin('base', 'a', a);
    fprintf('Filter coefficients saved to workspace as variables ''b'' and ''a''.\n');

    % Audio processing section
    fprintf('\nAudio Processing Options:\n');
    fprintf('1. Apply to test signal\n');
    fprintf('2. Apply to audio file\n');
    choice = input('Choose option (1 or 2): ');
    
    switch choice
        case 1 % Test signal
            % Generate a test signal
            duration = 3; % seconds
            t = 0:1/Fs:duration;
            test_signal = 0.5*sin(2*pi*500*t) + 0.5*sin(2*pi*fc*t) + 0.3*sin(2*pi*3000*t);
            
            % Apply filter
            filtered_signal = filter(b, a, test_signal);
            
            % Plot and play
            % Create a time window for better visualization
            window_start = 0; % Start time in seconds
            window_duration = 0.05; % Window duration in seconds (50ms)
            window_end = window_start + window_duration;
            window_samples = (t >= window_start) & (t <= window_end);
    


            figure;

            % Full signal view
            subplot(2,2,1);
            plot(t, test_signal);
            title('Original Signal (Full)');
            xlabel('Time (s)');
            ylabel('Amplitude');
            grid on;

            subplot(2,2,2);
            plot(t, filtered_signal);
            title('Filtered Signal (Full)');
            xlabel('Time (s)');
            ylabel('Amplitude');
            grid on;

            % Windowed view
            subplot(2,2,3);
            plot(t(window_samples), test_signal(window_samples));
            title(sprintf('Original Signal (%.0f-%.0f ms)', window_start*1000, window_end*1000));
            xlabel('Time (s)');
            ylabel('Amplitude');
            grid on;

            subplot(2,2,4);
            plot(t(window_samples), filtered_signal(window_samples));
            title(sprintf('Filtered Signal (%.0f-%.0f ms)', window_start*1000, window_end*1000));
            xlabel('Time (s)');
            ylabel('Amplitude');
            grid on;
            
            % Adjust figure size for better viewing
            set(gcf, 'Position', [100 100 1000 800]);

            soundsc(test_signal, Fs); 
            pause(duration+1);
            soundsc(filtered_signal, Fs);

        case 2 % Real audio file
            [filename, pathname] = uigetfile({'*.wav;*.mp3;*.ogg;*.flac','Audio Files'});
            if filename == 0
                fprintf('No file selected.\n');
                return;
            end
            
            % Read audio file
            [y, Fs_audio] = audioread(fullfile(pathname, filename));
            
            % Check if sampling rates match
            if Fs_audio ~= Fs
                fprintf('Warning: Audio sample rate (%.0f Hz) doesn''t match filter rate (%.0f Hz)\n', Fs_audio, Fs);
                resample_choice = input('Resample audio to filter rate? (y/n): ', 's');
                if lower(resample_choice) == 'y'
                    y = resample(y, Fs, Fs_audio);
                    Fs_audio = Fs;
                end
            end
            
            % Apply filter (handle stereo/mono)
            if size(y,2) == 2 % Stereo
                filtered_audio = [filter(b, a, y(:,1)), filter(b, a, y(:,2))];
            else % Mono
                filtered_audio = filter(b, a, y);
            end
            
            % Normalize to prevent clipping
            filtered_audio = filtered_audio/max(abs(filtered_audio(:)));
            
            % Play and save options
            soundsc(filtered_audio, Fs_audio);
            
            save_choice = input('Save filtered audio? (y/n): ', 's');
            if lower(save_choice) == 'y'
                [~,name,ext] = fileparts(filename);
                newname = [name '_filtered' ext];
                audiowrite(newname, filtered_audio, Fs_audio);
                fprintf('Saved as %s\n', newname);
            end
            
        otherwise
            fprintf('Invalid choice.\n');
    end

    % Nested filter design function
    function [b, a] = AudioEQFilter(fc, dBgain, Fs, Q, type)
        % Convert inputs to radians
        omega = 2 * pi * fc / Fs;
        sin_omega = sin(omega);
        cos_omega = cos(omega);
        
        % Compute gain (linear scale for Peaking/Shelving filters)
        A = 10^(dBgain/40);  % amplitude
        
        % alpha factor
        alpha = sin_omega / (2 * Q);
        
        % Initialize coefficients
        b = [0 0 0];
        a = [0 0 0];

        % Select filter type
        switch lower(type)
            case 'lpf'
                b(1) = (1 - cos_omega)/2;
                b(2) = 1 - cos_omega;
                b(3) = (1 - cos_omega)/2;
                a(1) = 1 + alpha;
                a(2) = -2 * cos_omega;
                a(3) = 1 - alpha;
                
            case 'hpf'
                b(1) = (1 + cos_omega)/2;
                b(2) = -(1 + cos_omega);
                b(3) = (1 + cos_omega)/2;
                a(1) = 1 + alpha;
                a(2) = -2 * cos_omega;
                a(3) = 1 - alpha;
                
            case 'bpf'
                b(1) = sin_omega/2;
                b(2) = 0;
                b(3) = -sin_omega/2;
                a(1) = 1 + alpha;
                a(2) = -2 * cos_omega;
                a(3) = 1 - alpha;
                
            case 'bpfq'
                alpha = sin_omega * sinh( log(2)/2 * Q * omega/sin_omega );
                b(1) = sin_omega/2;
                b(2) = 0;
                b(3) = -sin_omega/2;
                a(1) = 1 + alpha;
                a(2) = -2 * cos_omega;
                a(3) = 1 - alpha;
                
            case 'notch'
                b(1) = 1;
                b(2) = -2 * cos_omega;
                b(3) = 1;
                a(1) = 1 + alpha;
                a(2) = -2 * cos_omega;
                a(3) = 1 - alpha;
                
            case 'apf'
                b(1) = 1 - alpha;
                b(2) = -2 * cos_omega;
                b(3) = 1 + alpha;
                a(1) = 1 + alpha;
                a(2) = -2 * cos_omega;
                a(3) = 1 - alpha;
                
            case 'peakingeq'
                b(1) = 1 + alpha*A;
                b(2) = -2 * cos_omega;
                b(3) = 1 - alpha*A;
                a(1) = 1 + alpha/A;
                a(2) = -2 * cos_omega;
                a(3) = 1 - alpha/A;
                
            case 'lowshelf'
                sqrtA = sqrt(A);
                b(1) = A*((A+1) - (A-1)*cos_omega + 2*sqrtA*alpha);
                b(2) = 2*A*((A-1) - (A+1)*cos_omega);
                b(3) = A*((A+1) - (A-1)*cos_omega - 2*sqrtA*alpha);
                a(1) = (A+1) + (A-1)*cos_omega + 2*sqrtA*alpha;
                a(2) = -2*((A-1) + (A+1)*cos_omega);
                a(3) = (A+1) + (A-1)*cos_omega - 2*sqrtA*alpha;
                
            case 'highshelf'
                sqrtA = sqrt(A);
                b(1) = A*((A+1) + (A-1)*cos_omega + 2*sqrtA*alpha);
                b(2) = -2*A*((A-1) + (A+1)*cos_omega);
                b(3) = A*((A+1) + (A-1)*cos_omega - 2*sqrtA*alpha);
                a(1) = (A+1) - (A-1)*cos_omega + 2*sqrtA*alpha;
                a(2) = 2*((A-1) - (A+1)*cos_omega);
                a(3) = (A+1) - (A-1)*cos_omega - 2*sqrtA*alpha;
                
            otherwise
                error('Unknown filter type');
        end
        
        % Normalize by a0
        b = b / a(1);
        a = a / a(1);
    end
end