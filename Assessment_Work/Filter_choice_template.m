function interactiveAudioEQ()
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
    
    % Design the filter using the nested function
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

    % Nested function definition
    function [b, a] = AudioEQFilter(fc, dBgain, Fs, Q, type)
        % Convert inputs to radians
        omega = 2 * pi * fc / Fs;
        sin_omega = sin(omega);
        cos_omega = cos(omega);
        
        % Compute gain (linear scale for Peaking/Shelving filters)
        A = 10^(dBgain/40);  % amplitude
        
        % alpha factor
        alpha = sin_omega / (2 * Q);
        
        % For shelf filters
        beta = sqrt(A) / Q;

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
                alpha = sin_omega * sinh( log(2)/2 * Q * omega/sin_omega ); % Constant Q
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
                b(1) =    A*( (A+1) - (A-1)*cos_omega + 2*sqrtA*alpha );
                b(2) =  2*A*( (A-1) - (A+1)*cos_omega );
                b(3) =    A*( (A+1) - (A-1)*cos_omega - 2*sqrtA*alpha );
                a(1) =        (A+1) + (A-1)*cos_omega + 2*sqrtA*alpha;
                a(2) =   -2*( (A-1) + (A+1)*cos_omega );
                a(3) =        (A+1) + (A-1)*cos_omega - 2*sqrtA*alpha;
                
            case 'highshelf'
                sqrtA = sqrt(A);
                b(1) =    A*( (A+1) + (A-1)*cos_omega + 2*sqrtA*alpha );
                b(2) = -2*A*( (A-1) + (A+1)*cos_omega );
                b(3) =    A*( (A+1) + (A-1)*cos_omega - 2*sqrtA*alpha );
                a(1) =        (A+1) - (A-1)*cos_omega + 2*sqrtA*alpha;
                a(2) =    2*( (A-1) - (A+1)*cos_omega );
                a(3) =        (A+1) - (A-1)*cos_omega - 2*sqrtA*alpha;
                
            otherwise
                error('Unknown filter type');
        end
        
        % Normalize by a0
        b = b / a(1);
        a = a / a(1);
    end
end