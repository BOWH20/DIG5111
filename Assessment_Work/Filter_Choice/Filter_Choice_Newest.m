function interactiveAudioEQ()
    % This section starts the interactive tool and asks the user to choose
    % the filter topology (IIR or FIR) using a list dialog.
    %
    % listdlg: Opens a dialog for selecting from a list of strings.
    topoOptions = {'IIR', 'FIR'};  % Available filter topologies.
    [topoInd, ok] = listdlg('PromptString', 'Select Filter Topology:', ...
                            'SelectionMode', 'single', ...
                            'ListString', topoOptions);
    % Check if the user has made a valid selection.
    if ~ok || isempty(topoInd)
        % error: Stops execution and displays the message.
        error('No filter topology selected.');
    end
    % Save the selected topology as a string.
    topology = topoOptions{topoInd};

    %% === IIR Design Type Selection (if topology is IIR) ===
    % For IIR filters, we allow choosing between different design methods.
    if strcmp(topology, 'IIR')
        % Define the available IIR design options.
        iirOptions = {'Butterworth', 'Chebyshev', 'Custom (Manual Q and additional forms)'};
        % Again, use listdlg to let the user choose.
        [iirInd, ok] = listdlg('PromptString', 'Select IIR Design Type:', ...
                               'SelectionMode', 'single', ...
                               'ListString', iirOptions);
        if ~ok || isempty(iirInd)
            error('No IIR design type selected.');
        end
        % Map the selected index to a specific design type.
        switch iirInd
            case 1, iir_design = 'Butterworth';
            case 2, iir_design = 'Chebyshev';
            case 3, iir_design = 'Custom';
        end
    else
        % For FIR filters, we don’t need an IIR design type.
        iir_design = '';
    end

    %% === Filter Form Selection ===
    % This part defines what kind of filter (lowpass, highpass, bandpass, etc.)
    % the user wants to design. There are separate options for IIR and FIR.
    if strcmp(topology, 'IIR')
        % If the IIR design is Butterworth or Chebyshev (standard forms):
        if strcmp(iir_design, 'Butterworth') || strcmp(iir_design, 'Chebyshev')
            formLabels = { 'Low Pass (lp)', 'High Pass (hp)', 'Band Pass (bp)', 'Band Stop (bs)' };
            formCodes  = { 'lp', 'hp', 'bp', 'bs' };
        else
            % For Custom IIR designs, more filter forms are provided.
            formLabels = { 'Low Pass (lpf)', 'High Pass (hpf)', 'Band Pass (bpf)', ...
                           'Band Pass (Constant Q) (bpfq)', 'Band Stop / Notch (bs/notch)', ...
                           'All Pass (apf)', 'Peaking EQ (peakingeq)', 'Low Shelf (lowshelf)', ...
                           'High Shelf (highshelf)' };
            formCodes  = { 'lpf', 'hpf', 'bpf', 'bpfq', 'bs/notch', 'apf', 'peakingeq', 'lowshelf', 'highshelf' };
        end
    else  % For FIR filters, the options are limited.
        formLabels = { 'Low Pass (lp)', 'High Pass (hp)', 'Band Pass (bp)', 'Band Stop (bs)' };
        formCodes  = { 'lp', 'hp', 'bp', 'bs' };
    end
    % Open another list dialog for the filter form.
    [formInd, ok] = listdlg('PromptString', 'Select Filter Form:', ...
                            'SelectionMode', 'single', ...
                            'ListString', formLabels);
    if ~ok || isempty(formInd)
        error('No filter form selected.');
    end
    % Get the corresponding code for the selected filter form.
    form = strtrim(formCodes{formInd});

    %% === Filter Order Entry ===
    % The filter order (or filter length for FIR) is taken via an input dialog.
    % inputdlg: Prompts the user to enter text input.
    validOrder = false;
    while ~validOrder
        orderStr = inputdlg('Enter filter order (for IIR, this is the order; for FIR, it is filter length-1):', ...
                            'Filter Order', [1 50], {'4'});
        if isempty(orderStr)
            error('Filter order cancelled.');
        end
        % Convert the string input to a numeric value.
        order = str2double(orderStr{1});
        % Check if the order is a positive integer.
        if ~isnan(order) && order > 0 && mod(order, 1) == 0
            validOrder = true;
        else
            % msgbox and uiwait: Display an error message and wait for user acknowledgment.
            uiwait(msgbox('Order must be a positive integer. Please try again.', 'Error', 'error'));
        end
    end

    %% === Cutoff Frequency Entry ===
    % This section collects the cutoff frequency/frequencies.
    % For filters like bandpass and bandstop, two cutoff frequencies are required.
    if any(strcmpi(form, {'bp','bpf','bs','bpfq'}))
        validCutoff = false;
        while ~validCutoff
            % Ask for both lower and upper cutoff frequencies.
            prompt = {'Enter lower cutoff frequency (Hz):', 'Enter upper cutoff frequency (Hz):'};
            dlgTitle = 'Cutoff Frequencies';
            defInput = {'100', '500'};
            answer = inputdlg(prompt, dlgTitle, [1 50], defInput);
            if isempty(answer)
                error('Cutoff frequency entry cancelled.');
            end
            % Convert inputs to numbers.
            fc_low = str2double(answer{1});
            fc_high = str2double(answer{2});
            % Validate: lower > 0 and upper is greater than lower.
            if ~isnan(fc_low) && ~isnan(fc_high) && fc_low > 0 && fc_high > fc_low
                validCutoff = true;
            else
                uiwait(msgbox('Invalid cutoff frequencies. Ensure lower > 0 and upper > lower.', 'Error', 'error'));
            end
        end
        % Calculate the center frequency used in some filter designs.
        fc = (fc_low + fc_high) / 2;
    else
        % For filters that need only one cutoff.
        validCutoff = false;
        while ~validCutoff
            answer = inputdlg('Enter cutoff frequency (Hz):', 'Cutoff Frequency', [1 50], {'300'});
            if isempty(answer)
                error('Cutoff frequency entry cancelled.');
            end
            fc = str2double(answer{1});
            if ~isnan(fc) && fc > 0
                validCutoff = true;
            else
                uiwait(msgbox('Frequency must be positive. Please try again.', 'Error', 'error'));
            end
        end
        % Set lower and upper cutoff empty since they are not used in one-frequency filters.
        fc_low = []; fc_high = [];
    end

    %% === Gain Entry (if required) ===
    % Some filter forms (like peaking equalizers or shelving filters) need a gain value.
    if any(strcmpi(form, {'peakingeq','lowshelf','highshelf'}))
        validGain = false;
        while ~validGain
            answer = inputdlg('Enter gain in dB (e.g., 6 for boost, -6 for cut):', 'Gain', [1 50], {'6'});
            if isempty(answer)
                error('Gain entry cancelled.');
            end
            dBgain = str2double(answer{1});
            if ~isnan(dBgain)
                validGain = true;
            else
                uiwait(msgbox('Gain must be numeric. Please try again.', 'Error', 'error'));
            end
        end
    else
        % No gain is needed; set to zero.
        dBgain = 0;
    end

    %% === Sampling Rate Entry ===
    % Get the sampling rate (Fs) with which the filter will be used.
    validFs = false;
    while ~validFs
        answer = inputdlg('Enter sampling rate (Hz):', 'Sampling Rate', [1 50], {'44100'});
        if isempty(answer)
            error('Sampling rate entry cancelled.');
        end
        Fs = str2double(answer{1});
        % For filters that use two cutoffs, use the upper cutoff for validation.
        if any(strcmpi(form, {'bp','bpf','bs','bpfq'}))
            if ~isnan(Fs) && Fs > 2 * fc_high  % Nyquist rule: Fs must be > 2 * highest frequency
                validFs = true;
            else
                uiwait(msgbox('Sampling rate must be greater than twice the upper cutoff frequency. Please try again.', 'Error', 'error'));
            end
        else
            if ~isnan(Fs) && Fs > 2 * fc  % For single cutoff, Fs must be > 2 * cutoff frequency.
                validFs = true;
            else
                uiwait(msgbox('Sampling rate must be greater than twice the cutoff frequency. Please try again.', 'Error', 'error'));
            end
        end
    end

    %% === Q Factor Selection (for Custom IIR designs) ===
    % When using a custom IIR design, we may allow the user to set a Q factor
    % manually or choose a preset Butterworth Q factor.
    if strcmp(topology, 'IIR') && strcmp(iir_design, 'Custom') && ...
       any(strcmpi(form, {'lpf','hpf','bpf','bpfq','notch','bs','apf','peakingeq','lowshelf','highshelf'}))
        qOptions = {'Manual Entry','Butterworth (preset)'};
        [qInd, ok] = listdlg('PromptString', 'Select Q Factor Selection Method:', ...
                             'SelectionMode', 'single', ...
                             'ListString', qOptions);
        if ~ok || isempty(qInd)
            error('No Q factor selection method chosen.');
        end
        if qInd == 1
            % Let the user input the Q factor.
            validQ = false;
            while ~validQ
                answer = inputdlg('Enter Q factor (e.g., 0.707):', 'Q Factor', [1 50], {'0.707'});
                if isempty(answer)
                    error('Q factor entry cancelled.');
                end
                Q = str2double(answer{1});
                if ~isnan(Q) && Q > 0
                    validQ = true;
                else
                    uiwait(msgbox('Q factor must be positive. Please try again.', 'Error', 'error'));
                end
            end
            useButterworth = false;
        else
            % Use the Butterworth preset value (typical value ~0.707 for 2nd order).
            useButterworth = true;
            Q = 0.7071;
        end
    else
        % If not using custom IIR design, no Q is needed.
        Q = [];
        useButterworth = false;
    end

    %% === Filter Design Based on User Inputs ===
    % Depending on the selected topology, we call either the IIR or FIR design routine.
    if strcmp(topology, 'IIR')
        [b, a] = designIIR(form, order, Fs, fc, fc_low, fc_high, dBgain, Q, useButterworth, iir_design);
    else
        [b, a] = designFIR(form, order, Fs, fc, fc_low, fc_high);
    end

    %% === Display the Filter Coefficients ===
    % Print each section’s coefficients to the command window.
    fprintf('\nFilter coefficients:\n');
    for i = 1:length(b)
        fprintf('Section %d:\n', i);
        fprintf('Numerator (b): [ ');
        fprintf('%f ', b{i});
        fprintf(']\n');
        fprintf('Denominator (a): [ ');
        fprintf('%f ', a{i});
        fprintf(']\n\n');
    end

    %% === Combine Filter Sections ===
    % For filters designed in multiple sections, convolve coefficients.
    % conv: Convolution multiplies polynomial coefficients (combines sections).
    overall_b = b{1};
    overall_a = a{1};
    for i = 2:length(b)
        overall_b = conv(overall_b, b{i});
        overall_a = conv(overall_a, a{i});
    end

    %% === Compute Frequency Response ===
    % freqz: Computes the frequency response (H) and frequencies (w) of a digital filter.
    [H, w] = freqz(overall_b, overall_a, 1024, Fs);
    % Convert magnitude response to dB.
    magnitudeResponse = 20*log10(abs(H));
    % Unwrap phase response to remove discontinuities, then convert to degrees.
    phaseResponse     = unwrap(angle(H))*(180/pi);

    %% === Plot Frequency Response ===
    % Create a figure with two subplots: one for magnitude, one for phase.
    figure;
    % Magnitude Response Plot
    subplot(2,1,1);
    plot(w, magnitudeResponse);
    title('Magnitude Response');
    xlabel('Frequency (Hz)');
    ylabel('Magnitude (dB)');
    grid on;
    ax1 = gca;        % Get current axes handle.
    ax1.XAxis.Exponent = 0;  % Turn off exponential notation.
    xtickformat(ax1, '%.0f');  % Format tick labels as integers.
    
    % Phase Response Plot
    subplot(2,1,2);
    plot(w, phaseResponse);
    title('Phase Response');
    xlabel('Frequency (Hz)');
    ylabel('Phase (degrees)');
    grid on;
    ax2 = gca;
    ax2.XAxis.Exponent = 0;
    xtickformat(ax2, '%.0f');

    %% === Save Filter Coefficients to Base Workspace ===
    % assignin: Places the given variables into the base workspace.
    assignin('base', 'b', b);
    assignin('base', 'a', a);
    fprintf('Filter coefficients saved to workspace as cell arrays ''b'' and ''a''.\n');

    %% === Audio Processing Section ===
    % Let the user choose to test the filter on a synthetic test signal or on an audio file.
    audioOptions = {'Apply to test signal', 'Apply to audio file'};
    [audioInd, ok] = listdlg('PromptString','Select Audio Processing Option:', ...
                             'SelectionMode','single', ...
                             'ListString', audioOptions);
    if ~ok || isempty(audioInd)
        error('No audio processing option selected.');
    end
    choice = audioInd;
    
    switch choice
        case 1  % === Test Signal Processing ===
            % Create a random test signal for demonstration.
            duration = 3;  % duration in seconds
            t = 0:1/Fs:duration;
            test_signal = randn(size(t));  % Generate white noise.
            % Filter the test signal using the designed filter.
            filtered_signal = filter(overall_b, overall_a, test_signal);
            figure;
            subplot(2,1,1);
            plot(t, test_signal);
            title('Original Signal');
            xlabel('Time (s)'); ylabel('Amplitude'); grid on;
            subplot(2,1,2);
            plot(t, filtered_signal);
            title('Filtered Signal');
            xlabel('Time (s)'); ylabel('Amplitude'); grid on;
            % soundsc: Plays the sound while automatically scaling the signal.
            soundsc(test_signal, Fs);
            pause(duration+1);  % Wait until playback is finished.
            soundsc(filtered_signal, Fs);
            
        case 2  % === Audio File Processing ===
            % Open a file dialog to let the user select an audio file.
            [filename, pathname] = uigetfile({'*.wav;*.mp3;*.ogg;*.flac','Audio Files'});
            if isequal(filename,0)
                fprintf('No file selected.\n');
                return;
            end
            % Read the selected audio file.
            [y, Fs_audio] = audioread(fullfile(pathname, filename));
            % If the sample rates do not match, ask the user whether to resample.
            if Fs_audio ~= Fs
                choiceStr = questdlg(sprintf('Audio sample rate (%.0f Hz) does not match filter rate (%.0f Hz). Resample audio to filter rate?', Fs_audio, Fs), ...
                                       'Sample Rate Mismatch', ...
                                       'Yes','No','Yes');
                if strcmpi(choiceStr, 'Yes')
                    % resample: Changes the sampling rate of a signal.
                    y = resample(y, Fs, Fs_audio);
                    Fs_audio = Fs;
                end
            end
            % Process stereo or mono audio.
            if size(y,2)==2
                filtered_audio = y;
                % Apply filter to each channel separately.
                filtered_audio(:,1) = filter(overall_b, overall_a, y(:,1));
                filtered_audio(:,2) = filter(overall_b, overall_a, y(:,2));
            else
                filtered_audio = filter(overall_b, overall_a, y);
            end
            % Normalize the filtered audio so that it does not clip.
            filtered_audio = filtered_audio / max(abs(filtered_audio(:)));
            soundsc(filtered_audio, Fs_audio);
            % Ask user if they want to save the filtered audio.
            save_choice = questdlg('Save filtered audio?', 'Save Audio', 'Yes','No','No');
            if strcmpi(save_choice, 'Yes')
                [~, name, ext] = fileparts(filename);
                newname = fullfile(pathname, [name '_filtered' ext]);
                audiowrite(newname, filtered_audio, Fs_audio);
                fprintf('Filtered audio saved as %s\n', newname);
            end
        otherwise
            error('Invalid audio processing option.');
    end

    %% === Nested Function: IIR Filter Design ===
    % Designs an IIR filter based on user input.
    % Uses built-in design methods (Butterworth, Chebyshev) if applicable,
    % or a custom design using biquad formulas.
    function [b, a] = designIIR(form, order, Fs, fc, fc_low, fc_high, dBgain, Q, useButterworth, design_method)
        % Standard designs using built-in functions for common filter forms.
        if strcmpi(design_method, 'Butterworth') || strcmpi(design_method, 'Chebyshev')
            if any(strcmpi(form, {'lp','hp','bp','bs'}))
                if strcmpi(form, 'lp')
                    Wn = fc/(Fs/2);  % Normalize frequency (dividing by Nyquist frequency)
                    if strcmpi(design_method, 'Butterworth')
                        % butter: Designs a lowpass Butterworth filter.
                        [b_overall, a_overall] = butter(order, Wn, 'low');
                    else
                        % cheby1: Designs a lowpass Chebyshev Type I filter.
                        rp = str2double(inputdlg('Enter the passband ripple in dB (e.g., 1):', 'Passband Ripple', [1 50], {'1'}));
                        [b_overall, a_overall] = cheby1(order, rp, Wn, 'low');
                    end
                elseif strcmpi(form, 'hp')
                    Wn = fc/(Fs/2);
                    if strcmpi(design_method, 'Butterworth')
                        [b_overall, a_overall] = butter(order, Wn, 'high');
                    else
                        rp = str2double(inputdlg('Enter the passband ripple in dB (e.g., 1):', 'Passband Ripple', [1 50], {'1'}));
                        [b_overall, a_overall] = cheby1(order, rp, Wn, 'high');
                    end
                elseif strcmpi(form, 'bp')
                    Wn = [fc_low fc_high] / (Fs/2);
                    if strcmpi(design_method, 'Butterworth')
                        [b_overall, a_overall] = butter(order, Wn, 'bandpass');
                    else
                        rp = str2double(inputdlg('Enter the passband ripple in dB (e.g., 1):', 'Passband Ripple', [1 50], {'1'}));
                        [b_overall, a_overall] = cheby1(order, rp, Wn, 'bandpass');
                    end
                elseif strcmpi(form, 'bs')
                    Wn = [fc_low fc_high] / (Fs/2);
                    if strcmpi(design_method, 'Butterworth')
                        [b_overall, a_overall] = butter(order, Wn, 'stop');
                    else
                        rp = str2double(inputdlg('Enter the passband ripple in dB (e.g., 1):', 'Passband Ripple', [1 50], {'1'}));
                        [b_overall, a_overall] = cheby1(order, rp, Wn, 'stop');
                    end
                end
                % tf2sos: Converts a transfer function to second-order sections.
                sos = tf2sos(b_overall, a_overall);
                num_sections = size(sos, 1);
                % Initialize cell arrays to hold coefficients for each section.
                b = cell(1, num_sections);
                a = cell(1, num_sections);
                for k = 1:num_sections
                    b{k} = sos(k,1:3);
                    a{k} = sos(k,4:6);
                end
                % Finished designing standard IIR filter.
                return;
            else
                % If we reach here, revert to custom design.
                design_method = 'Custom';
            end
        end
        
        % === Custom IIR Filter Design using Biquad Formulas ===
        % Determine how many biquad sections are needed.
        num_sections = ceil(order/2);
        b = cell(1, num_sections);
        a = cell(1, num_sections);
        % For bandpass filters, calculate the center frequency and bandwidth.
        if any(strcmpi(form, {'bpf','bpfq'}))
            center = (fc_low + fc_high)/2;
            bw = fc_high - fc_low;
            Q_custom = center / bw;
        end
        % If Butterworth preset Q is to be used for multiple sections.
        if useButterworth && num_sections > 1
            Qs = zeros(1, num_sections);
            for k = 1:num_sections
                theta_k = pi/(2*order) + (k-1)*pi/order;
                Qs(k) = 1/(2*sin(theta_k));
            end
        else
            Qs = repmat(Q, 1, num_sections);
        end
        
        % Loop through each biquad section to calculate its coefficients.
        for section = 1:num_sections
            if any(strcmpi(form, {'bpf','bpfq'}))
                omega = 2*pi*((fc_low+fc_high)/2)/Fs;
                sin_omega = sin(omega);
                cos_omega = cos(omega);
                Q_used = Q_custom;
            else
                omega = 2*pi*fc/Fs;
                sin_omega = sin(omega);
                cos_omega = cos(omega);
                Q_used = Qs(section);
            end
            A = 10^(dBgain/40);  % Convert gain in dB into linear scale.
            alpha = sin_omega/(2*Q_used);
            % Switch based on filter form to calculate coefficients.
            switch lower(form)
                case 'lpf'
                    b_section = [(1-cos_omega)/2, 1-cos_omega, (1-cos_omega)/2];
                    a_section = [1+alpha, -2*cos_omega, 1-alpha];
                case 'hpf'
                    b_section = [(1+cos_omega)/2, -(1+cos_omega), (1+cos_omega)/2];
                    a_section = [1+alpha, -2*cos_omega, 1-alpha];
                case 'bpf'
                    b_section = [sin_omega/2, 0, -sin_omega/2];
                    a_section = [1+alpha, -2*cos_omega, 1-alpha];
                case 'bpfq'
                    % Adjust alpha for constant Q.
                    alpha = sin_omega * sinh(log(2)/2 * Q_used * omega/sin_omega);
                    b_section = [sin_omega/2, 0, -sin_omega/2];
                    a_section = [1+alpha, -2*cos_omega, 1-alpha];
                case {'bs','notch'}
                    b_section = [1, -2*cos_omega, 1];
                    a_section = [1+alpha, -2*cos_omega, 1-alpha];
                case 'apf'
                    b_section = [1-alpha, -2*cos_omega, 1+alpha];
                    a_section = [1+alpha, -2*cos_omega, 1-alpha];
                case 'peakingeq'
                    b_section = [1+alpha*A, -2*cos_omega, 1-alpha*A];
                    a_section = [1+alpha/A, -2*cos_omega, 1-alpha/A];
                case 'lowshelf'
                    sqrtA = sqrt(A);
                    b_section = [ A*((A+1)-(A-1)*cos_omega+2*sqrtA*alpha), ...
                                  2*A*((A-1)-(A+1)*cos_omega), ...
                                  A*((A+1)-(A-1)*cos_omega-2*sqrtA*alpha)];
                    a_section = [ (A+1)+(A-1)*cos_omega+2*sqrtA*alpha, ...
                                 -2*((A-1)+(A+1)*cos_omega), ...
                                  (A+1)-(A-1)*cos_omega-2*sqrtA*alpha];
                case 'highshelf'
                    sqrtA = sqrt(A);
                    b_section = [ A*((A+1)+(A-1)*cos_omega+2*sqrtA*alpha), ...
                                 -2*A*((A-1)+(A+1)*cos_omega), ...
                                  A*((A+1)+(A-1)*cos_omega-2*sqrtA*alpha)];
                    a_section = [ (A+1)-(A-1)*cos_omega+2*sqrtA*alpha, ...
                                  2*((A-1)-(A+1)*cos_omega), ...
                                  (A+1)-(A-1)*cos_omega-2*sqrtA*alpha];
                otherwise
                    error('Unknown filter form for custom design.');
            end
            % Normalize the section coefficients so that the first denominator coefficient equals 1.
            b_section = b_section / a_section(1);
            a_section = a_section / a_section(1);
            b{section} = b_section;
            a{section} = a_section;
        end
        % If the filter order is odd, add an extra first-order section.
        if mod(order, 2) == 1
            omega = 2*pi*fc/Fs;
            switch lower(form)
                case {'lpf','peakingeq','lowshelf'}
                    b_first = [1 - exp(-omega), 0];
                    a_first = [1, -exp(-omega)];
                case {'hpf','highshelf'}
                    b_first = [1 - exp(-omega), 0];
                    a_first = [1, -exp(-omega)];
                otherwise
                    b_first = b{end};
                    a_first = a{end};
            end
            b{end+1} = b_first;
            a{end+1} = a_first;
        end
    end

    %% === Nested Function: FIR Filter Design ===
    % Designs an FIR filter using the fir1 function.
    % fir1: Designs an FIR filter using a window-based method.
    function [b, a] = designFIR(form, order, Fs, fc, fc_low, fc_high)
        if strcmpi(form, 'lp')
            Wn = fc/(Fs/2);
            b_overall = fir1(order, Wn, 'low');
        elseif strcmpi(form, 'hp')
            Wn = fc/(Fs/2);
            b_overall = fir1(order, Wn, 'high');
        elseif strcmpi(form, 'bp')
            Wn = [fc_low fc_high] / (Fs/2);
            b_overall = fir1(order, Wn, 'bandpass');
        elseif strcmpi(form, 'bs')
            Wn = [fc_low fc_high] / (Fs/2);
            b_overall = fir1(order, Wn, 'stop');
        else
            error('Unknown FIR filter form.');
        end
        % For FIR filters, the denominator is always 1.
        b = {b_overall};
        a = {1};
    end
end
