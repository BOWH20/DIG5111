function helperPlotScalogram3d(sig, fs)
    % Ensure input is a vector
    if ~isvector(sig)
        error('Input signal must be a vector.');
    end
    
    % Convert to column vector if needed
    sig = sig(:);
    
    % Plot the 3D scalogram
    figure;
    [cfs, f] = cwt(sig, fs);  % Now sig is guaranteed to be a vector
    sigLen = numel(sig);
    t = (0:sigLen-1)/fs;
    
    surface(t, f, abs(cfs));
    xlabel("Time (s)");
    ylabel("Frequency (Hz)");
    zlabel("Magnitude (dB)");
    title("Scalogram In 3-D (Magnitude in dB)");
    
    % Set Y-axis to linear scale (normal numbers)
    set(gca, 'YScale', 'linear');
    
    shading interp;
    view([-40 30]);
    
    % Convert Z-axis to dB
    zTicks = get(gca, 'ZTick');
    zTickLabels = 20 * log10(zTicks + 1e-12);
    set(gca, 'ZTickLabel', num2str(zTickLabels', '%.1f'));
    
    % Enable interactive data cursor
    dcm = datacursormode(gcf);
    set(dcm, 'UpdateFcn', @dataCursorCallback);
    
    function output_txt = dataCursorCallback(~, event_obj)
        pos = get(event_obj, 'Position');
        magnitude = pos(3);
        db_value = 20 * log10(magnitude + 1e-12);
        output_txt = {['Time (s): ', num2str(pos(1))], ...
                     ['Frequency (Hz): ', num2str(pos(2))], ...
                     ['Magnitude (dB): ', num2str(db_value, '%.2f')]};
    end
end