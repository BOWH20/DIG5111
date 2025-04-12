function y = fft_conv(x, h)
    % Determine lengths of the input signal and impulse response
    N = length(x);
    M = length(h);
    % Calculate the full length required for a linear convolution
    L = N + M - 1;
    
    % Zero-pad both signals to length L to avoid circular convolution artifacts
    x_padded = [x; zeros(L - N, 1)];
    h_padded = [h; zeros(L - M, 1)];
    
    % Note: Removed the Hamming window multiplication to prevent imposing an
    % additional envelope on the impulse response which could cause artifacts.
    % If spectral leakage is a concern, consider alternative methods.
    
    % Compute the FFTs of the padded signals
    X = fft(x_padded);
    H = fft(h_padded);
    
    % Multiply in the frequency domain
    Y = X .* H;
    
    % Compute the symmetric inverse FFT to reduce numerical noise
    y = ifft(Y, 'symmetric');
    % Trim the output to the correct length
    y = y(1:L);
end
