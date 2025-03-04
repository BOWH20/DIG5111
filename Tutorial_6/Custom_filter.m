
function output = my_conv(signal, IR)
 
    signal_length = length(signal);
    IR_length = length(IR);
    

    output_length = signal_length + IR_length - 1;
    

    output = zeros(1, output_length);
    
    
    for i = 1:signal_length
        for ii = 1:IR_length
            output(i + ii - 1) = output(i + ii - 1) + signal(i) * IR(ii);
        end
    end
end

NoiseFs = 22050;
NoiseDur = 2;
Noise = randn(NoiseFs * NoiseDur, 1); 

IR = lowpass.Numerator;
output = my_conv(Noise,IR);
sound(output,NoiseFs);

signalAnalyzer(output)