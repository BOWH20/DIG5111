% Define the decimal numbers from 1 to 30
decimalNumbers = (1:30)';

% Convert to binary and hexadecimal
binaryNumbers = arrayfun(@dec2bin, decimalNumbers, 'UniformOutput', false);
hexNumbers = arrayfun(@dec2hex, decimalNumbers, 'UniformOutput', false);

% Create and display the table
T = table(decimalNumbers, binaryNumbers, hexNumbers, ...
    'VariableNames', {'Decimal', 'Binary', 'Hexadecimal'});

disp(T);
