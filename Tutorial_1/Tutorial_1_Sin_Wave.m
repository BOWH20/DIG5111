% Define the time vector (from 0 to 2*pi with 1000 points)
t = linspace(0, 2*pi, 1000);

% Calculate the sine wave
y = sin(t);

% Plot the sine wave
figure; % Create a new figure
plot(t, y, 'b-', 'LineWidth', 2); % Plot with a blue solid line and line width of 2

% Add labels and title
xlabel('Time (radians)'); % Label for the X-axis
ylabel('Amplitude'); % Label for the Y-axis
title('Sine Wave: y = sin(t)'); % Title of the plot

% Add grid for better visualization
grid on;