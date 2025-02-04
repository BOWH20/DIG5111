% Audio filename must be in the same directory as this m file
Filename = 'piano_middle_C.wav';

% Get file information
fileInfo = dir(Filename);

% Display file size in bytes
fileSizeBytes = fileInfo.bytes;
disp(['File size on disk: ', num2str(fileSizeBytes), ' bytes']);

% Convert bytes to KB, MB, or GB
fileSizeKB = fileSizeBytes / 1024; % Size in KB
fileSizeMB = fileSizeKB / 1024;    % Size in MB
fileSizeGB = fileSizeMB / 1024;    % Size in GB

disp(['File size on disk: ', num2str(fileSizeKB), ' KB']);
disp(['File size on disk: ', num2str(fileSizeMB), ' MB']);
disp(['File size on disk: ', num2str(fileSizeGB), ' GB']);
