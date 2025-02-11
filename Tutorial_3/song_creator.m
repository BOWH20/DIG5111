notes = {'C' 'G' 'A' 'F'}; %notes which will be used
freq = [261.60 391.99 440.00 349.23]; %frequencies of notes above
melody = {'C' 'G' 'A' 'F' 'C' 'G' 'A' 'F'}; %four chords played twice
a = [];

%For Loop
for k = 1:numel(melody) %for loop which will create the melody
note = 0:0.00025:1.0; %note duration (which can be edited for length)
a = [a sin(2*pi*freq(strcmp(notes,melody{k}))*note)]; %a will create the melody given variables defined above
end

sound(a); % plays the melody
%End of code 