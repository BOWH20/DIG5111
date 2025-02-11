function notex = note(A, keynum, dur)
%will create note with sampling frequency 11025
Fs = 11025;
Ts = 1/Fs;
A4 = 440;
ref_key = 49; %A4 is the 49th key
n = keynum - ref_key; % calculate the difference
freq = A4*2^(n/12);
Time = 0:Ts:dur;

notex = A*sin(2*pi*freq*Time);
end