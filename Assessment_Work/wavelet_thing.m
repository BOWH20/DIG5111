[sig,fs] = audioread("Dr - I Keep Holding On (My Hope Will Never Die).mp3");
if size(sig, 2) > 1  % If stereo, take the mean (or just one channel)
    sig = mean(sig, 2);  % Average to mono (or use sig(:,1) for left channel)
end
t = (0:length(sig) - 1/fs);

duration = 5;  % seconds

max_samples = min(length(sig), duration * fs);
sig = sig(1:max_samples);

cwt(sig,fs);
helperPlotScalogram3d(sig,fs);