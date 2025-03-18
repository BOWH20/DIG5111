Fs = 2000;
f0 = 500;
Q = 
DBgain = 12;

A = sqrt(10^(DBgain/20));
w0 = 2*pi*f0/Fs;

cos(w0);
sin(w0);

alpha = sin(w0)/(2*Q);

b0 = (1 - cos(w0))/2;
b1 = 1 - cos(w0);
b2 = (1 - cos(w0))/2;
a0 = 1 + alpha;
a1 = -2*cos(w0);
a2 = 1 - alpha;