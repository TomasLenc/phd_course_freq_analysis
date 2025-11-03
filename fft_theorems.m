% E-book with quite accessible resources on DFT theorems (and much more): 
% https://ccrma.stanford.edu/~jos/

% Math tools for neuroscience -> everything you ever wanted to know...
% http://www.cns.nyu.edu/~eero/math-tools19/


%% shift - multiplication

clear
addpath(genpath('lib'))

fs = 500; 
dur = 10; 
N = round(fs * dur); 
hN = floor(N / 2) + 1; 
t = [0 : N-1] / fs; 
freq = [0 : hN-1] / N * fs; 

col1 = [50, 168, 82]/255; 
col2 = [168, 144, 50]/255; 
col = [0, 0, 0]/255; 

x1 = sin(2 * pi * t * 100); 
x2 = 1 * sin(2 * pi * t * 3) + ...
     0.5 * sin(2 * pi * t * 11) + ...
     2 * cos(2 * pi * t * 0) ; 
x = x1 .* x2; 

mX1 = abs(fft(x1)); 
mX2 = abs(fft(x2)); 
mX = abs(fft(x)); 


f = figure('color', 'white', 'pos',[637 370 969 387]); 
pnl = panel(f); 

pnl.pack('h', 2); 
pnl(1).pack('v', 3); 
pnl(2).pack('v', 3); 

linew = 2; 

ax = pnl(1, 1).select(); 
plot(t, x1, 'color', col1, 'linew', linew); 
ax.XLim = [0, 1]; 
ax.XTick = []; 

ax = pnl(1, 2).select(); 
plot(t, x2, 'color', col2, 'linew', 3); 
ax.XLim = [0, 1]; 
ax.XTick = []; 

ax = pnl(1, 3).select(); 
plot(t, x, 'color', col, 'linew', linew); 
ax.XLim = [0, 1]; 


ax = pnl(2, 1).select(); 
stem(freq, mX1(1:hN), 'color', col1, 'linew', linew, 'marker', 'none'); 
ax.XLim = [0, fs/2]; 
ax.YTick = []; 
ax.XTick = []; 

ax = pnl(2, 2).select(); 
stem(freq, mX2(1:hN), 'color', col2, 'linew', linew, 'marker', 'none'); 
ax.XLim = [0, fs/2]; 
ax.YTick = []; 
ax.XTick = []; 

ax = pnl(2, 3).select(); 
stem(freq, mX(1:hN), 'color', col, 'linew', linew, 'marker', 'none'); 
ax.XLim = [0, fs/2]; 
ax.YTick = []; 

pnl.fontsize = 12; 
pnl(1).xlabel('time (s)');
pnl(2).xlabel('freq (Hz)');

pnl.de.margin = [10, 5, 5, 5]; 




%% convolution theorem

clear
addpath(genpath('lib'))
addpath(genpath('~/projects_git/rnb_tools/src')); 

fs = 100; 
dur = 0.2 * 12 * 4; 
N = round(fs * dur); 
hN = floor(N / 2) + 1; 
t = [0 : N-1] / fs; 
freq = [0 : hN-1] / N * fs; 

x = get_s([1,1,1,1,0,1,1,1,0,0,1,0], 0.2, fs, 'n_cycles', 4); 
mX = abs(fft(x)); 

ir = get_square_kernel(fs, 'duration', 0.100); 
t_ir = [0 : length(ir)-1] / fs; 

ir_pad = [ir, zeros(1, N-length(ir))]; 
mX_ir = abs(fft(ir_pad)); 

x_conv_ir = conv(x, ir, 'same'); 
mX_s_conv_ir = abs(fft(x_conv_ir)); 


col1 = [50, 168, 82]/255; 
col2 = [168, 144, 50]/255; 
col = [0, 0, 0]/255; 


f = figure('color', 'white', 'pos',[637 370 969 387]); 
pnl = panel(f); 

pnl.pack('h', 2); 
pnl(1).pack('v', 3); 
pnl(2).pack('v', 3); 

linew = 2; 

ax = pnl(1, 1).select(); 
plot(t, x, 'color', col1, 'linew', linew); 

pnl(1, 2).pack({[0, 0, 1, 1]}); 
pnl(1, 2).pack({[0.7, 0.5, 0.3, 0.6]}); 

ax = pnl(1, 2, 1).select(); 
plot(t, ir_pad, 'color', col2, 'linew', 3); 

ax = pnl(1, 2, 2).select(); 
plot(t_ir, ir, 'color', col2, 'linew', 3); 
ax.YTick = []; 

ax = pnl(1, 3).select(); 
plot(t, x_conv_ir, 'color', col, 'linew', linew); 


ax = pnl(2, 1).select(); 
stem(freq, mX(1:hN), 'color', col1, 'linew', linew, 'marker', 'none'); 
ax.XLim = [0, fs/2]; 
ax.YTick = []; 
ax.XTick = []; 

ax = pnl(2, 2).select(); 
stem(freq, mX_ir(1:hN), 'color', col2, 'linew', linew, 'marker', 'none'); 
ax.XLim = [0, fs/2]; 
ax.YTick = []; 
ax.XTick = []; 

ax = pnl(2, 3).select(); 
stem(freq, mX_s_conv_ir(1:hN), 'color', col, 'linew', linew, 'marker', 'none'); 
ax.XLim = [0, fs/2]; 
ax.YTick = []; 

pnl.fontsize = 12; 
pnl(1).xlabel('time (s)');
pnl(2).xlabel('freq (Hz)');

pnl.de.margin = [10,15, 5, 5]; 


% Can you now explain the multiplication-shift theorem using the convolution
% theorem? 


%% zero-padding - approaching continuous fourier in the limit

clear
addpath(genpath('lib'))
addpath(genpath('~/projects_git/rnb_tools/src')); 


fs = 500; 
dur = 0.5; 
N = round(fs * dur); 
hN = floor(N / 2) + 1; 
t = [0 : N-1] / fs; 
freq = [0 : hN-1] / N * fs; 

N_pad = N + N*5; 
hN_pad = floor(N_pad/2) + 1; 
t_pad = [0 : N_pad-1] / N_pad * fs; 
freq_pad = [0 : hN_pad-1] / N_pad * fs; 

col = [50, 168, 82]/255; 
col_pad = [168, 144, 50]/255; 

% Check what happens if the sine wave signal doesn't make an integer number of
% cycles within the N samples we have...
x = sin(2 * pi * t * 10.4); % try with e.g. 10 and then 10.4
mX = abs(fft(x)); 

x_pad = [x, zeros(1, N_pad -N)]; 
mX_pad = abs(fft(x_pad)); 


f = figure('color', 'white', 'pos',[637 370 969 387]); 
pnl = panel(f); 

pnl.pack('h', 2); 
pnl(1).pack('v', 2); 
pnl(2).pack('v', 2); 

linew = 2; 

ax = pnl(1, 1).select(); 
plot(t, x, 'color', col, 'linew', linew); 

ax = pnl(1, 2).select(); 
plot(t_pad, x_pad, 'color', col_pad, 'linew', 3); 

ax = pnl(2, 1).select(); 
stem(freq, mX(1:hN), 'color', col, 'linew', linew, 'marker', 'none'); 
ax.XLim = [0, fs/2]; 
ax.YTick = []; 
ax.XTick = []; 
ax.XLim = [0, 40]; 

ax = pnl(2, 2).select(); 
plot(freq_pad, mX_pad(1:hN_pad), 'color', col_pad, 'linew', linew); 
ax.XLim = [0, fs/2]; 
ax.YTick = []; 
ax.XTick = []; 
ax.XLim = [0, 40]; 

pnl.fontsize = 12; 
pnl(1).xlabel('time (s)');
pnl(2).xlabel('freq (Hz)');

pnl.de.margin = [10, 5, 5, 5]; 

% Discrete DFT samples the continuous Fourier transform function. The
% particular points where the continuous FT is sampled fall onto the "zeros"
% between sidelobes when the signal is ONLY composed of sine/cosine waves that
% complet an exactly integer number of cycles within the N samples of the
% signal. As soon as there are sines/cosines with frequencies that don't
% complete exactly an integer number of cycles, the continuous FT will be
% sampled at points where the sidelobes come above zero => we will have
% spectral leakage. 
% 
% This is why it's important to have an exact integer number of stimulation
% cycles when doing frequency tagging. 
% 
%
% Also, when performing time-frequency decompositions, we generally can't 
% know the exact frequencies of the sines/cosines making up the underlying signal. 
% In fact, it is safe to assume that there will be a big chunk of the recorded signal
% where the sine/cosine components will NOT have an integer number of cycles
% within out sampled duration => there WILL be spectral leakage for sure!!!
% 
% There are ways to minimize this leakage in a situation like this by first
% multiplying the signal with different windows (e.g. hann, hamming, blackman
% etc.). The idea is directly related to the convolution theorem (or
% multiplication-shift theorem) -> i.e. trying to find a window that has more 
% desirable spectrum than the implicit rectangular window. 



%% sampled finite signals as multiplication?  

% Using the theorems above, now we have tools to think about sampled finite signals
% (i.e. kinds of signals we normally deal with on our computers :)

% This is essentially what we're doing when taking finite sampled signal:
%     (1) Take an infinite continuous signal that repeates with period equal to
%         the duration of what will be our eventual sampled signal duration. 
%     (2) Multiply the continuous signal with a rectangular window 
%         (which is equal to 1 for one peiod of our eventual sampled signal 
%         duration, 0 otherwise. 
%     (3) Take continuous Fourier transform. 
%     (4) Sample the continuous Fourier transform at N points equivalently
%         spaced between 0 and fs/2 Hz. 


clear
addpath(genpath('lib'))

fs = 500; 

% We'll use sine wave with this frequency as an example signal. 
f = 10; 

% number of cycles of the sine wave we'll keep (after "windowing")
n_cycles = 5; 
% number of cycles before the window switches from 0 to 1
n_cycles_start = 15;
% number of cycles after the window switches from 1 back to 0
n_cycles_end = 15; 

% Note: Imagine what would happen if we increased the `n_cycles_start` and 
% `n_cycles_end` to a very large number? Given the example above with
% zero-padding, is this a bit more intuitive? 

% Duration of the original signal (that we're using to "pretend" we have an
% infinitely repeating continuous signal). 
dur = 1/f * (n_cycles_start + n_cycles + n_cycles_end);

N = round(dur * fs); 
hN = floor(N / 2) + 1; 
t = [0 : N-1] / fs; 
freq = [0 : hN-1] / N * fs; 

x = sin(2 * pi * t * f);

% Create the rectangular window 
win = zeros(1, N); 
% set it to 1 for `n_cycles` of the sinw wave signal frequency
idx = round(n_cycles_start * 1/f * fs); 
win(idx + 1 : idx + round(n_cycles * 1/f * fs)) = 1; 

% point-by-point multiply signal and rectangular window 
x_win = x .* win; 

% take DFT of each 
mX = abs(fft(x)); 
mX_win = abs(fft(win)); 
mX_x_win = abs(fft(x_win)); 


% plot
col_x = [50, 168, 82]/255; 
col_win = [168, 144, 50]/255; 
col = [0, 0, 0]/255; 

fig = figure('color', 'white', 'pos',[637 370 969 387]); 
pnl = panel(fig); 

pnl.pack('h', 2); 
pnl(1).pack('v', 3); 
pnl(2).pack('v', 3); 

linew = 2; 

ax = pnl(1, 1).select(); 
plot(t, x, 'color', col_x, 'linew', linew); 
ax.XTick = []; 

ax = pnl(1, 2).select(); 
plot(t, win, 'color', col_win, 'linew', 3); 
ax.XTick = []; 

ax = pnl(1, 3).select(); 
plot(t, x_win, 'color', col, 'linew', linew); 
ax.XTick = [0, dur]; 


maxfreqlim = 30; 

ax = pnl(2, 1).select(); 
plot(freq, mX(1:hN), '-o', 'color', col_x, 'linew', linew); 
ax.XLim = [0, maxfreqlim]; 
ax.YTick = []; 
ax.XTick = []; 

ax = pnl(2, 2).select(); 
plot(freq, mX_win(1:hN), '-o', 'color', col_win, 'linew', linew); 
ax.XLim = [0, maxfreqlim]; 
ax.YTick = []; 
ax.XTick = []; 

ax = pnl(2, 3).select(); 
plot(freq, mX_x_win(1:hN), '-o', 'color', col, 'linew', linew); 
ax.XLim = [0, maxfreqlim]; 
ax.YTick = []; 

pnl.fontsize = 12; 
pnl(1).xlabel('time (s)');
pnl(2).xlabel('freq (Hz)');

pnl.de.margin = [10, 5, 5, 5]; 






%% stretch - repeat

clear
addpath(genpath('lib'))
addpath(genpath('~/projects_git/rnb_tools/src')); 

fs = 100; 

col1 = [50, 168, 82]/255; 
col2 = [168, 144, 50]/255; 
col = [0, 0, 0]/255; 

nreps = [1 : 4]; 

x_unit = get_colored_noise(round(1 * fs), fs, -2); 

f = figure('color', 'white', 'pos',[637 370 969 387]); 
pnl = panel(f); 

pnl.pack('h', 2); 
pnl(1).pack('v', length(nreps)); 
pnl(2).pack('v', length(nreps)); 

linew = 2; 

max_dur = (max(nreps) * length(x_unit)) / fs; 

for i=1:length(nreps)
    
    nrep = nreps(i); 
    
    x = repmat(x_unit, 1, nrep); 
    mX = abs(fft(x)); 

    N = length(x); 
    hN = floor(N / 2) + 1; 
    t = [0 : N-1] / fs; 
    freq = [0 : hN-1] / N * fs; 

    ax = pnl(1, i).select(); 
    plot(t, x, 'color', col1, 'linew', linew); 
    ax.XLim = [0, max_dur]; 
    ax.XTick = [0, max_dur]; 
    ax.YTick = []; 

    ax = pnl(2, i).select(); 
    stem(freq, mX(1:hN), 'color', col1, 'linew', linew, 'marker', 'o'); 
    ax.XLim = [0, fs/10]; 
    ax.XTick = [0, fs/10]; 
    ax.YTick = []; 

end

pnl.fontsize = 12; 
pnl(1).xlabel('time (s)');
pnl(2).xlabel('freq (Hz)');

pnl.de.margin = [10, 8, 5, 5]; 








