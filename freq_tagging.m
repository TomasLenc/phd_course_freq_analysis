
% Okay so we've been talking about sine waves, and how a weighted
% combination of sine waves can be used to perfectly reconstruct any
% signal. 

% Let's see the DFTs of some example signals. This will be useful for
% intuition. 


clear 
addpath(genpath('lib'))
addpath(genpath('~/projects_git/rnb_tools/src')); 
addpath(genpath('~/projects_git/acf_tools/src')); 


%% parameters 

% sampling rate (Hz)
fs = 50; 

% number of repetitions 
n_rep = 16; 

% period 
period = 1; 

% total signal duration (s)
dur = period * n_rep; 
N = round(dur * fs); 
t = [0 : N-1] / fs; 
freq = [0 : N/2-1] / N * fs; 


%% 

f = figure('color', 'w', 'pos', [221 388 1380 603]); 
pnl = panel(f); 
pnl.pack('h', [50,50]); 
pnl(1).pack('v', 3); 
pnl(2).pack('v', 3); 


% sine wave
% ---------

x = sin(2*pi*1/period*t); 

% get magnitude spectrum
mX = abs(fft(x)); 
mX = mX(1:N/2); 

ax = pnl(1,1).select(); 
plot(t, x, 'color', 'k', 'linew', 2)
xlim([0, dur/2]); 
xticks([0,dur/2])
yticks([])

ax = pnl(2,1).select(); 
plot_fft(freq, mX, 'ax', ax, 'maxfreqlim', freq(end), 'linew', 3)
yticks([])
xlim([0, fs/2])
xticks([1/period : 2/period : fs/2])
ax.XAxis.Visible = 'on'; 



% square wave 
% -----------

% kernel 
ir = get_square_kernel(fs, 'duration', 0.2); 

% make full signal 
s = zeros(1, N); 
idx = round([0 : n_rep-1] * period * fs) + 1; 
x = insert_events(s, idx, ir); 

% get magnitude spectrum
mX = abs(fft(x)); 
mX = mX(1:N/2); 

ax = pnl(1,2).select(); 
plot(t, x, 'color', 'k', 'linew', 2)
xlim([0, dur/2]); 
xticks([0,dur/2])
yticks([])

ax = pnl(2,2).select(); 
plot_fft(freq, mX, 'ax', ax, 'maxfreqlim', freq(end), 'linew', 3)
yticks([])
xlim([0, fs/2])
xticks([1/period : 2/period : fs/2])
ax.XAxis.Visible = 'on'; 


% erp
% ---

% kernel
ir = get_erp_kernel(fs, ...
    'amplitudes', [0.2, 0.55],...
    't0s', [0, 0], ...
    'taus', [0.2, 0.050], ...
    'f0s', [1, 7], ...
    'duration', 0.98*period ...
    ); 

% make full signal 
x = zeros(1, N); 
idx = round([0 : n_rep-1] * period * fs) + 1; 
x = insert_events(x, idx, ir); 

% get magnitude spectrum
mX = abs(fft(x)); 
mX = mX(1:N/2); 

ax = pnl(1,3).select(); 
plot(t, x, 'color', 'k', 'linew', 2)
xlim([0, dur/2]); 
xticks([0,dur/2])
yticks([])

ax = pnl(2,3).select(); 
plot_fft(freq, mX, 'ax', ax, 'maxfreqlim', freq(end), 'linew', 3)
yticks([])
xlim([0, fs/2])
xticks([1/period : 2/period : fs/2])
ax.XAxis.Visible = 'on'; 


% format figure 
pnl(1).xlabel('time (s)')
pnl(2).xlabel('frequency (Hz)'); 

pnl(1,1).ylabel('amplitude')
pnl(1,2).ylabel('amplitude')
pnl(1,3).ylabel('amplitude')

pnl(2,1).ylabel('magnitude')
pnl(2,2).ylabel('magnitude')
pnl(2,3).ylabel('magnitude')

pnl.margin = [20, 20, 5, 5]; 
pnl.fontsize = 24; 
    


saveas(f, 'figures/fft_ex_resp_shapes.svg')




%% response and noise 


f = figure('color', 'w', 'pos', [221 388 1380 603]); 
pnl = panel(f); 
pnl.pack('h', [50,50]); 
pnl(1).pack('v', 3); 
pnl(2).pack('v', 3); 

% x = rand(1, 2*N); 
% [b,a] = butter(2, 10/(fs/2), 'low'); 
% x = filtfilt(b,a,x); 
% x = x(N/2+1 : end-N/2); 

% x = randn(1, N); 

noise = get_colored_noise(N, fs, -1.5); 


% get magnitude spectrum
mX = abs(fft(noise)); 
mX = mX(1:N/2); 
mX(1) = 0; 


ax = pnl(1,1).select(); 
plot(t, noise, 'color', 'k', 'linew', 1.5)
xlim([0, dur]); 
xticks([0,dur])
yticks([])
ylim([min(noise), max(noise)])

ax = pnl(2,1).select(); 
plot_fft(freq, mX, 'ax', ax, 'maxfreqlim', freq(end), 'linew', 3)
yticks([])
xlim([0, fs/2])
xticks([1/period : 2/period : fs/2])
ax.XAxis.Visible = 'on'; 


% make periodic response 
ir = get_square_kernel(fs, 'duration', 0.2); 

x = zeros(1, N); 
idx = round([0 : n_rep-1] * period * fs) + 1; 
x = insert_events(x, idx, ir); 

mX = abs(fft(x)); 
mX = mX(1:N/2); 
mX(1) = 0; 

ax = pnl(1,2).select(); 
plot(t, x, 'color', 'k', 'linew', 1.5)
xlim([0, dur]); 
xticks([0,dur])
yticks([])
ylim([min(x), max(x)])

ax = pnl(2,2).select(); 
plot_fft(freq, mX, 'ax', ax, 'maxfreqlim', freq(end), 'linew', 3, ...
    'frex_meter_rel', [1/period : 1/period : fs/2])
yticks([])
xlim([0, fs/2])
xticks([1/period : 2/period : fs/2])
ax.XAxis.Visible = 'on'; 


% add signal + noise 
x = add_signal_noise(x, noise, 0.6); 

mX = abs(fft(x)); 
mX = mX(1:N/2); 
mX(1) = 0; 

ax = pnl(1,3).select(); 
plot(t, x, 'color', 'k', 'linew', 1.5)
xlim([0, dur]); 
xticks([0,dur])
yticks([])
ylim([min(x), max(x)])

ax = pnl(2,3).select(); 
plot_fft(freq, mX, 'ax', ax, 'maxfreqlim', freq(end), 'linew', 3, ...
    'frex_meter_rel', [1/period : 1/period : fs/2])
yticks([])
xlim([0, fs/2])
xticks([1/period : 2/period : fs/2])
ax.XAxis.Visible = 'on'; 


% format figure 
pnl(1).xlabel('time (s)')
pnl(2).xlabel('frequency (Hz)'); 

pnl(1,1).ylabel('amplitude')
pnl(1,2).ylabel('amplitude')
pnl(1,3).ylabel('amplitude')

pnl(2,1).ylabel('magnitude')
pnl(2,2).ylabel('magnitude')
pnl(2,3).ylabel('magnitude')

pnl.margin = [20, 20, 5, 5]; 
pnl.fontsize = 24; 
    


saveas(f, 'figures/fft_ex_signal+noise.svg')




%% 
%% 
%% FREQUENCY TAGGING 


%% fast periodic oddball design 

% sampling rate (Hz)
fs = 50; 

% number of repetitions 
n_rep = 12; 
n_oddball = 3; 

% period 
period = 1; 

base_frex = [1/period : 1/period : fs/2]; 
oddball_frex = [1/(n_oddball*period) : 1/(n_oddball*period) : fs/2];  
oddball_frex = oddball_frex(~ismembertol(oddball_frex, base_frex, 1e-9)); 

% total signal duration (s)
dur = period * n_rep; 
N = round(dur * fs); 
t = [0 : N-1] / fs; 
freq = [0 : N/2-1] / N * fs; 


f = figure('color', 'w', 'pos', [221 388 1380 603]); 
pnl = panel(f); 
pnl.pack('h', [50,50]); 
pnl(1).pack('v', 3); 
pnl(2).pack('v', 3); 


% only base response 
% ------------------

ir = get_erp_kernel(fs, ...
    'amplitudes', [0.2, 0.55],...
    't0s', [0, 0], ...
    'taus', [0.2, 0.050], ...
    'f0s', [1, 7], ...
    'duration', 0.98*period ...
    ); 

x = zeros(1, N); 
oddball_pos = [0 : n_oddball : n_rep-1]; 
base_pos = setdiff([0:n_rep-1], oddball_pos); 

idx_base = round(base_pos * period * fs) + 1; 
idx_oddball = round(oddball_pos * period * fs) + 1; 

x = insert_events(x, idx_base, ir); 
x = insert_events(x, idx_oddball, ir); 

mX = abs(fft(x)); 
mX = mX(1:N/2); 

ax = pnl(1,1).select(); 
plot(t, x, 'color', 'k', 'linew', 2)
xlim([0, dur]); 
xticks([0,dur])
yticks([])

ax = pnl(2,1).select(); 
plot_fft(freq, mX, 'ax', ax, 'maxfreqlim', freq(end), 'linew', 3, ...
         'frex_meter_rel', [])
     
yticks([])
xlim([0, fs/2])
xticks([1/period : 2/period : fs/2])
ax.XAxis.Visible = 'on'; 



% base+oddball response 
% ---------------------

% Let's simulate what happens when the response elicited by the oddball has
% higher ampiltude. 

x = zeros(1, N); 
oddball_pos = [0 : n_oddball : n_rep-1]; 
base_pos = setdiff([0:n_rep-1], oddball_pos); 

idx_base = round(base_pos * period * fs) + 1; 
idx_oddball = round(oddball_pos * period * fs) + 1; 

x = insert_events(x, idx_base, ir); 
x = insert_events(x, idx_oddball, 1.5*ir); 

mX = abs(fft(x)); 
mX = mX(1:N/2); 

ax = pnl(1,2).select(); 
plot(t, x, 'color', 'k', 'linew', 2)
xlim([0, dur]); 
xticks([0,dur])
yticks([])

ax = pnl(2,2).select(); 
plot_fft(freq, mX, 'ax', ax, 'maxfreqlim', freq(end), 'linew', 3, ...
         'frex_meter_rel', oddball_frex)
       
yticks([])
xlim([0, fs/2])
xticks([1/period : 2/period : fs/2])
ax.XAxis.Visible = 'on'; 



% base+oddball response 
% ---------------------

% Let's simulate what happens when the response elicited by the oddball has
% different shape (not necessarily higher amplitude). 

x = zeros(1, N); 
oddball_pos = [0 : n_oddball : n_rep-1]; 
base_pos = setdiff([0:n_rep-1], oddball_pos); 

idx_base = round(base_pos * period * fs) + 1; 
idx_oddball = round(oddball_pos * period * fs) + 1; 

ir = get_erp_kernel(fs, ...
    'amplitudes', [0.2, 0.55],...
    't0s', [0, 0], ...
    'taus', [0.2, 0.050], ...
    'f0s', [1, 7], ...
    'duration', 0.98*period ...
    ); 

x = insert_events(x, idx_base, ir); 

ir = get_erp_kernel(fs, ...
    'amplitudes', [0.16, 0.26],...
    't0s', [0, 0], ...
    'taus', [0.2, 0.040], ...
    'f0s', [1, 7], ...
    'duration', 0.98*period ...
    ); 

x = insert_events(x, idx_oddball, ir); 

mX = abs(fft(x)); 
mX = mX(1:N/2); 

ax = pnl(1,3).select(); 
plot(t, x, 'color', 'k', 'linew', 2)
xlim([0, dur]); 
xticks([0,dur])
yticks([])

ax = pnl(2,3).select(); 
plot_fft(freq, mX, 'ax', ax, 'maxfreqlim', freq(end), 'linew', 3, ...
         'frex_meter_rel', oddball_frex)
       
yticks([])
xlim([0, fs/2])
xticks([1/period : 2/period : fs/2])
ax.XAxis.Visible = 'on'; 


% format figure 
pnl(1).xlabel('time (s)')
pnl(2).xlabel('frequency (Hz)'); 

pnl(1,1).ylabel('amplitude')
pnl(1,2).ylabel('amplitude')
pnl(1,3).ylabel('amplitude')

pnl(2,1).ylabel('magnitude')
pnl(2,2).ylabel('magnitude')
pnl(2,3).ylabel('magnitude')

pnl.margin = [20, 20, 5, 5]; 
pnl.fontsize = 24; 
    


saveas(f, 'figures/fft_ex_oddball.svg')








