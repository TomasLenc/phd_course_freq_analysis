

% Let's simulate 


clear 
fprintf('\n');
addpath(genpath('lib'))
addpath(genpath('~/projects_git/rnb_tools/src')); 
addpath(genpath('~/projects_git/acf_tools/src')); 


%% parameters 

% sampling rate (Hz)
fs = 5000; 

% total signal duration (s)
dur = 10; 

% frequency of input 1 and input 2  
f1 = 11; 
f2 = 13; 

N = round(dur * fs); 
t = [0 : N-1] / fs; 
freq = [0 : N/2-1] / N * fs; 

% kernel 
ir = get_square_kernel(fs, 'duration', 0.050); 

% make full signal 
n_periods = dur * f1; 
idx = round([0 : n_periods-1] * 1/f1 * fs) + 1; 
x1 = zeros(1, N); 
x1 = insert_events(x1, idx, ir); 

n_periods = dur * f2; 
idx = round([0 : n_periods-1] * 1/f2 * fs) + 1; 
x2 = zeros(1, N); 
x2 = insert_events(x2, idx, ir); 

% linear summation 
x_lin = x1 + x2; 

% summation followed by saturating nonlinearity 
x_nonlin = (x1 + x2) .^ 0.3; 

% get magnitude spectra
mX1 = abs(fft(x1)); 
mX1 = mX1(1:N/2); 
mX1(1) = 0; 

mX2 = abs(fft(x2)); 
mX2 = mX2(1:N/2); 
mX2(1) = 0; 

mX_lin = abs(fft(x_lin)); 
mX_lin = mX_lin(1:N/2); 
mX_lin(1) = 0; 

mX_nonlin = abs(fft(x_nonlin)); 
mX_nonlin = mX_nonlin(1:N/2); 
mX_nonlin(1) = 0; 



% plot 
% ----

f = figure('color', 'w', 'pos', [221 388 1380 603]); 
pnl = panel(f); 
pnl.pack('h', [50,50]); 
pnl(1).pack('v', 4); 
pnl(2).pack('v', 4); 

ax = pnl(1,1).select(); 
plot(t, x1, 'color', [222 45 38]/255, 'linew', 2)
xlim([0, 1]); 
ylim([0, 2]); 
xticks([])
yticks([])

ax = pnl(2,1).select(); 
plot_fft(freq, mX1, 'ax', ax, 'maxfreqlim', freq(end), 'linew', 3, ...
         'frex_meter_rel', f1*[1:10], 'frex_meter_unrel', f2*[1:10])
yticks([])
xlim([0, 5*max(f1,f2)])


ax = pnl(1,2).select(); 
plot(t, x2, 'color', [49, 130, 189]/255, 'linew', 2)
xlim([0, 1]); 
ylim([0, 2]); 
xticks([])
yticks([])

ax = pnl(2,2).select(); 
plot_fft(freq, mX2, 'ax', ax, 'maxfreqlim', freq(end), 'linew', 3, ...
         'frex_meter_rel', f1*[1:10], 'frex_meter_unrel', f2*[1:10])
yticks([])
xlim([0, 5*max(f1,f2)])



ax = pnl(1,3).select(); 
plot(t, x_lin, 'color', 'k', 'linew', 2)
xlim([0, 1]); 
ylim([0, 2]); 
xticks([])
yticks([])

ax = pnl(2,3).select(); 
plot_fft(freq, mX_lin, 'ax', ax, 'maxfreqlim', freq(end), 'linew', 3, ...
         'frex_meter_rel', f1*[1:10], 'frex_meter_unrel', f2*[1:10])
yticks([])
xlim([0, 5*max(f1,f2)])




ax = pnl(1,4).select(); 
plot(t, x_nonlin, 'color', 'k', 'linew', 2)
xlim([0, 1]); 
ylim([0, 2]); 
xticks([])
yticks([])

ax = pnl(2,4).select(); 
plot_fft(freq, mX_nonlin, 'ax', ax, 'maxfreqlim', freq(end), 'linew', 3, ...
         'frex_meter_rel', f1*[1:10], 'frex_meter_unrel', f2*[1:10])
yticks([])
xlim([0, 5*max(f1,f2)])





% format figure 
pnl(1).xlabel('time (s)')
pnl(2).xlabel('frequency (Hz)'); 

for i=1:4
    pnl(1,i).ylabel('amplitude')
    pnl(2,i).ylabel('magnitude')
end

pnl.margin = [20, 20, 5, 5]; 
pnl.fontsize = 24; 

print('figures/fft_ex_convergence.svg', '-dsvg', '-painters', '-r600', f); 


% plot the saturating nonlinearity function  
f = figure('color', 'white'); 
plot(t, t.^0.3)
box off
print('figures/fft_ex_convergence_saturation.svg', '-dsvg', '-painters', '-r600', f); 
