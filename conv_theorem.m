% CONVOLUTION THEOREM.

clear 
addpath(genpath('lib'))
addpath(genpath('~/projects_git/rnb_tools/src')); 

%%  motivating example (leakage)

% What happens if the repetition rate of a signal is such that it doesn't
% complete exact integer number of cycles within the time span of our
% signal? 

% By the time we're done today, you'll know exactly why this happens, and
% you'll have one of the most powerful tools of DSP in your pocket ;) 

% sampling rate (Hz)
fs = 64; 

% total signal duration (s)
dur = 1; 

% number of samples in our signal 
N = round(dur * fs); 

% vector of samples
n = [0 : N-1]; 

% time in seconds for each sample 
t = n / fs; 

% frequency vector for the DFT 
freq = [0 : N/2-1] / N * fs; 


% open new figure
f = figure('color', 'w', 'pos', [221 388 1380 603]); 
pnl = panel(f); 
pnl.pack('h', [50,50]); 
pnl(1).pack('v', 3); 
pnl(2).pack('v', 3); 

% signal completes exact integer number of cycles (k)
% --------------------------------------------------

% frequency of the sine wave
k = 3; 
omg = 2*pi*k / N; 

% make a sine wave 
x = cos(omg * n); 

% get magnitude spectrum
mX = abs(fft(x)); 
mX = mX(1:N/2); 

% plot 
ax = pnl(1,1).select(); 
plot(t, x, '-o', 'color', 'k', 'linew', 2)
xlim([0, dur]); 
xticks([])
yticks([])

ax = pnl(2,1).select(); 
plot_fft(freq, mX, 'ax', ax, 'maxfreqlim', freq(end), 'linew', 15)
yticks([])
xlim([0, fs/2])
xticks([])
ax.XAxis.Visible = 'on'; 


% signal doesn'tcomplete exact integer number of cycles (k)
% ---------------------------------------------------------

% frequency of the sine wave
k = 3.3; 
omg = 2*pi*k / N; 
x = cos(omg * n); 

% get magnitude spectrum
mX = abs(fft(x)); 
mX = mX(1:N/2); 

ax = pnl(1,2).select(); 
plot(t, x, '-o', 'color', 'k', 'linew', 2)
xlim([0, dur]); 
xticks([])
yticks([])

ax = pnl(2,2).select(); 
plot_fft(freq, mX, 'ax', ax, 'maxfreqlim', freq(end), 'linew',15)
yticks([])
xlim([0, fs/2])
xticks([])
ax.XAxis.Visible = 'on'; 


save_path = 'figures/conv_theorem'; 
mkdir(save_path); 
saveas(f, fullfile(save_path, 'k_not_integer.svg'))


%% convolution 

fs = 100; 

% We will use zero padding to approximate the spectrum we'd get from a
% continuous Fourier transform. The more zeros we append to our signal, the
% more we'll approximate what we'd get if we actually worked with an
% infinitely long signal. 

% duration of sampled signal 
dur = 5; 
% duration of zero padding 
dur_pad = dur * 30 ; 
% total duation (signal+zero padding)
dur_total = dur + dur_pad; 

N = fs * dur_total; 
n = [0 : N-1]; 
t = n / fs; 
freq = [0 : N/2-1] / N * fs; 

% prepare zero padding 
pad = zeros(1, round(dur_pad * fs)); 

% prepare kernel 
kernel = get_erp_kernel(fs, ...
    'amplitudes', [0.2, 0.55],...
    't0s', [0, 0], ...
    'taus', [0.2, 0.050], ...
    'f0s', [1, 7], ...
    'duration', 1 ...
    ); 
    
% insert Dirac impulses into the signal (manually try different positions) 
impulse_times = [1.1, 2.3]; %      1.5, 3.6       1.5, 1.7
impulse_amps = [1, 1]; 
impulse_idx = dsearchn(t', impulse_times'); 
x = zeros(1, N); 
x(impulse_idx) = impulse_amps; 

% % or insert a continuous boxcar and see what happens
% x = zeros(1, N); 
% x(impulse_idx(1):impulse_idx(2)) = 1; 

% convolve signal and kernel (we can do it in the freq. domain, see Cohen
% 2014)
x_conv = fft_convolve(x, kernel); 


% open figure  
f = figure('color', 'w', 'pos', [221 388 1380 603]); 
pnl = panel(f); 
pnl.pack('h', [50,50]); 
pnl(1).pack('v', 3); 
pnl(2).pack('v', 3); 


% plot kernel 
kernel = [kernel, zeros(1, N-length(kernel))]; 

ax = pnl(1,1).select(); 
plot(t, kernel, '-', 'color', 'k', 'linew', 2)
xlim([0, dur])
xticks([])
yticks([])

ax = pnl(2,1).select(); 
mX = abs(fft(kernel)); 
mX = mX(1:N/2); 
plot(freq, mX, 'color', 'k', 'linew', 2, 'marker', 'none')
yticks([])
xlim([0, 20])
xticks([])
ax.XAxis.Visible = 'on'; 



% plot plot signal  
ax = pnl(1,2).select(); 
plot(t, x, '-', 'color', 'k', 'linew', 2)
xlim([0, dur])
xticks([])
yticks([])

ax = pnl(2,2).select(); 
mX = abs(fft(x)); 
mX = mX(1:N/2); 
plot(freq, mX, 'color', 'k', 'linew', 2, 'marker', 'none')
yticks([])
xlim([0, 20])
xticks([])
ax.XAxis.Visible = 'on'; 


% plot convolved signal and kernel 
ax = pnl(1,3).select(); 
plot(t, x_conv, '-', 'color', 'k', 'linew', 2)
xlim([0, dur])
xticks([])
yticks([])

ax = pnl(2,3).select(); 
mX = abs(fft(x_conv)); 
mX = mX(1:N/2); 
plot(freq, mX, 'color', 'k', 'linew', 2, 'marker', 'none')
yticks([])
xlim([0, 20])
xticks([])
ax.XAxis.Visible = 'on';


save_path = 'figures/conv_theorem'; 
mkdir(save_path); 
saveas(f, fullfile(save_path, 'conv_ex_4.svg'))



%% sampling as convolution (towards tapering)

% sampling rate (Hz)
fs = 1000; 

% duration of the rectangular window we'll take 
dur_win = 1; 

% ----------------------------------------------------------- frequency of
% sine wave frequency (change this to integer vs. non-integer and see what
% happens)
f0 = 5.4; 
% -----------------------------------------------------------

% To pretend we have a continuous signal, we'll compute the DFT of a super
% long singal (acting like the sine wave goes into infinity). For this to
% look good, we need to ensure we get an integer number of cycles within
% this loooong signal. 
dur_pad_before = 200; 
dur_pad_after = (1/f0) * ceil((dur_pad_before+dur_win+dur_pad_before) / (1/f0)) - (dur_pad_before+dur_win); 
dur_total = dur_pad_before + dur_win + dur_pad_after; 

N = round(dur_total * fs); 
n = [0 : N-1]; 
t = n / fs; 
freq = [0 : N/2-1] / N * fs; 

% make the signal 
x = cos(2 * pi * f0 * t); 

% get its DFT
mX = abs(fft(x)); 
mX = mX(1:N/2); 

% plot 
f = figure('color', 'w', 'pos', [221 388 1380 603]); 
pnl = panel(f); 
pnl.pack('h', [50,50]); 
pnl(1).pack('v', 4); 
pnl(2).pack('v', 4); 

ax = pnl(1,1).select(); 
plot(t, x, '-', 'color', 'k', 'linew', 2)
xlim([dur_pad-dur_win, dur_pad+dur_win+dur_win]); 
xticks([])
yticks([])

ax = pnl(2,1).select(); 
plot(freq, mX,'-', 'color', 'k', 'linew', 2)
yticks([])
xlim([0, 20])
xticks([])
ax.XAxis.Visible = 'on'; 


% select window (manually) and see what happens 

% % rectangular window 
% % ------------------
% win = [zeros(1, round(fs*dur_pad_before)), ...
%        ones(1, round(dur_win*fs)), ...
%        zeros(1, round(fs*dur_pad_after))]; 

% hann window 
% -----------
win = [zeros(1, round(fs*dur_pad_before)), ...
       hann(round(dur_win*fs))', ...
       zeros(1, round(fs*dur_pad_after))]; 

   
assert(length(win) == N); 

mX = abs(fft(win)); 
mX = mX(1:N/2); 

ax = pnl(1,2).select(); 
plot(t, win, '-', 'color', 'k', 'linew', 2)
xlim([dur_pad_before-dur_win, dur_pad_before+dur_win+dur_win]); 
xticks([])
yticks([])

ax = pnl(2,2).select(); 
plot(freq, mX,'-', 'color', 'k', 'linew', 2)
yticks([])
xlim([0, 20])
xticks([])
ax.XAxis.Visible = 'on'; 


% multiply signal and window and see what you get 
x_win = x .* win; 

mX = abs(fft(x_win)); 
mX = mX(1:N/2); 

ax = pnl(1,3).select(); 
plot(t, x_win, '-', 'color', 'k', 'linew', 2)
xlim([dur_pad_before-dur_win, dur_pad_before+dur_win+dur_win]); 
xticks([])
yticks([])

ax = pnl(2,3).select(); 
plot(freq, mX,'-', 'color', 'k', 'linew', 2)
yticks([])
xlim([0, 20])
xticks([])
ax.XAxis.Visible = 'on'; 

% "simulate" sampling

% find sample points we'd get from taking a DFT of just the time range
% covered by the window
N_samples = dur_win * fs; 

% cycles per window 
k_samples = [0 : N_samples/2]; 

% cycles per second 
freq_samples = k_samples / dur_win; 

% find the values of the "continuous" DFT at these sample positions 
idx_samples = dsearchn(freq', freq_samples'); 

% get the magnitudes at those sample locations 
mX_samples = mX(idx_samples); 

hold on 
plot(freq_samples, mX_samples,'o', 'color', 'r', 'MarkerFaceColor', 'r')

ax = pnl(1,4).select(); 
idx_win_start = round(dur_pad_before * fs) + 1;
idx_win_end = round((dur_pad_before + dur_win) * fs) + 1;

plot(t(idx_win_start:idx_win_end), x_win((idx_win_start:idx_win_end)), '-', 'color', 'k', 'linew', 2)
xlim([dur_pad_before-dur_win, dur_pad_before+dur_win+dur_win]); 
hold on 
plot([t(idx_win_start), t(idx_win_start)], [-1, 1], 'k')
plot([t(idx_win_start), t(idx_win_end)], [-1,-1], 'k')
xticks([])
yticks([])
ax.Visible = 'off'; 

ax = pnl(2,4).select(); 
plot_fft(freq_samples, mX_samples, 'ax', ax, 'maxfreqlim', freq(end), 'linew', 25)
yticks([])
xlim([0, 20])
xticks([])
ax.XAxis.Visible = 'on'; 



save_path = 'figures/conv_theorem'; 
mkdir(save_path); 
% print(fullfile(save_path, sprintf('sampling_as_conv_rect_k=%.1f.png', f0)), '-dpng', '-painters', '-r600', f)
saveas(f, fullfile(save_path, sprintf('sampling_as_conv_hann_k=%.1f.svg', f0)))


%% filtering is convolution 

% simulate a signal that is made of 1/f noise, and for for a few seconds, we
% get a burst of 10Hz sine wave. 

fs = 100; 
dur = 10; 
dur_pad = dur * 3; 
dur_total = dur + dur_pad; 

N = fs * dur_total; 
n = [0 : N-1]; 
t = n / fs; 
freq = [0 : N/2-1] / N * fs; 
    
x = sin(2*pi*10*t); 
dur_on = 5; 
win = [zeros(1, ((dur-dur_on)/2)*fs), hann(round(dur_on*fs))', zeros(1, ((dur-dur_on)/2)*fs)]; 
win = [win, zeros(1, N-length(win))]; 
x = x .* win; 
x = x + 0.9 * get_colored_noise(N, fs, -1.5); 

% design a FIR band-bass filter centered on 10 Hz and get filter kernel
flims = [8 12];
order = round( 5*fs/flims(1) );
kernel = fir1(order, [8, 10] / (fs/2));

% apply the filter (it's just convolution!!!)
x_filt = filter(kernel, 1, x); 


% open figure 
f = figure('color', 'w', 'pos', [221 388 1380 603]); 
pnl = panel(f); 
pnl.pack('h', [50,50]); 
pnl(1).pack('v', 3); 
pnl(2).pack('v', 3); 

% plot filter kernel time domain 
kernel = [kernel, zeros(1, N-length(kernel))]; 

ax = pnl(1,1).select(); 
plot(t, kernel, '-', 'color', 'k', 'linew', 2)
xlim([0, dur])
xticks([])
yticks([])

% plot filter kernel frequency domain 
ax = pnl(2,1).select(); 
mX = abs(fft(kernel)); 
mX = mX(1:N/2); 
plot(freq, mX, 'color', 'k', 'linew', 2, 'marker', 'none')
yticks([])
xlim([0, 20])
xticks([])
ax.XAxis.Visible = 'on'; 


% plot signal time domain 
ax = pnl(1,2).select(); 
plot(t, x, '-', 'color', 'k', 'linew', 2)
xlim([0, dur])
xticks([])
yticks([])

% plot signal freq. domain 
ax = pnl(2,2).select(); 
mX = abs(fft(x)); 
mX = mX(1:N/2); 
plot(freq, mX, 'color', 'k', 'linew', 2, 'marker', 'none')
yticks([])
xlim([0, 20])
xticks([])
ax.XAxis.Visible = 'on'; 


% plot filtered siganl time domain 
ax = pnl(1,3).select(); 
plot(t, x_filt, '-', 'color', 'k', 'linew', 2)
xlim([0, dur])
xticks([])
yticks([])

% extract envelope of the filtered signal to see how magnitude of the
% filtered signal changes over time, and captures the " oscillation" burst
env = abs(hilbert(x_filt)); 
hold on 
plot(t, env, '-', 'color', 'r', 'linew', 2)

% plot filtered signal freq. domain 
ax = pnl(2,3).select(); 
mX = abs(fft(x_filt)); 
mX = mX(1:N/2); 
plot(freq, mX, 'color', 'k', 'linew', 2, 'marker', 'none')
yticks([])
xlim([0, 20])
xticks([])
ax.XAxis.Visible = 'on';



%% wavelet transform is just like filtering: convolution 

clear

fs = 200; 
dur = 8; 

N = fs * dur; 
n = [0 : N-1]; 
t = n / fs; 
freq = [0 : N/2-1] / N * fs; 
    
% create example signal 
x = sin(2*pi*10*t); 
dur_on = 2; 
win = [zeros(1, ((dur-dur_on)/2)*fs), hann(round(dur_on*fs))', zeros(1, ((dur-dur_on)/2)*fs)]; 
win = [win, zeros(1, N-length(win))]; 
x = x .* win; 
x = x + 0.9 * get_colored_noise(N, fs, -1.5); 
x = x'; 

% get wavelet
wt = [-dur/2 : 1/fs : dur/2]'; 
n_w = length(wt); 
n_hw = ceil(n_w/2); 
n_conv = N + n_w - 1; 

fwhmT = 0.4; 
wfrex = 10; 
gwin = exp( (-4*log(2)*wt.^2) ./ fwhmT^2 );
cos_wave = exp( 1j*2*pi*wfrex*wt ); 
wavelet = cos_wave .* gwin;  

% convolve wavelet and signal 
X_w = fft(wavelet, n_conv);
X_x = fft(x, n_conv);
x_filt = ifft(X_x .* X_w, n_conv);
x_filt = x_filt(n_hw : end-n_hw+1); 


% open figure 
f = figure('color', 'w', 'pos', [221 388 1380 603]); 
pnl = panel(f); 
pnl.pack('h', [50,50]); 
pnl(1).pack('v', 3); 
pnl(2).pack('v', 3); 


% plot the wavelet in the time domain 
ax = pnl(1,1).select(); 
plot(wt, real(wavelet), '-', 'color', [227, 87, 27]/255, 'linew', 1)
hold on 
plot(wt, imag(wavelet), '-', 'color', [14, 151, 196]/255, 'linew', 1)
xticks([])
yticks([])

% plot the magnitude DFT of the wavelet 
ax = pnl(2,1).select(); 
freq_w = [0 : floor(n_conv/2)] / n_conv * fs; 
plot(freq_w, abs(X_w(1:length(freq_w))), 'color', 'k', 'linew', 2, 'marker', 'none')
yticks([])
xlim([0, 20])
xticks([])
ax.XAxis.Visible = 'on'; 

% plot the signal in the time domain 
ax = pnl(1,2).select(); 
plot(t, x, '-', 'color', 'k', 'linew', 2)
xlim([(dur-dur_on)/2 - 2, (dur-dur_on)/2 + dur_on + 2])
xticks([])
yticks([])

% DFT of the signal 
ax = pnl(2,2).select(); 
mX = abs(fft(x)); 
mX = mX(1:N/2); 
plot(freq, mX, 'color', 'k', 'linew', 2, 'marker', 'none')
yticks([])
xlim([0, 20])
xticks([])
ax.XAxis.Visible = 'on'; 


% time domain of the filtered signal 
ax = pnl(1,3).select(); 
plot(t, real(x_filt), '-', 'color', [227, 87, 27]/255, 'linew', 1)
hold on 
plot(t, imag(x_filt), '-', 'color', [14, 151, 196]/255, 'linew', 1)
hold on 
plot(t, abs(x_filt), '-', 'color', 'k', 'linew', 2)
xlim([(dur-dur_on)/2 - 2, (dur-dur_on)/2 + dur_on + 2])
xticks([])
yticks([])

% DFT of the filtered signal 
ax = pnl(2,3).select(); 
mX = abs(X_x(1:length(freq_w)) .* X_w(1:length(freq_w))); 
plot(freq_w, mX, 'color', 'k', 'linew', 2, 'marker', 'none')
yticks([])
xlim([0, 20])
xticks([])
ax.XAxis.Visible = 'on';









