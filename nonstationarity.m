clear 
addpath(genpath('lib'))
addpath(genpath('~/projects_git/rnb_tools/src')); 

save_path = 'figures/nonstationarity'; 
mkdir(save_path)

%% 

% sampling rate (Hz)
fs = 128; 

% total signal duration (s)
dur = 5; 

N = round(dur * fs); 
n = [0 : N-1]; 
t = n / fs; 
freq = [0 : N/2-1] / N * fs; 


f0 = 10; 


x_in_phase = cos(2*pi*f0*t); 
x_anti_phase = cos(2*pi*f0*t - pi); 



env_in_phase = [zeros(1,fs*1), hann(fs*1)', zeros(1,fs*3)]; 
env_anti_phase = [zeros(1,fs*3), hann(fs*1)', zeros(1,fs*1)]; 

x = zeros(1, N); 
x = x + env_in_phase .* x_in_phase; 
x = x + env_anti_phase .* x_anti_phase; 

%%

% get DFT
X = fft(x); 

% find 10 Hz component
idx = 10 / fs * N + 1; 
freq(idx) 


f = figure('color', 'w', 'pos', [221 826 1410 165]); 
pnl = panel(f); 
ax = pnl.select(); 
plot(t, x, 'k', 'linew', 2)
xlabel('time')
ylabel('amplitude')
xticks([])
pnl.margin = [20, 20, 5, 5]; 
pnl.fontsize = 24; 
print(fullfile(save_path, sprintf('nonstationarity_time_1.png', f0)), '-dpng', '-painters', '-r600', f)

f = figure('color', 'w', 'pos', [221 826 1410 165]); 
pnl = panel(f); 
ax = pnl.select(); 
plot(t, x, 'k', 'linew', 2)
hold on 
plot(t, cos(2*pi*f0*t), 'r:', 'linew', 2)
xlabel('time')
ylabel('amplitude')
xticks([])
pnl.margin = [20, 20, 5, 5]; 
pnl.fontsize = 24; 
print(fullfile(save_path, sprintf('nonstationarity_time_2.png', f0)), '-dpng', '-painters', '-r600', f)

f = figure('color', 'w', 'pos', [221 826 1410 165]); 
pnl = panel(f); 
ax = pnl.select(); 
plot(t, x, 'k', 'linew', 2)
hold on 
plot(t, cos(2*pi*f0*t - pi), 'r:', 'linew', 2)
xlabel('time')
ylabel('amplitude')
xticks([])
pnl.margin = [20, 20, 5, 5]; 
pnl.fontsize = 24; 
print(fullfile(save_path, sprintf('nonstationarity_time_3.png', f0)), '-dpng', '-painters', '-r600', f)


%%

f = figure('color', 'w', 'pos', [221 826 700 165]); 
pnl = panel(f); 
ax = pnl.select(); 
stem(freq, abs(X(1:N/2)), 'k', 'linew', 2)
xlim([0, 17])
yticks([])
xlabel('frequency')
ylabel('magnitude')
pnl.margin = [20, 20, 5, 5]; 
pnl.fontsize = 16; 
print(fullfile(save_path, sprintf('nonstationarity_mX.png', f0)), '-dpng', '-painters', '-r600', f)
