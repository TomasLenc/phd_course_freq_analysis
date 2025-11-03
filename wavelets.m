% Here we explore how to design a family of Morlet wavelets. 

clear 
addpath(genpath('lib'))
addpath(genpath('~/projects_git/rnb_tools/src')); 

save_path = 'figures/wavelets'; 
mkdir(save_path)

%% 

% sampling rate 
fs = 128; 

% wavelet parameters
% ------------------

% wavelet frequencies 
n_wfrex = 8; 
wfrex_min = 2; 
wfrex_max = 30; 
wfrex = logspace(log10(wfrex_min), log10(wfrex_max), n_wfrex);
         
% time-domain full width at half maximum for each wavelet (we can increase
% the temporal resolution with increasing frequency)
fwhmT = 2 * 1./wfrex; 
                
% convolution parameters
% ----------------------

dur_data        = 10; 
n_data          = round(dur_data * fs); 
wt              = [-5:1/fs:5]'; 
n_w             = length(wt); 
n_hw            = floor(n_w/2); 
n_conv          = n_data+n_w-1; 
n_conv_pow2     = pow2(nextpow2(n_conv)); 
n_out_data      = round(dur_data * fs); 
freq_w          = [0 : floor(n_conv_pow2/2)] / n_conv_pow2*fs; 


% prepare wavelets 
% ----------------

% NOTE: we need to make the spectra column-wise because of bsxfun
% later...trasposing this later would do complex conjugate!

% so let's just make everything column-wise...

gwin = nan(length(wt), n_wfrex); 
cmw = nan(length(wt), n_wfrex); 
cmwX = nan(n_conv_pow2, n_wfrex); 

for i_f=1:n_wfrex

    % gaussian (width based on the requested FWHM)
    gwin(:,i_f) = exp( (-4*log(2)*wt.^2) ./ fwhmT(i_f)^2 );
    %     % Mike Cohen has a mistake in his code, squaring the whole thing...
    %     gwin = exp(-(4*log(2)*wt).^2/hgb.fwhmT(fi).^2)

    % cosine (complex, so we get cos and sin bundled together for free)
    cos_wave = exp( 1j*2*pi*wfrex(i_f)*wt ); 

    % multiply cosine wave and gaussian 
    cmw(:,i_f) = cos_wave .* gwin(:,i_f);    

    % get DFT of wavelet
    cmwX(:,i_f) = fft(cmw(:,i_f), n_conv_pow2); 
    
    % Normalization of the wavelet for accurate reconstruction of the signal 
    % amplitude can be achieved by setting the amplitude of the Fourier transform 
    % of the wavelet to have a peak of 1; this is analogous to conceptualizing the 
    % frequency response of a temporal filter as a gain function and setting the 
    % passband frequencies with a gain of 1. 
    cmwX(:,i_f) = cmwX(:,i_f) ./ max(cmwX(:,i_f));
    
end


%% plot example wavelet

f = figure('color', 'w', 'pos', [221 785 1300 600]); 
pnl = panel(f); 
pnl.pack('h', 2); 
pnl(1).pack('v', 3); 
pnl(2).pack('v', 3); 

% select one example frequency 
i_f = 5; 

% normalize everything to 1 for plotting
cos_wave = real(exp(1j*2*pi*wfrex(i_f)*wt)); 
gwin_norm = gwin(:,i_f) ./ max(gwin(:,i_f)); 
cmw_norm = cmw(:,i_f) ./ max(gwin(:,i_f)); 
cmwX_norm  = cmwX(:,i_f) ./ max(cmwX(:,i_f));

mX_gwin = abs(fft(gwin_norm, n_conv_pow2)); 
mX_gwin(1) = 0; 
mX_gwin = mX_gwin ./ max(mX_gwin); 


ax = pnl(1, 1).select(); 
plot(wt, cos_wave, 'color', [227, 87, 27]/255, 'linew', 2); 
xlim([-2, 2])
xticks([-1, 0, 1])
xlabel('time')
ylabel('amplitude')

ax = pnl(2, 1).select(); 
plot([wfrex(i_f), wfrex(i_f)], [0, 1], '-', 'color', [227, 87, 27]/255, 'linew', 2); 
xlim([0, wfrex(i_f)*4])
yticks([0, 1])
xlabel('frequency')
ylabel('magnitude')


ax = pnl(1, 2).select(); 
plot(wt, gwin_norm, ':k', 'linew', 2); 
xlim([-2, 2])
xticks([-1, 0, 1])
xlabel('time')
ylabel('amplitude')

ax = pnl(2, 2).select(); 
plot(freq_w, mX_gwin(1:length(freq_w)), ':', 'color', 'k', 'linew', 2); 
xlim([0, wfrex(i_f)*4])
yticks([0, 1])
xlabel('frequency')
ylabel('magnitude')


ax = pnl(1, 3).select(); 
plot(wt, real(cmw_norm), 'color', [227, 87, 27]/255, 'linew', 2); 
hold on 
plot(wt, imag(cmw_norm), 'color', [14, 151, 196]/255, 'linew', 2); 
plot(wt, gwin_norm, ':k', 'linew', 2); 
xlim([-2, 2])
xticks([-1, 0, 1])
xlabel('time')
ylabel('amplitude')

ax = pnl(2, 3).select(); 
plot(freq_w, abs(cmwX_norm(1:length(freq_w))), 'color', [227, 87, 27]/255, 'linew', 2); 
xlim([0, wfrex(i_f)*4])
yticks([0, 1])
xlabel('frequency')
ylabel('magnitude')

pnl.title(sprintf('f_w = %.1f Hz', wfrex(i_f))); 
    
pnl.de.margin = [25, 25, 0, 0]; 
pnl.margin = [20, 20, 5, 15]; 
pnl.fontsize = 20; 


print(fullfile(save_path, sprintf('example_wavelet.svg')), '-dsvg', '-painters', '-r600', f)


%% empirical wavelet properties

% Here, we take each wavelet generated above, and measure it's FWHM
% empirically, both, in the time domain and the frequency domain. 

% We will also plot everything in one figure 
f_wav = figure('color','white','position', [-371 1868 2418 326]); 
pnl = panel(f_wav);
pnl().pack('v',2); % time and frequency plot
pnl(1).pack('h',n_wfrex); 
pnl(2).pack('h',n_wfrex); 

fwhmT = nan(1,n_wfrex); 
fwhmF = nan(1,n_wfrex); 

for i_f=1:n_wfrex
        
    % compute the empirical temporal FWHM in seconds (later converted to ms)
    gwin_norm = gwin(:,i_f)./max(gwin(:,i_f)); % normalize to 1
    cmw_norm = cmw(:,i_f)./max(gwin(:,i_f)); 
    midp = dsearchn(wt,0);
    fwhmT_upper = wt(midp-1+dsearchn(gwin_norm(midp:end),.5)); 
    fwhmT_lower = wt(dsearchn(gwin_norm(1:midp),.5)); 
    fwhmT(i_f) = fwhmT_upper - fwhmT_lower;

    % compute the empirical frequency FWHM in Hz
    cmwX_norm  = cmwX(:,i_f)./max(cmwX(:,i_f)); % normalize to 1 to get the right units 
%         cmwX_norm = cmwX_norm.^2; % get power (not amplitude)??? (I saw both in Cohen...sometimes he just calls amplitude power...or maybe he has a mistake in the code?)
    frex_idx = dsearchn(freq_w',wfrex(i_f));
    fwhmF_upper = freq_w(frex_idx-1+dsearchn(abs(cmwX_norm(frex_idx:end)),.5)); 
    fwhmF_lower = freq_w(dsearchn(abs(cmwX_norm(1:frex_idx)),.5)); 
    fwhmF(i_f) = fwhmF_upper - fwhmF_lower;

    % plot tiome domain 
    pnl(1,i_f).select(); 
    plot(wt(1:1:end),real(cmw_norm(1:1:end)), 'color', 'k'); 
    hold on
    plot([fwhmT_lower,fwhmT_lower],[-1,1],'r'); 
    plot([fwhmT_upper,fwhmT_upper],[-1,1],'r'); 
    ax = gca; 
    ax.XLim = [-0.6,0.6]; 
    ax.XTick = [-0.6, 0, 0.6]; 
    if i_f>1
        ax.XTick = []; 
        ax.YTick = []; 
    end
    title(sprintf('f = %.1f Hz \nfwhmT = %.0f ms', wfrex(i_f), 1000*fwhmT(i_f))); 

    % plot frequency domain 
    pnl(2,i_f).select(); 
    cmw_plot_maxfreq = wfrex_max * 2; 
    cmw_plot_maxfreq_idx = round(cmw_plot_maxfreq/fs * n_conv_pow2)+1; 
    cmw_freq_plot = [0:cmw_plot_maxfreq_idx-1]/n_conv_pow2*fs; 
    plot(cmw_freq_plot(1:1:end), abs(cmwX_norm(1:1:length(cmw_freq_plot))), 'color', 'k'); 
    hold on
    plot([fwhmF_lower,fwhmF_lower],[0,1],'r'); 
    plot([fwhmF_upper,fwhmF_upper],[0,1],'r'); 
    title(sprintf('fwhmF = %.2f Hz',fwhmF(i_f))); 
    ax = gca; 
    ax.XLim = [0,cmw_plot_maxfreq]; 
    ax.XTick = [round(wfrex(i_f), 1)]; 
    ax.YTick = []; 
    
    fprintf('f = %.1f Hz \t fwhmT = %.3f sec \t fwhmF = %.3f Hz \n', ...
            wfrex(i_f), fwhmT(i_f), fwhmF(i_f)); 
      
end


pnl.de.margin = [5,1,1,20]; 
pnl.margin = [15, 15, 5, 15]; 

pnl.fontsize = 16; 

print(fullfile(save_path, sprintf('wavelet_family.svg')), '-dsvg', '-painters', '-r600', f_wav)
