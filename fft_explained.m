% Illustrate how FFT is computed just using cosine waves. 

clear
addpath(genpath('lib'))

% number of samples
N = 32; 
n = [0 : N-1]; 

% some ploting parameters
col = [50, 168, 82]/255; 
linew = 3; 

save_path = 'figures/fft_explained'; 
mkdir(save_path)

%% generating sine waves
% In terms of radians per sample. 

f = figure('color', 'white', 'pos', [637 556 475 400]); 
pnl = panel(f); 
pnl.pack('v', 2); 

% k expresses how many cycles the sine wave will complete within the whole
% signal (i.e. within the N samples we have)
k = 3; 

% Omega expresses how many radians the sine wave will jump between two
% successive samples.
omg = (2 * pi * k) / N; 

% Multiply by the vector of samples (0, 1, 2, 3, ... , N-1) to get the
% actual phase values for each sample. 
phase = omg * n; 

% Then run it through the cosine function to get amplitudes.
x = cos(phase); 

% plot the sine wave 
ax = pnl(1).select(); 
stem(n, x, 'color', col, 'linew', linew, 'marker', 'o'); 
hold on 
plot(n, x, 'color', col, 'linew', linew); 

ax.YLim = [-2, 2]; 
ax.YTick = [-2, 2]; 
ax.XLim = [0, N-1]; 
pnl(1).xlabel('sample');
pnl(1).ylabel('amplitude');



ax = pnl(2).select(); 

% Sine wave at a particular frequency omega has 2 parameters:
%     A : amplitude (aka scale, i.e. how much up/down it goes) phi :
%     phi : shift (aka phase offset, i.e. what phase it starts at the 0th sample)
A = 1.7; 
phi = pi / 2; 

x = A * cos(omg * n - phi); 

stem(n, x, 'color', col, 'linew', linew, 'marker', 'o'); 
hold on 
plot(n, x, 'color', col, 'linew', linew); 

ax.YLim = [-2, 2]; 
ax.YTick = [-2, 2]; 
ax.XLim = [0, N-1]; 
pnl(2).xlabel('sample');
pnl(2).ylabel('amplitude');

pnl.fontsize = 16; 
pnl.de.margin = [10, 25, 5, 5]; 



%% sinewave with any A and phi is just a weighted cos + sin
%
% Remeber from highschool? 
% 
%   cos(a-b) = cos(a)cos(b) + sin(a)sin(b)
% 
% Now applied to a scaled shifted sine wave (with frequency omg): 
%
%     A cos(omg n - phi) = A cos(phi)cos(omg n) + A sin(phi)sin(omg n)
% 
% So we have two weights, let's call them c and s
% 
%     c = A cos(phi)
%     s = A sin(phi)
% 
% `c` multiplies a standard cosine wave, `s` multiplies a standard sine wave
% and when we add thohse scaled waves up (sample by sample) we get the sine
% wave with the required amplitude A and phase offset phi. 

f = figure('color', 'white', 'pos', [632 564 544 380]); 
pnl = panel(f); 
pnl.pack('v', 2); 
pnl(1).pack('h', [70, 30]); 
pnl(2).pack('h', [70, 30]); 

% set the frequency (radians per sample) 
omg = (2 * pi * 3) / N; 

% amplitude
A = 3; 

% delay
phi = pi / 3; 

% we can generate the scaled+shifted sine wave directly from the A and phi
% parameter
x = A * cos(omg * n - phi); 

ax = pnl(1, 1).select(); 
plot(n, x, 'color', 'black', 'linew', linew); 
ax.YLim = [-3, 3]; 
ax.YTick = [-3, 3]; 
ax.XLim = [0, N-1]; 
ylabel('amplitude'); 

% Let's plot the paramers A and phi using polar coordinates -> let's make A
% the radius and phi the angle of a vector
ax = polaraxes; 
polarplot([phi, phi], [0, A], 'linew', linew); 
% ax.RTick = [3];
ax.FontSize = 12; 
pnl(1, 2).select(ax); 

% This is just a fun way to visualize the parameters of the sine wave
% (assuming we know its frequency, this is all we need to know to contruct
% the wave).

% Now, let's see if we can reconstruct our wave using the weights `c` and
% `s`

% Let's calculate them from the formula above:
c = A * cos(phi); 
s = A * sin(phi); 

% create standard sine and cosine
cos_wave = cos(omg * n); 
sin_wave = sin(omg * n); 

% sum the weighted sine and cosine to create the target wave that should
% have amplitude A and phase offset phi
x_sum = c * cos_wave + s * sin_wave; 

ax = pnl(2, 1).select(); 
h = plot(n, c * cos_wave, 'linew', linew); 
h.Color(4) = 0.5; 
hold on
h = plot(n, s * sin_wave, 'linew', linew); 
h.Color(4) = 0.5; 
plot(n, x_sum, 'color', 'black', 'linew', linew); 
ax.YLim = [-3, 3]; 
ax.YTick = [-3, 3]; 
ax.XLim = [0, N-1]; 
ylabel('amplitude'); 

% Interestingly, if we plot `c` on x-axis and `s` on y-axis in a cartesian
% space, we get exactly the same vector that we plotted using A and phi in
% polar coordinates before. => This is the link between two ways of
% thinking about the same sine wave: (1) in terms of A and phi, (2) in
% terms of c and s.
ax = pnl(2, 2).select(); 
plot([0, c], [0, s], 'linew', linew); 
ax.XLim = [-4, 4]; 
ax.YLim = [-4, 4]; 
ax.XTick = [-3, 3]; 
ax.YTick = [-3, 3]; 
xlabel('cos'); 
ylabel('sin'); 
ax.XAxisLocation = 'origin'; 
ax.YAxisLocation = 'origin'; 

% Oh yeah, fun fact: When we have the c and s parameters, and we want to
% know the amplitude of the wave they describe, we just need to go from the
% cartesian into polar, using pythagorean theorem (how do you spell
% pytaghoras?...ooft).
% 
% Remeber from highschool?
% 
% In our case: A = sqrt(c^2 + s^2)
%
pnl(2, 1).xlabel('sample');
pnl.de.margin = [15, 15, 5, 5]; 
pnl.margin = [15, 20, 15, 15]; 

pnl.fontsize = 16; 

saveas(f, fullfile(save_path, 'weigted_sum_sin_cos.svg')); 

%% exponential numbers are just a convenient way to do arithmetics with cos and sin

% Instead of keeping track of the pair `c` and `s` as 2 separate numbers,
% there's a convenient way to represent them together -> we can use a
% complex number. Complex number has a real component -> let's put the
% value of `c` there. And an imaginary component -> let's put the value of
% `s` there. So now we have a way to conveniently keep track of the `c, s`
% pair that represents a specific sine wave!
% 
% Moreover, the connection between the polar and cartesian representation
% we've seen above is directly obvious when we use complex numbers thanks
% to Euler's formula:
% 
% Euler's formula:
%     A exp(i*phi) = A cos(phi) + A i*sin(phi)
% 

A = 3; 
phi = pi / 3; 

c = A * cos(phi); 
s = A * sin(phi); 

% One way of defining the complex number, directly assigning the real (`c`) and
% imaginary (`s`) value. 
x = c + 1j*s; 

% Equivalent way is to specify the complex number using A and phi, just 
% plugging it into an exponential. Matlab internally uses Euler's formula to
% convert into cartesian represenation for the real and imaginary compoent. 
y = A * exp(1j * phi); 

% Why complex numbers? 
% E.g., using complex numbers, it's easy to show how multiplying 2 sinewaves  
% with the same frequency works.
% 
%     A * exp(i*alpha)   *   B * exp(i*beta) =  A * B * exp(i*[alpha+beta])
% 
% -> exp representation super handy if you want to change phase


%% DFT matrix

% Let's generate the DFT matrix that contains sine and cosine waves which we'll
% decompose our signal into. In other words, the DFT will find a weight for
% each cosine and sine. Using these weights, we can take a weighted combination
% of sines+cosines to reconstruct are signal (perfectly!). 

linew = 2; 

% We start with a cosine, sine pair that does exactly 0 cycles within our N
% samples: 
omg = (2 * pi * 0) / N; 
col0_c = cos(omg * n); 
col0_s = sin(omg * n); 

% Next, we'll need a cosine, sine pair where each wave does exactly 1 cycle
% within the N samples
omg = (2 * pi * 1) / N; 
col1_c = cos(omg * n); 
col1_s = sin(omg * n); 

% next, 2 cycles
omg = (2 * pi * 2) / N; 
col2_c = cos(omg * n); 
col2_s = sin(omg * n); 

% 3 cycles
omg = (2 * pi * 3) / N; 
col3_c = cos(omg * n); 
col3_s = sin(omg * n); 

% etc etc...... up to N/2+1

% Here we generate and plot all the cosine/sine waves in the DFT matrix 
f = figure('color', 'white', 'pos', [98 139 1747 578]); 
pnl = panel(f); 
pnl.pack('h', 16+1); 

cols = brewermap(16+1, '-Set3'); 
n_interp = [0 : 0.1 : N-1]; 

for i_col=0:16
    
    pnl(17-i_col).pack('h', 2); 

    omg = (2 * pi * i_col) / N; 
    col_c = cos(omg * n); 
    col_s = sin(omg * n); 

    ax = pnl(17-i_col, 2).select(); 
    stem(n, col_c, 'linew', linew, 'color', cols(i_col+1, :)); 
    hold on 
    plot(n_interp, cos(omg * n_interp), 'linew', linew, 'color', cols(i_col+1, :)); 
    ax.XLim = [0, N-1];
    ax.XAxis.Visible = 'off'; 
    ax.YAxis.Visible = 'off'; 
    view(ax, 90, 90); 
    
    ax = pnl(17-i_col, 1).select(); 
    stem(n, col_s, 'linew', linew, 'color', cols(i_col+1, :)); 
    hold on 
    plot(n_interp, sin(omg * n_interp), 'linew', linew, 'color', cols(i_col+1, :)); 
    ax.XLim = [0, N-1];
    ax.YLim = [-1, 1]; 
    ax.XAxis.Visible = 'off'; 
    ax.YAxis.Visible = 'off'; 
    view(ax, 90, 90); 
    
    pnl(17-i_col).title(sprintf('%d', i_col))
    
end

pnl.de.margin = 0; 
pnl.de.marginright = 3; 

pnl.margintop = 15; 
pnl.fontsize = 12; 

% Note: sine with frequency 0 is always 0 => it's redundant (we can keep it
% in the matrix but we can ignore it for the DFT)
% 
% Same with the sine at frequency that does exactly N/2 cycles within N
% samples -> the sine wave is always at 0. Again, to be ignored...
% 
% Besides the two sines above, we should end up with N unique weights.
%
% Note also that the cosine with frequency 0 is always one => the
% point-by-point multiplication with the signal, and adding the results in
% the end will be proportional to the signal mean. This component is
% therefore also called "DC" (cominf from direct current in electrical
% engineering).

saveas(f, fullfile(save_path, 'fft_matrix.svg'))


% let's plot example random signal (for the figure on slides)
f = figure('color', 'white', 'pos', [98 139 120 578]); 
pnl = panel(f); 

ax = pnl.select(); 
x = randn(1, N)'; 
x = x ./ max(abs(x)); 
stem(n, x, 'linew', linew, 'color', 'k'); 
hold on 
ax.XLim = [0, N-1];
ax.YLim = [-1, 1]; 
ax.XAxis.Visible = 'off'; 
ax.YAxis.Visible = 'off'; 
view(ax, 90, 90); 
pnl.de.margin = 0; 
pnl.de.marginright = 3; 

pnl.margintop = 15; 
pnl.fontsize = 12; 

saveas(f, fullfile(save_path, 'example_signal.svg'))


%% DFT sinewaves in polar coordinates 

% What about waves faster than N/2 cycles per N samples?

% This is where aliasing comes into play. Dut to the fact that we have a
% sampled signal, the cosines and sines start to look very much like the
% cosies and sines with slower frequencies. In fact, these higher frequency
% components are redundant (for real signals), and are sometimes also
% called "negative frequencies". Why? Try the simulation below:

linew = 2; 

% try with 3, -3, 32-3
n_cycles = 32-3; 

omg = (2 * pi * n_cycles) / N; 


f = figure('color', 'white', 'pos', [98 139 800 300]); 
pnl = panel(f); 

pnl.pack('h', [70, 30]); 
pnl(1).pack('v', 2); 


ax_cos = pnl(1, 1).select(); 
ax_cos.XLim = [0, N-1]; 
ax_cos.YLim = [-1, 1]; 
stem(0, cos(omg * 0), 'o', ...
    'color', 'r', 'markerfacecolor', 'r', 'markersize', 10); 
hold(ax_cos, 'on'); 


ax_sin = pnl(1, 2).select(); 
ax_sin.XLim = [0, N-1]; 
ax_sin.YLim = [-1, 1]; 
plot(0, sin(omg * 0), 'o', ...
    'color', 'b', 'markerfacecolor', 'b', 'markersize', 10); 
hold(ax_sin, 'on'); 


ax_polar = polaraxes; 
hold on
polarplot([pi, 0], [1, 1], 'linew', 3, 'color', 'k')
polarplot([-pi/2, pi/2], [1, 1], 'linew', 3, 'color', 'k')

ax_polar.RTick = []; 
ax_polar.RLim = [0, 1]; 
ax_polar.ThetaAxisUnits = 'radians'; 
ax_polar.ThetaTick = 2*pi * [0:N-1]/N; 
ax_polar.ThetaTickLabel = cellfun(@num2str, num2cell([0:N-1]), 'uni', 0); 


pnl(2).select(ax_polar); 
h_polar = polarplot(omg * 0, 1, 'o',...
    'color', 'k', 'markerfacecolor', 'k', 'markersize', 10); 

% plot thre rest of the points one by one
for n=1:N-1
    
    h_polar.ThetaData = omg * n; 
    
    stem(ax_cos, n, cos(omg * n), 'o', ...
        'color', 'r', 'markerfacecolor', 'r', 'markersize', 10); 
    stem(ax_sin, n, sin(omg * n), 'o', ...
        'color', 'b', 'markerfacecolor', 'b', 'markersize', 10); 
    
    pause(1); 
end

% plot everything together
delete(h_polar); 

polarplot(ax_polar, omg * [0:N-1], 0.95, 'o', ...
    'color', 'k', 'markerfacecolor', 'k', 'markersize', 10); 

pnl.margin = [10, 10, 10, 10]; 

pnl.title(sprintf('%d cycles per vector', n_cycles)); 
pnl.fontsize = 12; 



%% redundancy + symmetry in DFT after frequency N/2 

N = 32; 

x = randn(1, N)'; 

X = fft(x); 

% get magniudes and phases using Euler's formula (same thing can be done in
% Matlab directly using abs() and angle() functions respectively)
magnitudes = sqrt(real(X).^2 + imag(X).^2); 
phases = atan2(imag(X), real(X)); 

% plot real and imaginary components (weights for cos and sin respectively)
c = real(X); 
s = imag(X); 

f = figure('color', 'w', 'pos', [363 617 1058 500]); 
pnl = panel(f); 
pnl.pack('v', 2); 

ax = pnl(1).select(); 
stem([0:N-1], c, 'color', 'k')
hold on 
stem(N/2, c(N/2+1), 'color', 'm'); 
xlim([-1, 32])
xticks([0:1:31])
ylim([min(c)-1, max(c)+1])
pnl(1).xlabel('frequency (k)')
pnl(1).ylabel('c')
ax.XAxis.Visible = 'off'; 

ax = pnl(2).select(); 
stem([0:N-1], s, 'color', 'k')
hold on 
stem(N/2, s(N/2+1), 'color', 'm'); 
xlim([-1, 32])
xticks([0:1:31])
ylim([min(s)-1, max(s)+1])
pnl(2).xlabel('frequency (k)')
pnl(2).ylabel('s')

pnl.margin = [20, 20, 5, 5]; 
pnl.fontsize = 16; 

saveas(f, fullfile(save_path, 'fft_symmetries_cartesian.svg'))
print(fullfile(save_path, 'fft_symmetries_cartesian.png'), '-dpng', '-painters', '-r600', f); 


% plot magnitude and phase 
f = figure('color', 'w', 'pos', [363 617 1058 500]); 
pnl = panel(f); 
pnl.pack('v', 2); 

ax = pnl(1).select(); 
stem([0:N-1], magnitudes, 'color', 'k')
hold on 
stem(N/2, magnitudes(N/2+1), 'color', 'm'); 
xlim([-1, 32])
xticks([0:1:31])
pnl(1).xlabel('frequency (k)')
pnl(1).ylabel('magnitude')

ax = pnl(2).select(); 
plot([0:N-1], phases, '-o', 'linew', 1, 'color', 'k')
hold on 
plot(N/2, phases(N/2+1), 'o', 'color', 'm'); 
xlim([-1, 32])
xticks([0:1:31])
yticks([-pi, 0, pi])
yticklabels({'-?', '0', '?'})
pnl(2).xlabel('frequency (k)')
pnl(2).ylabel('phase')

pnl.margin = [20, 20, 5, 5]; 
pnl.fontsize = 16; 

saveas(f, fullfile(save_path, 'fft_symmetries_polar.svg'))
print(fullfile(save_path, 'fft_symmetries_polar.png'), '-dpng', '-painters', '-r600', f); 


%% normalizing DFT? 

% To obtain the DFT weights, we take the dot product of each column in the
% DFT matrix with the signal we want to decompose/analyse. The thing is
% (wihtout going too much into linear algebra), that the columns are not
% vectors of length one (now we mean linear algebra length in the
% N-dimensional vector space where the vector lives). In order to retain
% the same scaling when doing to decomposition, each column in the DFT
% matrix should be divided by a constant N (the number fo samples). Matlab
% doesn't do that. When calling the function fft() on a signal, it uses the
% unscaled DFT matrix. However, when calling ifft() to to the weighted
% combination of cosine/sine waves in the DFT matrix in order to
% reconstruct the original signal, Matlab divides the result by N, thus
% scaling back to the correct units.
%
% In fact, to reconstruct the amplitude of sine waves comprising the input
% signal, the magnitude of each `c, s` pair should be further multiplied by
% 2. This is because the energy from each frequency is split between the
% positive and negative frequency in the DFT (see above for explanation).
% Except for the frequency 0 and N/2 of course.
% 
% In practice, nobody cares about units much...unless you're doing physics.
% So don't worry about normalizing...

N = 64; 
n = [0 : N-1]; 

% create signal that's a sine wave with Amplitude = 5. 
x = 5 * sin(2 * pi * 3 * n / N); 

% Do its FFT
X = fft(x); 

% For each frequency, take the `c, s` vector in the cartesian space, and
% calculate its magniude (remember, this is exactly the paramter A of the
% wave described by the `c, s` pair).
mX = abs(X); 

figure
stem(mX); 

% now, try the same but with proper scaling
X = fft(x) / N * 2; 
mX = abs(X); 
figure
stem(mX); 











