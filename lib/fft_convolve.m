function y = fft_convolve(x, h)
%FFT_CONVOLVE Convolution using FFT
%
%   y = FFT_CONVOLVE(x, h)
%
%   Computes the convolution of two input signals x and h using the
%   Fast Fourier Transform (FFT) method.
%
%   INPUTS:
%       x - first input signal (vector)
%       h - second input signal (vector)
%
%   OUTPUT:
%       y - convolution result (same as conv(x, h))
%
%   This function performs linear convolution efficiently by
%   using FFT-based multiplication in the frequency domain.

    % Ensure column vectors
    x = x(:);
    h = h(:);

    % Length of the resulting convolution
%     N = length(x) + length(h) - 1;
    N = length(x); 

    % Compute FFT length (next power of 2 for speed)
    Nfft = 2^nextpow2(N);

    % Compute FFTs of both signals (zero-padded)
    X = fft(x, Nfft);
    H = fft(h, Nfft);

    % Multiply in the frequency domain
    Y = X .* H;

    % Inverse FFT to get time-domain result
    y = ifft(Y, Nfft);

    % Truncate to correct length
    y = y(1:N);

    % Ensure real output if inputs are real
    if isreal(x) && isreal(h)
        y = real(y);
    end
end