clear

N = 32; 
n = [0:N-1]; 
k = 3; 
omg = 2*pi*k/N; 

x = rand(1, N); 
r = rand(1, N); 



cx = cos(omg*n) * x'; 
sx = sin(omg*n) * x'; 

cr = cos(omg*n) * r'; 
sr = sin(omg*n) * r'; 

c = (cx*cr - sx*sr); 
s = (cx*sr - sx*cr); 
c * cos(omg*n) + s * sin(omg*n)




Ax = sqrt(cx^2 + sx^2); 
phx = atan2(sx, cx); 

Ar = sqrt(cr^2 + sr^2); 
phr = atan2(sr, cr); 

A = Ax * Ar; 
ph = phx + phr; 

A * cos(omg*n - ph)

