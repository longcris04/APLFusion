function output = Qabf(pA, pB, pF)
% QABF  Vectorised copy of metrics/Qabf.m (Octave speed-up), same result bit for bit.
%
%   The original fills aA/aB/aF, GAF/GBF and the Q maps pixel by pixel in
%   interpreted loops (about 5 s per 256x256 image under Octave).  Here every
%   map is built with the same element-wise expressions over whole arrays --
%   the same IEEE operations on the same operands -- and the two final sums
%   are the original lines unchanged.

L=1; Tg=0.9994;kg=-15;Dg=0.5;Ta=0.9879;ka=-22;Da=0.8;

h1=[1 2 1;0 0 0;-1 -2 -1]; h2=[0 1 2;-1 0 1;-2 -1 0]; h3=[-1 0 1;-2 0 2;-1 0 1];

if(size(pA,3)>1)
    pA = rgb2gray(pA);
    pB = rgb2gray(pB);
    pF = rgb2gray(pF);
end

pA = double(pA);
pB = double(pB);
pF = double(pF);

SAx = conv2(pA,h3,'same'); SAy = conv2(pA,h1,'same');
gA = sqrt(SAx.^2 + SAy.^2);
aA = angle_map(SAx, SAy);

SBx = conv2(pB,h3,'same'); SBy = conv2(pB,h1,'same');
gB = sqrt(SBx.^2 + SBy.^2);
aB = angle_map(SBx, SBy);

SFx = conv2(pF,double(h3),'same'); SFy = conv2(pF,double(h1),'same');
gF = sqrt(SFx.^2 + SFy.^2);
aF = angle_map(SFx, SFy);

QAF = edge_preservation(gA, aA, gF, aF, Tg, kg, Dg, Ta, ka, Da);
QBF = edge_preservation(gB, aB, gF, aF, Tg, kg, Dg, Ta, ka, Da);

deno = sum(sum( gA + gB ));
nume = sum(sum( QAF.*gA + QBF.*gB ));
output = nume / deno;
end


function a = angle_map(Sx, Sy)
% aA(i,j) = pi/2 if SAx(i,j) == 0, else atan(SAy(i,j)/SAx(i,j))
a = atan(Sy ./ Sx);
a(Sx == 0) = pi/2;
end


function Q = edge_preservation(gX, aX, gF, aF, Tg, kg, Dg, Ta, ka, Da)
% GXF(i,j) = gF/gX if gX > gF; gF if gX == gF; gX/gF otherwise  (the original's branches)
G = gX ./ gF;
over = gX > gF;
G(over) = gF(over) ./ gX(over);
same = gX == gF;
G(same) = gF(same);
A = 1 - abs(aX-aF)/(pi/2);
Qg = Tg ./ (1 + exp(kg*( G - Dg )));
Qa = Ta ./ (1 + exp(ka*( A - Da )));
Q = Qg .* Qa;
end
