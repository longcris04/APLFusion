function nfmi = fmi(ima, imb, imf, feature, w)
% FMI  Feature mutual information -- memoised copy of metrics/fmi.m (Octave speed-up).
%
%   Same signature and same result as metrics/fmi.m, bit for bit.  The
%   original evaluates, for every pixel, a per-window block of code that only
%   depends on the contents of the (2w+1)x(2w+1) windows of the two feature
%   maps it compares.  Here that block -- copied verbatim from metrics/fmi.m
%   into fmi_block_a / fmi_block_b below -- is evaluated once per distinct
%   window pair instead of once per pixel, and the per-pixel map is rebuilt by
%   lookup, so fmi_map, and its mean2, are the same numbers.
%
%   For the 'edge' feature with w = 3 the windows are binary 3x3 patterns, so
%   the block has at most 2^18 distinct inputs: they are tabulated once
%   (fmi_edge3_table.mat, built by fmi_build_table) and every call is a lookup.
%   The two blocks live in private/fmi_block_a.m and private/fmi_block_b.m.
%   Any other feature / window size falls back to per-call memoisation.
%
%   Note the original's order dependence, kept here: the a-block normalises
%   fSub in place, and the b-block then sees that normalised fSub.


% FMI calculates the Feature Mutual Information (FMI), the non-reference 
% performance metric for fusion algorithms, proposed in:
% 
% M.B.A. Haghighat, A. Aghagolzadeh, H. Seyedarabi, "A Non-Reference Image 
% Fusion Metric Based on Mutual Information of Image Features," Computers 
% and Electrical Engineering, vol. 37, no. 5, pp. 744-756, Sept. 2011.
% http://dx.doi.org/10.1016/j.compeleceng.2011.07.012
% 
% 
% This code is the implementation of FAST-FMI, presented in:
% 
% M. Haghighat, M.A. Razian, "Fast-FMI: non-reference image fusion metric,"
% 8th International Conference on Application of Information and 
% Communication Technologies (AICT), pp. 1-3, 2014.
% 
% 
% Inputs:
% 		ima 	: 	First source image
% 		imb 	: 	Second source image
% 		imf 	: 	Fused image
% 		feature : 	Feature extraction method: gradient, edge, dct, wavelet, none (raw pixels) (default: image with no feature extraction)
% 		w       : 	Sliding window size w by w (default: 3)
% 
% Output:
% 		nfmi 	: 	Normalized Feature Mutual Information
% 
% 
% Sample use:
% nfmi = fmi(ima,imb,imf,'none',3);
% % Or simply:
% nfmi = fmi(ima,imb,imf);
% 
% 
% (C)	Mohammad Haghighat, University of Miami
%       haghighat@ieee.org
%       IF YOU USE THIS CODE IN YOUR WORK, PLEASE CITE THE ABOVE PAPERS.



if nargin < 3               % Check correct number of arguments
    error('There should be three input images (2 source images and 1 fused image)!');
end

if size(ima) ~= size(imb)	% Check if the source images are of the same size
    error('Size of the source images must be the same!');
end

if size(ima) ~= size(imf)	% Check if the source images are of the same size
    error('Size of the source and fused images must be the same!');
end

if ~exist('feature', 'var')
    feature = 'edge';       % Default feature extraction
end

if ~exist('w', 'var')
    w = 3;                  % Default window size
end

ima = double(ima);
imb = double(imb);
imf = double(imf);


% Feature Extraction

switch feature
    
	case 'none'         % Raw pixels (no feature extraction)              
        aFeature = ima;
        bFeature = imb;
        fFeature = imf;
        
    case 'gradient'     % Gradient
        aFeature = gradient(ima);
        bFeature = gradient(imb);
        fFeature = gradient(imf);
        
    case 'edge'         % Edge
        aFeature = edge(ima);
        bFeature = edge(imb);
        fFeature = edge(imf);
        
    case 'dct'          % DCT
        aFeature = dct2(ima);
        bFeature = dct2(imb);
        fFeature = dct2(imf);
        
    case 'wavelet'      % Discrete Meyer wavelet
        [cA,cH,cV,cD] = dwt2(ima,'dmey');
        aFeature = rerange([cA,cH;cV,cD]);
        [cA,cH,cV,cD] = dwt2(imb,'dmey');
        bFeature = rerange([cA,cH;cV,cD]);
        [cA,cH,cV,cD] = dwt2(imf,'dmey');
        fFeature = rerange([cA,cH;cV,cD]);
       
    otherwise
        error('Please specify a feature extraction method among ''gradient'', ''edge'', ''dct'', ''wavelet'', or ''none'' (raw pixels)!');
        
end


% Sliding window

[m,n] = size(aFeature);

w = floor(w/2);
s = 2*w+1;
k = s*s;
A = im2col(double(aFeature), [s s], 'sliding');
B = im2col(double(bFeature), [s s], 'sliding');
Fw = im2col(double(fFeature), [s s], 'sliding');

if k == 9 && islogical(aFeature) && islogical(bFeature) && islogical(fFeature)
    [Ta, Tf, Tb] = fmi_table();
    pw = 2.^(0:k-1);
    keyAF = (pw*A + 512*(pw*Fw) + 1).';        % a-window bits, then f-window bits (column)
    fmi_af = Ta(keyAF);
    fAfter = double(Tf(keyAF));                 % f-window after the a-block (9-bit code)
    keyBF = (pw*B).' + 512*fAfter(:) + 1;
    fmi_bf = Tb(keyBF);
else
    [kAF, ~, jAF] = unique([A; Fw].', 'rows');
    vAF = zeros(size(kAF, 1), 1);
    fA = zeros(size(kAF, 1), k);
    for u = 1:size(kAF, 1)
        aSub = reshape(kAF(u, 1:k), s, s);
        fSub = reshape(kAF(u, k+1:end), s, s);
        if islogical(aFeature), aSub = logical(aSub); end
        if islogical(fFeature), fSub = logical(fSub); end
        [vAF(u), fSub] = fmi_block_a(aSub, fSub, w);
        fA(u, :) = double(fSub(:)).';
    end
    fmi_af = vAF(jAF);
    [kBF, ~, jBF] = unique([B; fA(jAF, :).'].', 'rows');
    vBF = zeros(size(kBF, 1), 1);
    for u = 1:size(kBF, 1)
        bSub = reshape(kBF(u, 1:k), s, s);
        fSub = reshape(kBF(u, k+1:end), s, s);
        if islogical(bFeature), bSub = logical(bSub); end
        vBF(u) = fmi_block_b(bSub, fSub, w);
    end
    fmi_bf = vBF(jBF);
end

fmi_map = reshape((fmi_af(:) + fmi_bf(:))./2, m-2*w, n-2*w);
nfmi = mean2(fmi_map);
end


function [Ta, Tf, Tb] = fmi_table()
% Ta(key), Tb(key): fmi_af / fmi_bf of the binary window pair encoded in key;
% Tf(key): the f-window left by the a-block, as a 9-bit code.
persistent ta tf tb
if isempty(ta)
    here = fileparts(mfilename('fullpath'));
    file = fullfile(here, 'fmi_edge3_table.mat');
    if ~exist(file, 'file')
        error('fmi: %s missing -- run fmi_build_table first', file);
    end
    d = load(file);
    ta = d.Ta; tf = d.Tf; tb = d.Tb;
end
Ta = ta; Tf = tf; Tb = tb;
end


function imNorm = rerange(im)

% RERANGE changes the range of images into interval [0,1]

im = double(im);

[m,n] = size(im);

imMax = max(im(:));
imMin = min(im(:));

if imMax == imMin
    imNorm = ones(m,n);
else
    imNorm = (im - imMin)/(imMax - imMin);
end

end
