function aplfusion_metrics(list_file, out_file)
% APLFUSION_METRICS  The eight fusion metrics of the APLFusion paper, one CSV row per image.
%
%   aplfusion_metrics(LIST_FILE, OUT_FILE)
%
%   LIST_FILE  text file, one image per line, tab separated:
%                  name <TAB> source A <TAB> MRI <TAB> fused grey image
%              source A as stored: CT, or the colour PET / SPECT image.
%   OUT_FILE   CSV with the columns Name, SD, Qabf, FMI, Qs, Qw, Qe, SSIM, Qcb.
%
%   Per image: source A goes through rgb2gray when it has three channels and the MRI through
%   ConvertRGBtoYUV (its Y channel); the three images are then converted to uint8 and passed,
%   in the order (A, MRI, fused), to
%
%     SD    standard deviation of the fused image (on [0, 1])
%     Qabf  Qabf.m             gradient-based metric of Xydeas and Petrovic
%     FMI   fmi(.., 'edge', 3) feature mutual information of the edge maps, 3x3 windows
%     Qs    metricPeilla(.., 1)  Piella's fusion quality index (Q_S in Liu et al., TPAMI 2012)
%     Qw    metricPeilla(.., 2)  weighted fusion quality index
%     Qe    metricPeilla(.., 3)  edge-dependent index (Qw of the Prewitt edge images, squared)
%     SSIM  MS_SSIM.m          (SSIM(A, fused) + SSIM(MRI, fused)) / 2, single scale
%     Qcb   metricChenBlum.m   Chen-Blum metric
%
%   A metric that fails on an image gives NaN for that image (and a message on stderr).
%   Under GNU Octave, the folder octave/ is searched first: it supplies freqspace, which
%   Octave lacks, and faster copies of Qabf and fmi that return the same values bit for bit.
%   Octave needs the Octave Forge image package (pkg install -forge image).

  here = fileparts(mfilename('fullpath'));
  addpath(fullfile(here, 'metrics'));
  if exist('OCTAVE_VERSION', 'builtin')
    addpath(fullfile(here, 'octave'));   % added last: searched first
    pkg load image
    warning('off', 'all');
  end

  fid = fopen(list_file, 'r');
  if fid < 0
    error('aplfusion_metrics: cannot open %s', list_file);
  end
  rows = {};
  while true
    line = fgetl(fid);
    if ~ischar(line), break; end
    if isempty(strtrim(line)), continue; end
    rows{end+1} = strsplit(line, sprintf('\t'));  %#ok<AGROW>
  end
  fclose(fid);

  tmp_file = [out_file '.part'];
  out = fopen(tmp_file, 'w');
  fprintf(out, 'Name,SD,Qabf,FMI,Qs,Qw,Qe,SSIM,Qcb\n');
  for k = 1:numel(rows)
    r = rows{k};
    name = r{1};
    A = im2double(imread(r{2}));
    B = im2double(imread(r{3}));
    F = im2double(imread(r{4}));
    if size(A, 3) > 1
      A = rgb2gray(A);
    end
    B_YUV = ConvertRGBtoYUV(B);
    B = B_YUV(:, :, 1);

    im1 = im2uint8(A);
    im2 = im2uint8(B);
    imf = im2uint8(F);

    y = zeros(1, 8);
    y(1) = safe(@() sqrt(var(F(:), 1)), name, 'SD');
    y(2) = safe(@() Qabf(im1, im2, imf), name, 'Qabf');
    y(3) = safe(@() fmi(im1, im2, imf, 'edge', 3), name, 'FMI');
    y(4) = safe(@() metricPeilla(im1, im2, imf, 1), name, 'Qs');
    y(5) = safe(@() metricPeilla(im1, im2, imf, 2), name, 'Qw');
    y(6) = safe(@() metricPeilla(im1, im2, imf, 3), name, 'Qe');
    y(7) = safe(@() MS_SSIM(im1, im2, imf), name, 'SSIM');
    y(8) = safe(@() metricChenBlum(im1, im2, imf), name, 'Qcb');

    fprintf(out, '%s', name);
    fprintf(out, ',%.10g', y);
    fprintf(out, '\n');
  end
  fclose(out);
  movefile(tmp_file, out_file, 'f');
end


function value = safe(fn, name, label)
  try
    value = double(fn());
    value = value(1);
  catch err
    fprintf(2, 'aplfusion_metrics: %s failed on %s: %s\n', label, name, err.message);
    value = NaN;
  end
end
