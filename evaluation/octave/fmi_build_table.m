function fmi_build_table(file, part, nparts)
% FMI_BUILD_TABLE  Tabulate the per-window blocks of metrics/fmi.m for binary 3x3 windows.
%
%   fmi_build_table()            build octave/fmi_edge3_table.mat in one process
%   fmi_build_table(FILE, P, N)  compute part P of N (keys P, P+N, P+2N, ...) into FILE.partP.mat
%   fmi_build_table(FILE, 0, N)  merge the N parts into FILE
%
%   For every pair of binary 3x3 windows (x, f) -- key = bits of x(:) + 512*bits of f(:) + 1 --
%   it stores
%     Ta(key) = fmi_af of private/fmi_block_a(x, f)      (verbatim "a" block of metrics/fmi.m)
%     Tf(key) = the f window that block leaves behind, as a 9-bit code
%     Tb(key) = fmi_bf of private/fmi_block_b(x, f)      (verbatim "b" block)
%   so that fmi.m can replace the per-pixel evaluation by a lookup.

  here = fileparts(mfilename('fullpath'));
  if nargin < 1 || isempty(file)
    file = fullfile(here, 'fmi_edge3_table.mat');
  end
  if nargin < 3
    part = 1; nparts = 1;
  end
  nkeys = 2^18;
  if part == 0
    Ta = zeros(nkeys, 1); Tf = zeros(nkeys, 1, 'uint16'); Tb = zeros(nkeys, 1);
    for p = 1:nparts
      d = load(sprintf('%s.part%d.mat', file, p));
      Ta(d.keys) = d.ta; Tf(d.keys) = d.tf; Tb(d.keys) = d.tb;
    end
    save('-v7', file, 'Ta', 'Tf', 'Tb');
    return
  end
  keys = part:nparts:nkeys;
  ta = zeros(numel(keys), 1); tf = zeros(numel(keys), 1, 'uint16'); tb = zeros(numel(keys), 1);
  pw = 2.^(0:8);
  for t = 1:numel(keys)
    code = keys(t) - 1;
    xbits = logical(bitand(mod(code, 512), pw));
    fbits = logical(bitand(floor(code / 512), pw));
    x = reshape(xbits, 3, 3);
    f = reshape(fbits, 3, 3);
    [ta(t), fAfter] = fmi_block_a(x, f, 1);
    tf(t) = uint16(sum(double(fAfter(:)).' .* pw));
    tb(t) = fmi_block_b(x, f, 1);
  end
  if nparts == 1
    Ta = ta; Tf = tf; Tb = tb;
    save('-v7', file, 'Ta', 'Tf', 'Tb');
  else
    save('-v7', sprintf('%s.part%d.mat', file, part), 'keys', 'ta', 'tf', 'tb');
  end
end
