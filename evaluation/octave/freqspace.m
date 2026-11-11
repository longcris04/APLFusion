function [f1, f2] = freqspace(n, flag)
  % Octave shim for MATLAB's freqspace().
  if numel(n) == 1, n = [n n]; end
  f1 = ((0:n(2)-1) - floor(n(2)/2)) * (2/n(2));
  f2 = ((0:n(1)-1) - floor(n(1)/2)) * (2/n(1));
  if nargin > 1 && strcmpi(flag, 'meshgrid')
    [f1, f2] = meshgrid(f1, f2);
  end
end
