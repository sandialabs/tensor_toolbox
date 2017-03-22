function G = tt_coregen(subsz, suberr)
%TT_COREGEN Generate tensor with special error properties.
%
%   G = TT_COREGEN(SZ,ERR) creates a tensor G so that the I-th subblock
%   given by G(1:SZ(I,1), 1:SZ(I,2), ..., 1:SZ(I,D)) has norm 1-ERR(I). The
%   final tensor is a tensor of size SZ(end,:) with norm 1. There can be
%   any number of subblocks of strictly increasing size.

%% Check inputs

% --- Check sizes ---

% Must be a matrix
if ~ismatrix(subsz)
    error('First argument must be a matrix');
end

% Extract size: D = # dimenions, L = # levels
[D,L] = size(subsz);

% Check strictly increasing size
for i = 2:L
    if isequal(subsz(i,:), subsz(i-1,:)) || any(subsz(i,:) < subsz(i-1))
        error('Must be strictly increasing in size');
    end
end

% Final size
gsz = subsz(end,:);

% --- Check errors ---

% Check errors are okay
if sum(suberr) > 1
    error('Errors for each chunk must sum to less than one');
end

for i = 2:L
    if suberr(i) > suberr(i-1)
        error('Errors much be strictly decreasing');
    end
end

%% Create tensor

% Figure out norm of each delta-tensor
subnrm = zeros(L,1);
for i = 1:L
    subnrm(i) = 1 - sum(subnrm(1:i-1)) - suberr(i).^2;
end

G = tensor(@zeros,gsz);

for i = 1:L
    % Figure out delta-pattern
    if i > 1
        zerosrange = onesrange;
    end    
    onesrange = cell(D,1);
    for k = 1:D
        onesrange{k} = 1:subsz(i,k);
    end
    pattern = tensor(@zeros,gsz);
    pattern(onesrange{:}) = 1;
    if (i > 1)
        pattern(zerosrange{:}) = 0;
    end
    
    % Randomly fill delta-pattern
    delta = tensor(@(sz) sign(randn(sz)), gsz);
    delta = delta .* pattern;
    sse = collapse(delta.^2);
    delta = sqrt(subnrm(i)/sse) .* delta;
    G = G + delta;
end
