function [fest,Gest,info,Glambda] = fg_est(M, X, subs, varargin)
%FG_EST Objective and gradient function estimation for fitting Ktensor to
%data. Estimation is done by using a few samples of the data rather than
%the full data.

%% Extract problem dimensions
d = ndims(X);
n = size(X);
r = ncomponents(M);

%% Set algorithm parameters from input or by using defaults
params = inputParser;
params.addParameter('xvals',[]);
params.addParameter('Uexplode',[]);
params.addParameter('objfh', @(x,m) (x-m).^2, @(f) isa(f,'function_handle'));
params.addParameter('gradfh', @(x,m) -2*(x-m), @(f) isa(f,'function_handle'));
params.parse(varargin{:});

%% Copy from params object
xvals  = params.Results.xvals;
Uvals  = params.Results.Uexplode;
objfh  = params.Results.objfh;
gradfh = params.Results.gradfh;

%% Extract entries specified by subs
if isempty(xvals)
    xvals = X(subs);
end

%% Extract appropriate rows of each factor matrix
if isempty(Uvals)
    Uvals = Uexplode(M.u,subs);
end

%% Calculate model values
KRfull = KRexplode(0,Uvals);
mvals = sum(bsxfun(@times,KRfull,transpose(M.lambda)),2);

%% Compute mean function value and scale
fest = mean(objfh(xvals,mvals)) * prod(n);

%% Compute gradient and scale
% This is inherently doing an MTTKRP, but we avoid forming the G tensor or
% calling the MTTKRP function. THis is probably inefficient.
gvals = gradfh(xvals,mvals);

Glambda = transpose(sum(bsxfun(@times,KRfull,gvals),1));

Gest = cell(d,1);
for k = 1:d
    gm = zeros(n(k),r);
    tmp2 = bsxfun(@times,KRexplode(k,Uvals),M.lambda.');
    gmvals = bsxfun(@times,tmp2,gvals);
    msubs = subs(:,k);
    for r = 1:ncomponents(M)
        gm(:,r) = accumarray(msubs,gmvals(:,r),[n(k),1]);
    end
    Gest{k} = gm / size(subs,1) * prod(n);
end

%% Output potentially reused values
info.xvals = xvals;
info.Uexplode = Uvals;

end

function KRvals = KRexplode(k,Uvals)
d = length(Uvals);

if k == 0    % Return full Khatri-Rao
    KRvals = Uvals{1};
    for kp = 2:d
        KRvals = KRvals .* Uvals{kp};
    end
else         % Return Khatri-Rao exluding the kth matrix
    KRvals = ones(size(Uvals{1}));
    for kp = [1:k-1 k+1:d]
        KRvals = KRvals .* Uvals{kp};
    end
end

end

function Uvals = Uexplode(U,subs)
d = length(U);

Uvals = cell(d,1);
for k = 1:d
    Uvals{k} = U{k}(subs(:,k),:);
end

end