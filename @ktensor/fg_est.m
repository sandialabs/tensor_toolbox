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
xvals    = params.Results.xvals;
Uexplode = params.Results.Uexplode;
objfh    = params.Results.objfh;
gradfh   = params.Results.gradfh;

%% Extract entries specified by subs
if isempty(xvals)
    xvals = X(subs);
end

%% Extract appropriate rows of each factor matrix
if isempty(Uexplode)
    Uexplode = cell(d,1);
    for k = 1:d
        Uexplode{k} = M.u{k}(subs(:,k),:);
    end
end

%% Calculate model values
tmp = Uexplode{1};
for k = 2:d
    tmp = tmp .* Uexplode{k};
end
tmp2 = bsxfun(@times,tmp,M.lambda.');
mvals = sum(tmp2,2);

%% Compute mean function value and scale
fest = mean(objfh(xvals,mvals)) * prod(n);

%% Compute gradient and scale
% This is inherently doing an MTTKRP, but we avoid forming the G tensor or
% calling the MTTKRP function. THis is probably inefficient and also
% ignores the lambdas (i.e., assumes they are one).
gvals = gradfh(xvals,mvals);

KRexplode = cell(size(M.u));
for k = 1:d
    KRexplode{k} = ones(size(Uexplode{1}));
end
for k = 1:d
    for kalt = 1:d
        if k ~= kalt
            KRexplode{kalt} = KRexplode{kalt} .* Uexplode{k};
        end
    end
end

Gest = cell(d,1);
for k = 1:d
    gm = zeros(n(k),r);
    tmp = bsxfun(@times,KRexplode{k},M.lambda.');
    gmvals = bsxfun(@times,tmp,gvals);
    msubs = subs(:,k);
    for r = 1:ncomponents(M)
        gm(:,r) = accumarray(msubs,gmvals(:,r),[n(k),1]);
    end
    Gest{k} = gm / size(subs,1) * prod(n);
end

tmp2 = bsxfun(@times,tmp,gvals);
Glambda = sum(tmp2,1).';

%% Output potentially reused values
info.xvals = xvals;
info.Uexplode = Uexplode;