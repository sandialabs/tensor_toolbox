function [fest,Gest,info,Glambda] = fg_est(M, X, subs, varargin)
%FG_EST Objective and gradient function estimation for fitting Ktensor to
%data. Estimation is done by using a few samples of the data rather than
%the full data.

%% Extract problem dimensions
d = ndims(X);
n = size(X);

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
    Uvals = explode(M.u,subs);
end

%% Calculate model values
KRfull = khatrirao_explode(Uvals);
mvals = sum(bsxfun(@times,KRfull,transpose(M.lambda)),2);

%% Compute mean function value and scale
fest = mean(objfh(xvals,mvals)) * prod(n);

%% Compute gradient and scale
gvals = gradfh(xvals,mvals);

Glambda = transpose(sum(bsxfun(@times,KRfull,gvals),1));

Gest = cell(d,1);
for k = 1:d
    Gest{k} = mttkrp_explode(gvals,Uvals,M.lambda,k,n,subs) / size(subs,1) * prod(n);
end

%% Output potentially reused values
info.xvals = xvals;
info.Uexplode = Uvals;

end

%% Utility functions

% This is inherently doing an MTTKRP, but we avoid forming the G tensor or
% calling the MTTKRP function. THis is probably inefficient.
function V = mttkrp_explode(Xvals,Uvals,lambda,n,sz,subs)

r = size(Uvals{1},2);

tmp2 = bsxfun(@times,khatrirao_explode(Uvals([1:n-1,n+1:end])),transpose(lambda));
gmvals = bsxfun(@times,tmp2,Xvals);
msubs = subs(:,n);
V = zeros(sz(n),r);
for rp = 1:r
    V(:,rp) = accumarray(msubs,gmvals(:,rp),[sz(n),1]);
end

end

function Pvals = khatrirao_explode(Cvals)

Pvals = Cvals{1};
for i = 2:length(Cvals)
    Pvals = Pvals .* Cvals{i};
end

end

function Cvals = explode(C,subs)

Cvals = cell(length(C),1);
for i = 1:length(C)
    Cvals{i} = C{i}(subs(:,i),:);
end

end