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
    Gest{k} = mttkrp_explode(gvals,Uvals,M.lambda,k,subs,n) / size(subs,1) * prod(n);
end


%% Output potentially reused values
info.xvals = xvals;
info.Uexplode = Uvals;

end

%% Utility functions

% This is probably inefficient.
function V = mttkrp_explode(Xvals,Uvals,lambda,n,subs,sz)

% Get dimensions and sample indices
if n == 1
    R = size(Uvals{2},2);
else
    R = size(Uvals{1},2);
end
szn = sz(n);
subsn = subs(:,n);

% Calculate Khatri-Rao values and scale by lambda
KRvals = khatrirao_explode(Uvals([1:n-1,n+1:end]));
KRvals = bsxfun(@times,KRvals,transpose(lambda));

% Calculate summands of matrix times matrix sum
Vvals = bsxfun(@times,KRvals,Xvals);

% Sum up to get mttkrp and place in correct slots
V = zeros(szn,R);
for r = 1:R
    V(:,r) = accumarray(subsn,Vvals(:,r),[szn,1]);
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