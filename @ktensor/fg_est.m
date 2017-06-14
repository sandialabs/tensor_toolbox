function [fest,Gest,info] = fg_est(M,X,subs,varargin)
%FG_EST Objective and gradient function estimation for fitting Ktensor to
%data. Estimation is done by using a few samples of the data rather than
%the full data.

%% Process inputs

sz = size(M);
nd = ndims(M);

params = inputParser;
params.addParameter('GradMode', -1, @(x) isscalar(x) && all(ismember(x,-1:ndims(X))));
params.addParameter('GradVec', false, @(x) isscalar(x) && islogical(x));
params.addParameter('IgnoreLambda', false, @(x) isscalar(x) && islogical(x));
params.addParameter('xvals',[]);
params.addParameter('Uexplode',[]);
params.addParameter('objfh', @(x,m) (x-m).^2, @(f) isa(f,'function_handle'));
params.addParameter('gradfh', @(x,m) -2*(x-m), @(f) isa(f,'function_handle'));
params.parse(varargin{:});

GradMode = params.Results.GradMode;
GradVec  = params.Results.GradVec;
IgnoreLambda = params.Results.IgnoreLambda;
Xvals  = params.Results.xvals;
Uvals  = params.Results.Uexplode;
objfh  = params.Results.objfh;
gradfh = params.Results.gradfh;

%% Extract entries of X and rows of factor matrices specified by subs
if isempty(Xvals)
    Xvals = X(subs);
end

if isempty(Uvals)
    Uvals = explode(M.u,subs);
end

%% Calculate model values and function value estimate
KRfull = khatrirao_explode(Uvals);
Mvals = sum(bsxfun(@times,KRfull,transpose(M.lambda)),2);

Fvals = objfh(Xvals,Mvals);
fest  = mean(Fvals,1) * prod(sz);

%% QUIT IF ONLY NEED FUNCTION EVAL
if nargout <= 1
    return;
end

%% Gradient calculation
BigGvals = gradfh(Xvals,Mvals);

% Gradient wrt Lambda
if GradMode == 0 || (GradMode == -1 && ~IgnoreLambda)
    GradLambda = transpose(sum(bsxfun(@times,KRfull,BigGvals),1));
end

% Gradient wrt U's
GradU = cell(nd,1);
for k = 1:nd
    if GradMode == k || GradMode == -1
        if IgnoreLambda
            GradU{k} = mttkrp_explode(BigGvals,Uvals,ones(length(M.lambda),1),k,sz,subs) / size(subs,1) * prod(sz);
        else
            GradU{k} = mttkrp_explode(BigGvals,Uvals,M.lambda,k,sz,subs) / size(subs,1) * prod(sz);
        end
    end
end

%% Assemble gradient
if GradMode == 0
    Gest = GradLambda;
elseif GradMode > 0
    Gest = GradU{GradMode};
    if GradVec
        Gest = Gest(:);
    end
elseif GradMode == -1
    if IgnoreLambda
        Gest = GradU;
        if GradVec
            Gest = cell2mat(cellfun(@(x) x(:), Gest, 'UniformOutput', false));
        end
    else
        Gest = ktensor(GradLambda, GradU);
        if GradVec
            Gest = tovec(Gest);
        end
    end
end

%% Output potentially reused values
info.xvals = Xvals;
info.Uexplode = Uvals;

end

%% Exploded utilities
% Functions for working with exploded values.

% Explode cell array of matrices at the indices specified by subs
% Satisfies:
%   Cvals{i}(j,:) = C{i}(subs(j,i),:);
% Namely, the jth row of Cvals{i} is the row of C{i} corresponding to the
% jth sample specified by subs.
function Cvals = explode(C,subs)

Cvals = cell(length(C),1);
for i = 1:length(C)
    Cvals{i} = C{i}(subs(:,i),:);
end

end

% Calculate values of exploded Khatri-Rao product from exploded values
% Satisfies:
%   Pvals(j,:) = Row of khatrirao(C) corresponding to subs(j,:)
% Namely, the jth row of Pvals is the row of khatrirao(C) that corresponds
% to the jth sample specified by subs.
function Pvals = khatrirao_explode(Cvals)

Pvals = Cvals{1};
for i = 2:length(Cvals)
    Pvals = Pvals .* Cvals{i};
end

end

% Calculate MTTKRP from exploded values. This is probably inefficient.
% Satisfies
%   V = mttrkp(X,M,n)
% where X has Xvals at subs and is zero elsewhere and M = ktensor(lambda,U)
% is the ktensor associated with Uvals and lambda.
function V = mttkrp_explode(Xvals,Uvals,lambda,n,sz,subs)

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