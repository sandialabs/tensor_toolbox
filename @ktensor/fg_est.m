function [fest,Gest,info] = fg_est(M,X,subs,varargin)
%FG_EST Objective and gradient function estimation for fitting Ktensor to
%data. Estimation is done by using a few samples of the data rather than
%the full data.
%
%   [Fest,Gest] = fg_est(M,X,subs) takes a ktensor M, a data tensor X and
%   a subset subs and returns the estimated value of the scalar loss
%   function (see loss function definitions below) as Fest and the
%   estimated gradient with respect to the loss function as G. The gradient
%   is stored as a ktensor unless modified per the optional parameters
%   below. The tensors M and X must be the same order and size.
%
%   [Fest,Gest] = fg(M,X,subs,'parameter',value,...) also accepts parameter
%   value pairs as follows:
%
%   'objfh' - Handle to objective function. The form should be f(x,m) where
%   x and m are scalars or vectors. In the vector case, each one should be
%   processed as if it were a scalar. Default: @(x,m) (x-m).^2
%
%   'gradfh' - Handle to gradient function. The form should be g(x,m) where
%   x and m are scalars or vectors. In the vector case, each one should be
%   processed as if it were a scalar. Default: @(x,m) -2*(x-m)
%
%   'IgnoreLambda' - True says to ignore lambda in the function and
%   gradient calculations. In this case, G is a cell array whose K-th cell
%   is the gradient in matrix form with respect to the K-th factor matrix
%   in M. Default: false.
%
%   'GradMode' - Says which gradient to compute. A value K between 1 and
%   ndims(X) says to compute the gradient with respect to the K-th factor
%   matrix; in this case, G is a matrix. A value of 0 says to compute the
%   gradient with respect to lambda; in this case, G is a vector. A value
%   of -1 says to compute all the gradients; in this case, G is either a
%   ktensor or a cell array, depending on the value of 'IgnoreLambda'.
%   Default: -1.
%
%   'GradVec' - True says to vectorize the gradient G. For a vector (e.g.,
%   'GradMode' == 0), this is no change. For a matrix (e.g., 'GradMode' ==
%   K), this is equivalent to G = G(:). For a cell array (e.g.,
%   'IgnoreLambda' == true), this is equivalent to
%      G = cell2mat(cellfun(@(x) x(:), G, 'UniformOutput', false))
%   For a ktensor, this is equivalent to G = tovec(G). Default: false.
%
%   'Weights' - Defines a weight tensor W that applies a weight to the
%   objective function value of each entry.
%
%   'xvals' - List of values of X at subs. Shortcuts evaluation of X(subs).
%   'wvals' - List of values of W at subs. Shortcuts evaluation of W(subs).
%   'Uexplode' - List of values of exploded U values at subs. Shortcuts
%                evaluation of Uvals = explode(M.u,subs);

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
params.addParameter('Weights',[]);
params.addParameter('wvals',[]);
params.parse(varargin{:});

GradMode = params.Results.GradMode;
GradVec  = params.Results.GradVec;
IgnoreLambda = params.Results.IgnoreLambda;
Xvals  = params.Results.xvals;
Uvals  = params.Results.Uexplode;
objfh  = params.Results.objfh;
gradfh = params.Results.gradfh;
W      = params.Results.Weights;
Wvals  = params.Results.wvals;

%% Extract entries of X, W and rows of factor matrices specified by subs
if isempty(Xvals)
    Xvals = X(subs);
end

if isempty(Wvals) && ~isempty(W)
    Wvals = W(subs);
end

if isempty(Uvals)
    Uvals = explode(M.u,subs);
end

%% Calculate model values and function value estimate
KRfull = khatrirao_explode(Uvals);
Mvals = sum(bsxfun(@times,KRfull,transpose(M.lambda)),2);

Fvals = objfh(Xvals,Mvals);
if ~isempty(Wvals)
    Fvals = Wvals.*Fvals;
end
fest  = mean(Fvals,1) * prod(sz);

%% QUIT IF ONLY NEED FUNCTION EVAL
if nargout <= 1
    return;
end

%% Gradient calculation
BigGvals = gradfh(Xvals,Mvals);
if ~isempty(Wvals)
    BigGvals = Wvals.*BigGvals;
end

% Gradient wrt Lambda
if GradMode == 0 || (GradMode == -1 && ~IgnoreLambda)
    GradLambda = transpose(sum(bsxfun(@times,KRfull,BigGvals),1)) / size(subs,1) * prod(sz);
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
info.wvals = Wvals;

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