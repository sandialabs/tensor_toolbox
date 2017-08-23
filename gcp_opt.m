function [M,info] = gcp_opt(X,r,varargin)
%GCP_OPT Fits a Generalized CP model to a tensor via optimization.
%
%   P = GCP_OPT(X,R) computes an estimate of the best rank-R Generalized
%   CP model of a tensor X using optimization. The input X can be a tensor,
%   sptensor, ktensor, or ttensor. The result P is a ktensor.
%
%   P = GCP_OPT(X,R,'param',value,...) specifies optional parameters and
%   values. Valid parameters and their default values are:
%   -- GCP Parameters --
%   'mask'     - Tensor marking missingness (0 = missing, 1 = present) {[]}
%   'objfh'    - Loss function {@(x,m) (x-m).^2}
%   'gradfh'   - Gradient of loss function {@(x,m) -2*(x-m)}
%   'lowbound' - Low bound constraint {-Inf}
%   -- Optimization Parameters --
%   'init'        - Initial guess [{'random'}|cell array]
%   'maxIts'      - Maximum iterations {100}
%   'maxTotalIts' - Maximum iterations including linesearch {5000}
%   -- Reporting --
%   'verbosity' - Verbosity level {11}
%   'printitn'  - Number of iterations per print {1}
%
%   [P,out] = GCP_SGD(...) also returns a structure with additional
%   optimization information.
%
%   Examples:
%   X = sptenrand([5 4 3], 10);
%   P = gcp_opt(X,2);
%   P = gcp_opt(X,2, ...
%               'objfh',@(x,m) log(m+1)-x.*log(m+1e-7), ...
%               'gradfh',@(x,m) 1./(m+1)-x./(m+1e-7), ...
%               'lowbound',1e-6);
%
%   See also KTENSOR, TENSOR, SPTENSOR, TTENSOR, GCP_SGD, CP_ALS, CP_OPT,
%   CP_WOPT.
%
%MATLAB Tensor Toolbox.
%Copyright 2015, Sandia Corporation.

% This is the MATLAB Tensor Toolbox by T. Kolda, B. Bader, and others.
% http://www.sandia.gov/~tgkolda/TensorToolbox.
% Copyright (2015) Sandia Corporation. Under the terms of Contract
% DE-AC04-94AL85000, there is a non-exclusive license for use of this
% work by or on behalf of the U.S. Government. Export of this data may
% require a license from the United States Government.
% The full license terms can be found in the file LICENSE.txt

%% Extract number of dimensions and norm of X.
n  = ndims(X);
sz = size(X);

%% Set algorithm parameters from input or by using defaults
isfunc    = @(f) isa(f,'function_handle');

params = inputParser;
% -- GCP Parameters --
params.addParameter('mask', [], @(mask) isa(mask,'sptensor') || isa(mask,'tensor'));
params.addParameter('objfh', @(x,m) (x-m).^2, isfunc);
params.addParameter('gradfh', @(x,m) -2*(x-m), isfunc);
params.addParameter('lowbound', -Inf, @isnumeric);
% -- Optimization Parameters --
params.addParameter('init', 'random', @(init) iscell(init) || strcmp(init,'random'));
params.addParameter('maxIts', 100);
params.addParameter('maxTotalIts', 5000);
% -- Reporting --
params.addParameter('verbosity', 11);
params.addParameter('printitn', 1);

params.parse(varargin{:});

%% Copy from params object
% -- GCP Parameters --
mask        = params.Results.mask;
objfh       = params.Results.objfh;
gradfh      = params.Results.gradfh;
lowbound    = params.Results.lowbound;
% -- Optimization Parameters --
init        = params.Results.init;
maxIts      = params.Results.maxIts;
maxTotalIts = params.Results.maxTotalIts;
% -- Reporting --
verbosity   = params.Results.verbosity;
printitn    = params.Results.printitn;

%% Welcome
if verbosity > 10
    fprintf('\n-----\nWelcome to GCP-OPT\n\n');
end

%% Create initialization (and normalize if needed)
if iscell(init)
    Uinit = init;
    M0 = ktensor(Uinit);
elseif strcmp(init,'random')
    Uinit = cell(n,1);
    for k = 1:n
        Uinit{k} = rand(sz(k),r);
    end
    M0 = ktensor(Uinit);
    
    % Normalize
    M0 = M0 * (norm_est(X,mask)/norm(M0));
    M0 = normalize(M0,0);
end

%% Run optimization
% Setup GCP problems
fcn = @(x) fg(update(M0,1:n,x),X, ...
    'objfh',objfh,'gradfh',gradfh,'IgnoreLambda',true,'GradVec',true, ...
    'Weights',mask);
lower = lowbound*ones(sum(sz) * r,1);
upper = inf(sum(sz) * r,1);

% Setup optimization
opts.x0          = tovec(M0,false);
opts.maxIts      = maxIts;
opts.maxTotalIts = maxTotalIts;
opts.printEvery  = printitn;

[x,~,info] = lbfgsb(fcn, lower, upper, opts);
M = update(M0,1:n,x);

%% Clean up final result
% Save final tensor before normalization
info.M_preclean = M;

% Arrange the final tensor so that the columns are normalized.
M = arrange(M);
% Fix the signs
M = fixsigns(M);

%% Wrap up
if verbosity > 10
    fprintf('Goodbye!\n-----\n');
end

end

%% General utilities
function normX = norm_est(X,mask)
%NORM_EST Estimate the norm of a tensor that can have missingness.

sz = size(X);

if isempty(mask)
    normX = norm(X);
else
    if isa(mask,'sptensor')
        normX = sqrt( sum(X(mask.subs).^2) / nnz(mask) * prod(sz) );
    else % Avoid forming sptensor when the mask is dense (still inefficient)
        maskmat = logical(double(mask)); Xmat = double(X);
        normX = sqrt( sum(Xmat(maskmat).^2) / nnz(mask) * prod(sz) );
    end
end

end

%% Function and gradient calculation
function [f,G] = fg(M,X,varargin)
%FG Objective and gradient function evaluation for fitting Ktensor to data.
%
%   [F,G] = fg(M,X) takes a ktensor M and a data tensor X and returns the
%   value of the scalar loss function (see loss function definitions below)
%   as F and the gradient with respect to the loss function as G. The
%   gradient is stored as a ktensor unless modified per the optional
%   parameters below. The tensors M and X must be the same order and size.
%
%   [F,G] = fg(M,X,'parameter',value,...) also accepts parameter value
%   pairs as follows:
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

%% Process inputs

sz = size(M);
nd = ndims(M);

params = inputParser;
params.addParameter('GradMode', -1, @(x) isscalar(x) && all(ismember(x,-1:ndims(X))));
params.addParameter('GradVec', false, @(x) isscalar(x) && islogical(x));
params.addParameter('IgnoreLambda', false, @(x) isscalar(x) && islogical(x));
params.addParameter('objfh', @(x,m) (x-m).^2, @(f) isa(f,'function_handle'));
params.addParameter('gradfh', @(x,m) -2*(x-m), @(f) isa(f,'function_handle'));
params.addParameter('Weights',[]);
params.parse(varargin{:});

GradMode = params.Results.GradMode;
GradVec  = params.Results.GradVec;
IgnoreLambda = params.Results.IgnoreLambda;
objfh  = params.Results.objfh;
gradfh = params.Results.gradfh;
W      = params.Results.Weights;

%% Use fg_est if weights are sparse
if isa(W,'sptensor')
    subs = find(W);
    if nargout <= 1
        f = fg_est(M,X,subs,'GradMode',GradMode,'GradVec',GradVec, ...
            'IgnoreLambda',IgnoreLambda,'objfh',objfh,'gradfh',gradfh, ...
            'wvals',W.vals);
        f = f * size(subs,1) / prod(sz);
        return;
    else
        [f,G] = fg_est(M,X,subs,'GradMode',GradMode,'GradVec',GradVec, ...
            'IgnoreLambda',IgnoreLambda,'objfh',objfh,'gradfh',gradfh, ...
            'wvals',W.vals);
        f = f * size(subs,1) / prod(sz);
        
        if isa(G,'ktensor')
            G.lambda = G.lambda * size(subs,1) / prod(sz);
            G.u = cellfun(@(u) u * size(subs,1) / prod(sz),G.u,'UniformOutput',false);
        elseif iscell(G)
            G = cellfun(@(u) u * size(subs,1) / prod(sz),G,'UniformOutput',false);
        else
            G = G * size(subs,1) / prod(sz);
        end
        return;
    end
end

%% Calculate function value
Mfull = full(M);
F = tenfun(objfh, X, Mfull);
if ~isempty(W)
    F = W.*F;
end
f = collapse(F);

%% QUIT IF ONLY NEED FUNCTION EVAL
if nargout <= 1
    return;
end

%% Gradient calculation
BigG = tenfun(gradfh,X,Mfull);
if ~isempty(W)
    BigG = W.*BigG;
end

% Gradient wrt Lambda
if GradMode == 0 || (GradMode == -1 && ~IgnoreLambda)
    KRfull = khatrirao(M.u,'r');
    GradLambda = KRfull'*BigG(:);
end

% Gradient wrt U's
GradU = cell(nd,1);
for k = 1:nd
    if GradMode == k || GradMode == -1
        if IgnoreLambda
            GradU{k} = mttkrp(BigG,M.u,k);
        else
            GradU{k} = mttkrp(BigG,M,k);
        end
    end
end

%% Assemble gradient
if GradMode == 0
    G = GradLambda;
elseif GradMode > 0
    G = GradU{GradMode};
    if GradVec
        G = G(:);
    end
elseif GradMode == -1
    if IgnoreLambda
        G = GradU;
        if GradVec
            G = cell2mat(cellfun(@(x) x(:), G, 'UniformOutput', false));
        end
    else
        G = ktensor(GradLambda, GradU);
        if GradVec
            G = tovec(G);
        end
    end
end

end