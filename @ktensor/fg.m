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
%   'Mask' - Defines a mask tensor M that is 1 for known values and 0 for
%   missing values. The tensor M can be dense or sparse.
%
%   'Type' - Which scalar loss function to use. This should be a
%   function/gradient pair of the form f(x,m) and g(x,m) where x and m are
%   scalars or vectors. In the vector case, each one should be processed as
%   if it were a scalar. Some defaults are predefined.
%      'G': Gaussian
%           objfh = @(x,m) 0.5 * (x-m).^2;
%           gradfh = @(x,m) -x + m;
%      'B': Bernoulli/Binary, assumes m >= 0:
%           bthresh = 1e-7; 
%           objfh = @(x,m) -x.*log(m+bthresh) + log(m+1);
%           gradfh = @(x,m) -x./(m+bthresh) + 1./(m+1);
%
%   'FuncOnly' - True says to only compute the function value. This is the
%   default behavior if there is only one output argument and 'GradOnly' is
%   false. Throws an error if there is more than one output argument.
%   Default: false. 
%
%   'GradOnly' - True says to only compute the gradient value. Throws an
%   error if there is more than one output argument. Default: false.
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
%   of -1 says to compute all the gradients; in thie case, G is either a
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


% FUTURE OPTIONS...
%
%   [F,G] = fg(K,X,...,'Reg',REG,...} specifies the
%   regularization for lambda and/or any subset of factor matrices. REG is
%   a cell array that has three columns and as many rows as different types
%   of regularizers. The first column specifies what is to be regularized
%   and is a subset of {0,...,N}. The value zero indiates to regularize
%   lambda. The second column specifies the weight of the regularization
%   and can be any positive value. The third column specifies the type of
%   regularization from the following choices:
%      'L1': sum(abs(V(:)))
%      'L2': sum(V(:).^2)
%   The default is no regularization.

%% Process inputs

% Check size match between X and K
sz = size(M);
nd = ndims(M);
if ~isequal(size(X),sz)
    error('Model and tensor sizes do not match');
end

params = inputParser;
params.addParameter('FuncOnly', false, @(x) isscalar(x) && islogical(x));
params.addParameter('GradOnly', false, @(x) isscalar(x) && islogical(x));
params.addParameter('GradMode', -1, @(x) isscalar(x) && all(ismember(x,-1:ndims(X))));
params.addParameter('GradVec', false, @(x) isscalar(x) && islogical(x));
params.addParameter('IgnoreLambda', false, @(x) isscalar(x) && islogical(x));
params.addParameter('Mask',[]);
params.addParameter('Type', 'B', @(x) ismember(x, {'B','G'}));
params.parse(varargin{:});

FuncOnly = params.Results.FuncOnly;
GradOnly = params.Results.GradOnly;
GradMode = params.Results.GradMode;
GradVec = params.Results.GradVec;
IgnoreLambda = params.Results.IgnoreLambda;
W = params.Results.Mask;
FuncType = params.Results.Type;

if FuncOnly && GradOnly
    error('Cannot set both ''FuncOnly'' and ''GradOnly'' to true');
end

if (nargout > 2) || ((nargout > 1) && (FuncOnly || GradOnly))
    error('Too many output arguments');
end

if nargout == 1
    if GradOnly
        CompFunc = false;
        CompGrad = true;
    else
        CompFunc = true;
        CompGrad = false;
    end
else
    CompFunc = true;
    CompGrad = true;  
end

% Figure out sparsity situation with X and Mask
if isa(W,'sptensor')
    CalcType = 'Sparse';
elseif ~isa(X,'sptensor') 
    CalcType = 'Dense';
end
    

%% Setup objective function
switch FuncType
    case 'G' % Gaussian
        objfh = @(x,m) 0.5 * (x-m).^2;
        gradfh = @(x,m) -x + m;
    case 'B' % Bernoulli
        bthresh = 1e-7; % Really important! Can't be any smaller!
        objfh = @(x,m) -x.*log(m+bthresh) + log(m+1);
        gradfh = @(x,m) -x./(m+bthresh) + 1./(m+1);
    otherwise
        error('Unsupported function type %s', FuncType);
end


%% Calculation function value and optionally information for gradient
switch CalcType
    case 'Dense'
        Mfull = full(M);
        F = tenfun(objfh, X, Mfull);
        if ~isempty(W)
            F = W.*F;
        end
        f = collapse(F);
        if nargout > 1
            BigG = tenfun(gradfh, X, Mfull);
            if ~isempty(W)
                BigG = W.*BigG;
            end
        end
    case 'Sparse'
        %TODO: Reuse mvals for sparse MTTKRP
        xvals = mask(X,W);
        mvals = mask(M,W);
        fvals = objfh(xvals, mvals);
        f = sum(fvals);
        if nargout > 1
            biggvals = gradfh(xvals, mvals);
            BigG = sptensor(find(W), biggvals, sz);
        end
    case 'SparseDense'
    % Onle works in the case of Gaussian or Poisson
end

%% ADD REGULARIZATION TO FUNCTION - TBD

%% QUIT IF ONLY NEED FUNCTION EVAL
if nargout <= 1 
    return;
end


%% Gradient calculation - works the same whether BigG is dense or sparse

% Gradient wrt Lambda
if GradMode == 0 || (GradMode == -1 && ~IgnoreLambda)
    tmp = mttkrp(BigG, M.u, 1);
    GradLambda = sum(M.u{1} .* tmp)';
    %TBD: Add regularization
end   

% Gradient wrt U's
GradU = cell(nd,1);
for k = 1:nd
    if GradMode == k || GradMode == -1
        if IgnoreLambda
            GradU{k} = mttkrp(BigG, M.u, k);
        else
            GradU{k} = mttkrp(BigG, M, k);
        end        
        %TBD: Add regularization
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



