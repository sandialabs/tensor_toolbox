function [f,G] = fg(M,X,varargin)
%FG Objective and gradient function evaluation for fitting Ktensor to data.
%
%   F = fg(M,X) takes a ktensor M and a tensor X and returns the value of
%   the loss function (see loss function definitions below). The tensors
%   M and X must be the same order and size. 
%
%   [F,G] = fg(M,X) also returns the gradients (with respect to lambda and
%   each factor matrix in M) as a ktensor G.
%
%   [F,G] = fg(M,X,'IgnoreLambda',true) instead returns the gradients (with
%   respect to each factor matrix) as a cell array and ignores lambda in
%   the calculation of the gradient.
%
%   [F,G] = fg(M,X,'GradMode',K) returns the gradient with respect to the
%   Kth factor matrix as a single matrix. 
%
%   [F,G] = fg(M,X,'GradMode',K,'IgnoreLambda',true) ignores lambda in the
%   calculation of the gradient.
%
%   [F,G] = fg(K,X,'GradMode',0) returns the gradient with respect
%   to lambda.
%
%   [F,G] = fg(K,X,...,'GradVec',true, ...) vectorizes G. For a vector,
%   this is no change. For a matrix, this is equivalent to G = G(:). For a
%   cell array, this is equivalent to 
%      G = cell2mat(cellfun(@(x) x(:), G, 'UniformOutput', false))
%   For a ktensor, this is equivalent to G = tovec(G).
%
%   [F,G] = fg(K,X,...,'Mask',M) defines a mask tensor M that is 1 for
%   known values and 0 for missing values.
%
%   Loss Function Options: TBD


% FUTURE OPTIONS...
%   [F,G] = fg(K,X,...,'Type',T,...) chooses the objective function type.
%   In every case, we return F = sum(L(:)) for different choices of L:
%      'G': Gaussian, L = (X-K).^2 [Default]
%      'P': Poisson, L = K - X*log(K), assumes K >= 0
%      'LP': Log-Poisson, L = exp(K) - X*K
%      'B': Bernoulli, L = log(K+1) - X*log(K), assumes K >= 0
%      'LB': Logit-Bernoulli, L = log(1+K) - X*K
%   Here all operations are written as if they are elementwise and applied
%   to full(K).
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
params.addParameter('GradMode', -1, @(x) isscalar(x) && all(ismember(x,-1:ndims(X))));
params.addParameter('GradVec', false, @(x) isscalar(x) && islogical(x));
params.addParameter('IgnoreLambda', false, @(x) isscalar(x) && islogical(x));
params.addParameter('Mask',[]);
params.addParameter('Type', 'B', @(x) ismember(x, {'B'}));
%params.addParameter('Reg',[]);
params.parse(varargin{:});

GradMode = params.Results.GradMode;
GradVec = params.Results.GradVec;
IgnoreLambda = params.Results.IgnoreLambda;
W = params.Results.Mask;
FuncType = params.Results.Type;

% Figure out sparsity situation with X and Mask
if ~isa(X,'sptensor') 
    CalcType = 'Dense';
elseif isa(W,'sptensor')
    CalcType = 'Sparse';
else
    CalcType = 'SparseDense';
end
    

%Reg = params.Results.Reg;

% Check Reg
% if ~isempty(Reg)
%     if ~iscell(Reg)
%         error('Parameter ''Reg'' should be a cell array');
%     end
%     if size(Reg,2) ~= 3
%         error('Parameter ''Reg'' should have three columns');
%     end
%     % Check first column: Specific dimenions
%     if ~all(cellfun(@(x) isvector(x) && all(ismember(x,0:ndims(X))), Reg(:,1)))
%         error('Parameter ''Reg'' first column validation error'); 
%     end
%     % Check second column: weights
%     if ~all(cellfun(@(x) isscalar(x) && x > 0, Reg(:,2)))
%         error('Parameter ''Reg'' second column validation error'); 
%     end        
%     % Check second column: weights
%     if ~all(cellfun(@(x) ismember(x,{'L1','L2'}), Reg(:,3)))
%         error('Parameter ''Reg'' third column validation error'); 
%     end        
% end

%% SPECIAL CASES - SPARSEDENSE WITH GAUSSIAN OR POISSON - TBD



%% Pick master function
switch FuncType
    case 'B'
        objfh = @(x,m) -x.*log(m) + log(1+m);
        gradfh = @(x,m) -x./m + 1./(m+1);
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
        xvals = mask(X,W);
        mvals = mask(M,W);
        fvals = objfh(xvals, mvals);
        f = sum(fvals);
        if nargout > 1
            biggvals = gradfh(xvals, mvals);
            BigG = sptensor(biggvals, W.subs, M.size);
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



