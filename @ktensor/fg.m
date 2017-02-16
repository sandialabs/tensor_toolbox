function [f,G] = fg(M,X,varargin)
%FG Master objective function for optimization of Kruskal model.
%
%   F = fg(K,X) takes a ktensor K and a tensor X and returns the sum of
%   squares, i.e., F = sum(L(:)) where L = (X-K).^2. This assumes that K
%   and X are the same order and size.
%
%   [F,G] = fg(K,X,'GradCalc',N) returns the gradient with respect to the
%   Nth factor matrix as a single matrix (not in a cell array). We assume N
%   is in the range {1,...,ndims(K)}. The gradient with respect to the Nth
%   factor matrix, i.e., K.U{N}, is a matrix of size I x R where I =
%   size(K,N) and R = ncomponents(K). 
%
%   [F,G] = fg(K,X) or [F,G] = fg(K,X,'GradCalc',RNG) returns the
%   gradient of all the factor matrices specified by RNG as a cell array.
%   Each entry of RNG must be in the set {1,...,ndims(K)}. 
%
%   [F,G] = fg(K,X,'GradCalc',0) returns the gradient with respect
%   to the factor weights, i.e., K.lambda, as a vector of length R where R
%   = ncomponents(K).
%
%   [F,G] = fg(K,X,'GradCalc',0:N) returns the gradient of lambda and
%   all the factor matrices combined into a ktensor. 
%
%   [F,G] = fg(K,X,...,'GradVec',true, ...) vectorizes G. For a vector,
%   this is no change. For a matrix, this is equivalent to G = G(:). For a
%   cell array, this is equivalent to 
%      G = cell2mat(cellfun(@(x) x(:), G, 'UniformOutput', false)')
%   For a ktensor, this is equivalent to G = tovec(G).
%
%   [F,G] = fg(K,X,...,'Type',T,...) chooses the objective function type.
%   In every case, we return F = sum(L(:)) for different choices of L:
%      'G': Gaussian, L = (X-K).^2 [Default]
%      'P': Poisson, L = K - X*log(K), assumes K >= 0
%      'LP': Log-Poisson, L = exp(K) - X*K
%      'B': Bernoulli, L = log(K+1) - X*log(K)m assues K >= 0
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
sz = size(X);
if ~isequal(size(M),sz)
    error('Model and tensor sizes do not match');
end

params = inputParser;
params.addParameter('GradCalc', 1:ndims(X), @(x) isvector(x) && all(ismember(x,0:ndims(X))));
params.addParameter('GradVec', false, @(x) isscalar(x) && islogical(x));
params.addParameter('Type', 'B', @(x) ismember(x, {'B'}));
params.addParameter('Reg',[]);
params.parse(varargin{:});

GradCalc = sort(params.Results.GradCalc);
GradVec = params.Results.GradVec;
Type = params.Results.Type;
Reg = params.Results.Reg;

% Check Reg
if ~isempty(Reg)
    if ~iscell(Reg)
        error('Parameter ''Reg'' should be a cell array');
    end
    if size(Reg,2) ~= 3
        error('Parameter ''Reg'' should have three columns');
    end
    % Check first column: Specific dimenions
    if ~all(cellfun(@(x) isvector(x) && all(ismember(x,0:ndims(X))), Reg(:,1)))
        error('Parameter ''Reg'' first column validation error'); 
    end
    % Check second column: weights
    if ~all(cellfun(@(x) isscalar(x) && x > 0, Reg(:,2)))
        error('Parameter ''Reg'' second column validation error'); 
    end        
    % Check second column: weights
    if ~all(cellfun(@(x) ismember(x,{'L1','L2'}), Reg(:,3)))
        error('Parameter ''Reg'' third column validation error'); 
    end        
end

if numel(GradCalc) > 1
    fprintf('Only supporting single derivative computations');
end



%% B (Bernoulli)
k = GradCalc; % Extract mode for gradient calculation
F = tenfun(@(x,m) -x * log(m) + log(1+m), X, full(M));

Mfull = full(M);
%F = -X .* tenfun(@log,Mfull) + tenfun(@log,Mfull+1);
f = collapse(F);
BigG = ( -X ./ Mfull ) + ( 1 ./ (Mfull + 1) );
G = mttkrp(BigG,M.u,k);

%% PLL
% 
% U = tocell(K);
% M = full(K);
% expM = exp(M);
% f = sum(expM(:) - X(:).*M(:));
% D = expM-X;
% 
% Gcell = cell(nd,1);
% for n = 1:nd
%     Gcell{n} = mttkrp(D,U,n);
% end
% 
% % Finalize G
% if GradVec
%     Gcell = cellfun(@(x) x(:), Gcell, 'UniformOutput', false);
%     G = cell2mat(Gcell);
% else
%     G = Gcell;
% end

%%
if GradVec
    G = G(:);
end