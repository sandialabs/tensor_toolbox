function [M,info] = gcp_sgd(X,r,varargin)
%GCP_SGD Fits a Generalized CP model to a tensor via stochastic
%optimization with adaptive moment estimation (Adam).
%
%   P = GCP_SGD(X,R) computes an estimate of the best rank-R Generalized
%   CP model of a tensor X using stochastic optimization with adaptive
%   moment estimation (Adam). The input X can be a tensor, sptensor,
%   ktensor, or ttensor. The result P is a ktensor.
%
%   P = GCP_SGD(X,R,'param',value,...) specifies optional parameters and
%   values. Valid parameters and their default values are:
%   -- GCP Parameters --
%   'mask'     - Tensor marking missingness (0 = missing, 1 = present) {[]}
%   'objfh'    - Loss function {@(x,m) (x-m).^2}
%   'gradfh'   - Gradient of loss function {@(x,m) -2*(x-m)}
%   'lowbound' - Low bound constraint {-Inf}
%   -- Stochastic Gradient Steps --
%   'init'     - Initial guess [{'random'}|cell array]
%   'gsamples' - Number of samples to calculate the gradient estimate {1}
%   'gsampler' - Entry sampler for gradient
%                [{'unif'}|'all'|'strat'|'prop'|function handle]
%   'rate'     - Step size {1e-3}
%   'adam'     - Use adaptive moment estimation {true}
%   -- Convergence Criteria --
%   'epochiters' - Number of iterations per epoch {1000}
%   'maxepochs'  - Maximum number of epochs {100}
%   'fsamples'   - Number of samples to calculate the loss function
%                  estimate {1000}
%   'fsampler'   - Entry sampler for loss function
%                  [{'unif'}|'all'|'strat'|'prop'|function handle]
%   'conv_cond'  - Convergence condition {@(f,fold) f > fold}
%   -- Additional parameters for adam --
%   'beta1'   - First moment decay {0.9}
%   'beta2'   - Second moment decay {0.999}
%   'epsilon' - Small value to help with numerics in division {1e-8}
%   -- Additional parameters for stratified sampler --
%   'prop_nz'  - Proportion of samples that should be nonzero {0.5}
%   'num_cand' - Number of candidate zeros to pull ahead of time {1e6}
%   -- Additional parameters for proportional sampler --
%   'offset' - Offset to make probabilities more balanced {1e3}
%   -- Experimental: Renormalization/Decreasing rate --
%   'renormalize' - Renormalize at each epoch. {false}
%                   If adam is used, this also restarts moment estimates
%   'dec_rate'    - Halve the step size at each epoch where the loss
%                   function value increases. {false}
%   -- Reporting --
%   'verbosity'   - Verbosity level {11}
%   'print_ftrue' - Print true loss function value at each epoch {false}
%   'save_ftrue'  - Save the true loss function value at each epoch {false}
%   'gradcheck'   - Trigger error if the gradient is ever infinite {true}
%
%   [P,out] = GCP_SGD(...) also returns a structure with the trace of the
%   function value.
%
%   Examples:
%   X = sptenrand([5 4 3], 10);
%   P = gcp_sgd(X,2);
%   P = gcp_sgd(X,2, ...
%               'objfh',@(x,m) log(m+1)-x.*log(m+1e-7), ...
%               'gradfh',@(x,m) 1./(m+1)-x./(m+1e-7), ...
%               'lowbound',1e-6);
%
%   See also KTENSOR, TENSOR, SPTENSOR, TTENSOR, GCP_OPT, CP_ALS, CP_OPT,
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
issampler = @(s) isfunc(s) || any(strcmp(s,{'unif','all','strat','prop'}));

params = inputParser;
% -- GCP Parameters --
params.addParameter('mask', [], @(mask) isa(mask,'sptensor') || isa(mask,'tensor'));
params.addParameter('objfh', @(x,m) (x-m).^2, isfunc);
params.addParameter('gradfh', @(x,m) -2*(x-m), isfunc);
params.addParameter('lowbound', -Inf, @isnumeric);
% -- Stochastic Gradient Steps --
params.addParameter('init', 'random', @(init) iscell(init) || strcmp(init,'random'));
params.addParameter('gsamples', 1);
params.addParameter('gsampler', 'unif', issampler);
params.addParameter('rate', 1e-3);
params.addParameter('adam', true, @islogical);
% -- Convergence Criteria --
params.addParameter('epochiters', 1000);
params.addParameter('maxepochs', 100);
params.addParameter('fsamples', 1000);
params.addParameter('fsampler', 'unif', issampler);
params.addParameter('conv_cond', @(f,fold) f > fold, isfunc);
% -- Additional parameters for adam --
params.addParameter('beta1', 0.9);
params.addParameter('beta2', 0.999);
params.addParameter('epsilon', 1e-8);
% -- Additional parameters for stratified sampler --
params.addParameter('prop_nz', 0.5);
params.addParameter('num_cand', 1e6);
% -- Additional parameters for proportional sampler --
params.addParameter('offset', 1e3);
% -- Experimental: Renormalization/Decreasing rate --
params.addParameter('renormalize', false, @islogical);
params.addParameter('dec_rate', false, @islogical);
% -- Reporting --
params.addParameter('verbosity', 11);
params.addParameter('print_ftrue', false, @islogical);
params.addParameter('save_ftrue', false, @islogical);
params.addParameter('gradcheck', true, @islogical);

params.parse(varargin{:});

%% Copy from params object
% -- GCP Parameters --
mask        = params.Results.mask;
objfh       = params.Results.objfh;
gradfh      = params.Results.gradfh;
lowbound    = params.Results.lowbound;
% -- Stochastic Gradient Steps --
init        = params.Results.init;
gsamples    = params.Results.gsamples;
gsampler    = params.Results.gsampler;
rate        = params.Results.rate;
adam        = params.Results.adam;
% -- Convergence Criteria --
epochiters  = params.Results.epochiters;
maxepochs   = params.Results.maxepochs;
fsamples    = params.Results.fsamples;
fsampler    = params.Results.fsampler;
conv_cond   = params.Results.conv_cond;
% -- Additional parameters for adam --
beta1       = params.Results.beta1;
beta2       = params.Results.beta2;
epsilon     = params.Results.epsilon;
% -- Additional parameters for stratified sampler --
prop_nz     = params.Results.prop_nz;
num_cand    = params.Results.num_cand;
% -- Additional parameters for proportional sampler --
offset      = params.Results.offset;
% -- Experimental: Renormalization/Decreasing rate --
renormalize = params.Results.renormalize;
dec_rate    = params.Results.dec_rate;
% -- Reporting --
verbosity   = params.Results.verbosity;
print_ftrue = params.Results.print_ftrue;
save_ftrue  = params.Results.save_ftrue;
gradcheck   = params.Results.gradcheck;

%% Setup samplers
if ~isfunc(gsampler)
    switch gsampler
        case 'unif'  % Uniform sampling of entries
            gsampler = @samp_unif;
        case 'all'   % Sampling of all entries
            gsampler = @samp_all;
        case 'strat' % Stratified sampling of entries
            gsampler = @(nsample,sz,mask,meta) ...
                samp_strat(nsample,sz,mask,meta,X,prop_nz,num_cand);
        case 'prop'  % Proportional sampling of entries based on slice norm
            gsampler = @(nsample,sz,mask,meta) ...
                samp_prop(nsample,sz,mask,meta,X,offset);
        otherwise
            error('Sampler not implemented yet');
    end
end

if ~isfunc(fsampler)
    switch fsampler
        case 'unif'  % Uniform sampling of entries
            fsampler = @samp_unif;
        case 'all'   % Sampling of all entries
            fsampler = @samp_all;
        case 'strat' % Stratified sampling of entries
            fsampler = @(nsample,sz,mask,meta) ...
                samp_strat(nsample,sz,mask,meta,X,prop_nz,num_cand);
        case 'prop'  % Proportional sampling of entries based on slice norm
            fsampler = @(nsample,sz,mask,meta) ...
                samp_prop(nsample,sz,mask,meta,X,offset);
        otherwise
            error('Sampler not implemented yet');
    end
end

%% Welcome
if verbosity > 10
    fprintf('\n-----\nWelcome to GCP-SGD\n\n');
    fprintf('# f-samples: %d\n', fsamples);
    fprintf('# g-samples: %d\n', gsamples);
end

%% Initialize factor matrices and moment matrices
% Initialize factor matrices (and normalize if needed)
if iscell(init)
    Uinit = init;
    M = ktensor(Uinit);
elseif strcmp(init,'random')
    Uinit = cell(n,1);
    for k = 1:n
        Uinit{k} = rand(sz(k),r);
    end
    M = ktensor(Uinit);
    
    % Normalize
    M = M * (norm_est(X,mask)/norm(M));
    M = normalize(M,0);
end

% Initialize moments
if adam
    m = cell(n,1); v = cell(n,1);
    for k = 1:n
        m{k} = zeros(sz(k),r);
        v{k} = zeros(sz(k),r);
    end
end

%% Extract samples for estimating function value
fsubs = fsampler(fsamples,sz,mask,[]);
fvals = X(fsubs);

%% Initial function value
fest = fg_est(M,X,fsubs,'xvals',fvals,'objfh',objfh,'gradfh',gradfh,'IgnoreLambda',true);

if verbosity > 10
    fprintf('Initial f-est: %e\n', fest);
end
if verbosity > 20
    start_time = clock;
    last_time = clock;
end

info.fest = fest;
if save_ftrue
    info.ftrue = collapse(tenfun(objfh,X,full(M)));
end

%% Main loop
nepoch = 0; niters = 0; gsubs_meta = [];
while nepoch < maxepochs
    nepoch = nepoch + 1;
    
    for iter = 1:epochiters
        niters = niters + 1;
        % Select subset for stochastic gradient
        [gsubs,wvals,gsubs_meta] = gsampler(gsamples,sz,mask,gsubs_meta);
        
        % Compute gradients for each mode
        [~,Gest] = fg_est(M,X,gsubs,'wvals',wvals,'objfh',objfh,'gradfh',gradfh,'IgnoreLambda',true);
        if gradcheck && any(any(isinf(cell2mat(Gest))))
            error('Infinite gradient reached! (epoch = %g, iter = %g)',nepoch,iter);
        end
        
        % Take a step
        if adam
            m = cellfun(@(mk,gk) beta1*mk + (1-beta1)*gk,m,Gest,'UniformOutput',false);
            v = cellfun(@(vk,gk) beta2*vk + (1-beta2)*gk.^2,v,Gest,'UniformOutput',false);
            mhat = cellfun(@(mk) mk/(1-beta1^niters),m,'UniformOutput',false);
            vhat = cellfun(@(vk) vk/(1-beta2^niters),v,'UniformOutput',false);
            M.u = cellfun(@(uk,mhk,vhk) max(lowbound,uk-rate*mhk./(sqrt(vhk)+epsilon)),M.u,mhat,vhat,'UniformOutput',false);
        else
            M.u = cellfun(@(uk,gk) max(lowbound,uk-rate*gk),M.u,Gest,'UniformOutput',false);
        end
    end
    
    % Estimate objective function value
    festold = fest;
    fest = fg_est(M,X,fsubs,'xvals',fvals,'objfh',objfh,'gradfh',gradfh,'IgnoreLambda',true);
    info.fest = [info.fest fest];
    
    % Reporting
    if verbosity > 10
        fprintf(' Epoch %2d: fest = %e', nepoch, fest);
        
        if print_ftrue
            fprintf(' ftrue = %e',collapse(tenfun(objfh,X,full(M))));
        end
        if verbosity > 20
            fprintf(' (%4.3g seconds, %4.3g seconds elapsed)', ...
                etime(clock,last_time),etime(clock,start_time));
            last_time = clock;
        end
        fprintf('\n');
    end
    
    if save_ftrue
        info.ftrue = [info.ftrue collapse(tenfun(objfh,X,full(M)))];
    end
    
    % Renormalization
    if renormalize
        M = normalize(M,0);
        if adam
            for k = 1:n
                m{k} = zeros(sz(k),r);
                v{k} = zeros(sz(k),r);
            end
            niters = 0;
        end
    end
    
    % Decreasing rate
    if dec_rate && (fest > festold)
        rate = rate/2;
    end
    
    % Check convergence condition
    if conv_cond(fest,festold)
        break;
    end
end

%% Clean up final result
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

%% Entry samplers

function [subs,wvals,meta] = samp_unif(nsample,sz,mask,meta)
%SAMP_UNIF Subindices to sample nsample entries of known entries of the
%tensor sampled uniformly at random (with replacement).

if isempty(mask) % No missing entries
    subs = zeros(nsample,length(sz));
    for k = 1:length(sz)
        subs(:,k) = randi(sz(k),nsample,1);
    end
else % Missing entries
    if isa(mask,'sptensor')
        idx = randi(nnz(mask), nsample, 1);
        subs = mask.subs(idx,:);
    else
        if ~isfield(meta,'lin_idx') % Extract linear indices for dense
            meta.lin_idx = find(double(mask));
        end
        idx = randi(length(meta.lin_idx),nsample,1);
        subs = tt_ind2sub(sz,meta.lin_idx(idx));
    end
end

wvals = [];

end

function [subs,wvals,meta] = samp_all(~,sz,mask,meta)
%SAMP_ALL Subindices to sample all known entries of the tensor.

if ~isfield(meta,'allsubs')
    if isempty(mask) % No missing entries
        meta.allsubs = tt_ind2sub(sz,1:prod(sz));
    else % Missing entries
        meta.allsubs = find(mask);
    end
end

subs = meta.allsubs;
wvals = [];
meta = [];

end

function [subs,wvals,meta] = samp_strat(nsample,sz,mask,meta,X,prop_nz,num_cand)
%SAMP_STRAT Subindices to sample nsample entries of a sparse tensor with a
%fixed number of zeros and fixed number of nonzeros.

% Only handle fully-sampled data for now
if ~isempty(mask)
    if ~isfield(meta,'strat_missing_flag')
        warning('Stratified sampling not yet supported for missing data. Using uniform.');
    end
    [subs,wvals,meta] = samp_unif(nsample,sz,mask,meta);
    meta.strat_missing_flag = true;
    return;
end

% Calculate sizes
nsample_nz = floor(nsample*prop_nz);
nsample_z  = nsample-nsample_nz;

% Get linear indices of nonzeros
if ~isfield(meta,'nz_list')
    meta.nz_list = tt_sub2ind(sz,find(X));
end

% Generate large list of zeros
if ~isfield(meta,'z_list') || (meta.z_idx + nsample_z - 1) > length(meta.z_list)
    meta.z_list = randi(prod(sz),num_cand,1);
    meta.z_list = meta.z_list(~ismember(meta.z_list,meta.nz_list));
    meta.z_idx  = 1;
end

% Choose nonzeros
idx_nz  = meta.nz_list(randi(length(meta.nz_list),nsample_nz,1));
prob_nz = nsample_nz/nsample * 1/length(meta.nz_list);

% Choose zeros
idx_z      = meta.z_list(meta.z_idx:(meta.z_idx+nsample_z-1));
meta.z_idx = meta.z_idx + nsample;
prob_z     = nsample_z/nsample * 1/(prod(sz)-length(meta.nz_list));

% Compile values
subs  = tt_ind2sub(sz,[idx_nz; idx_z]);
wvals = [1/prob_nz*ones(nsample_nz,1); 1/prob_z*ones(nsample-nsample_nz,1)];

end

function [subs,wvals,meta] = samp_prop(nsample,sz,mask,meta,X,offset)
%SAMP_PROP Subindices to sample nsample entries of the tensor sampled with
%probability proportional to the product of the marginal sums (with an
%added offset)
%
%   For example, entry (i,j,k) of a 3-way tensor X is sampled with
%   probability proportional to
%
%     (offset + collapse(X(i,:,:)))
%       * (offset + collapse(X(:,j,:)))
%       * (offset + collapse(X(:,:,k)))
%
%   Offset can be used to make the probabilities generally more balanced.

% Only handle fully-sampled data for now
if ~isempty(mask)
    if ~isfield(meta,'prop_missing_flag')
        warning('Proportional sampling not yet supported for missing data. Using uniform.');
    end
    [subs,wvals,meta] = samp_unif(nsample,sz,mask,meta);
    meta.prop_missing_flag = true;
    return;
end

n = length(sz);

% Calculate probabilities
if ~isfield(meta,'probs')
    meta.probs = arrayfun(@(k) offset+collapse(X,setdiff(1:n,k),@sum),1:n,'UniformOutput',false);
    meta.probs = cellfun(@(p) p / sum(p),meta.probs,'UniformOutput',false);
end

% Calculate cumulative probabilities
if ~isfield(meta,'cprobs')
    meta.cprobs = cellfun(@(p) [0; cumsum(p(1:end-1)); 1],meta.probs,'UniformOutput',false);
end

subs = zeros(nsample,n);
wvals = ones(nsample,1);
for k = 1:n
    [~,~,subs(:,k)] = histcounts(rand(nsample,1),meta.cprobs{k});
    wvals = wvals./meta.probs{k}(subs(:,k));
end

end