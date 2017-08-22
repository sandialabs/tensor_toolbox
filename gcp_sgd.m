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
%   -- General --
%   'mask' - Tensor marking missing data (0 = missing, 1 = present). {[]}
%   'init' - Initial guess [{'random'}|'nvecs'|cell array]
%   'adam' - Use adaptive moment estimation {true}
%   -- Batch sizes --
%   'fsamples' - Batch size for calculating the function value {1000}
%   'fsampler' - Function to generate function value samples {@cp_adam_unif}
%   'gsamples' - Batch size for calculating the gradient {1}
%   'gsampler' - Function to generate gradient samples {@cp_adam_unif}
%   -- Iterations/Epochs --
%   'epochiters' - Number of iterations per epoch {1000}
%   'maxepochs'  - Maximum number of epochs {100}
%   'conv_cond'  - Convergence condition {@(f,fold) f > fold}
%   -- Rates --
%   'rate'    - Step size {1e-3}
%   'beta1'   - First moment decay (relevant for adam) {0.9}
%   'beta2'   - Second moment decay (relevant for adam) {0.999}
%   'epsilon' - Small value to help with numerics in division
%               (relevant for adam) {1e-8}
%   -- Loss function --
%   'objfh'    - Loss function {@(x,m) (x-m).^2}
%   'gradfh'   - Gradient of loss function {@(x,m) -2*(x-m)}
%   'lowbound' - Low bound constraint {-Inf}
%   -- Renormalization/Decreasing rate --
%   'renormalize' - Renormalize at each epoch and restart moment estimates
%                   (relevant for adam). {false}
%   'dec_rate'    - Halve the step size at each epoch where the function
%                   value increases. {false}
%   -- Reporting --
%   'verbosity'   - Verbosity level {11}
%   'print_ftrue' - Print the true objective function value at each epoch
%                   {false}
%   'save_ftrue'  - Save the true objective function value at each epoch
%                   {false}
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
params = inputParser;
% -- General --
params.addParameter('mask', [], @(mask) isa(mask,'sptensor') || isa(mask,'tensor'));
params.addParameter('init', 'random');
params.addParameter('adam', true, @islogical);
% -- Batch sizes --
params.addParameter('fsamples', 1000);
params.addParameter('fsampler', @cp_adam_unif, @(x) isa(x,'function_handle'));
params.addParameter('gsamples', 1);
params.addParameter('gsampler', @cp_adam_unif, @(x) isa(x,'function_handle'));
% -- Iterations/Epochs --
params.addParameter('epochiters', 1000);
params.addParameter('maxepochs', 100);
params.addParameter('conv_cond', @(f,fold) f > fold, @(c) isa(c,'function_handle'));
% -- Rates --
params.addParameter('rate', 1e-3);
params.addParameter('beta1', 0.9);
params.addParameter('beta2', 0.999);
params.addParameter('epsilon', 1e-8);
% -- Loss function --
params.addParameter('objfh', @(x,m) (x-m).^2, @(f) isa(f,'function_handle'));
params.addParameter('gradfh', @(x,m) -2*(x-m), @(f) isa(f,'function_handle'));
params.addParameter('lowbound', -Inf, @isnumeric);
% -- Renormalization/Decreasing rate --
params.addParameter('renormalize', false, @islogical);
params.addParameter('dec_rate', false, @islogical);
% -- Reporting --
params.addParameter('verbosity', 11);
params.addParameter('print_ftrue', false, @islogical);
params.addParameter('save_ftrue', false, @islogical);
params.addParameter('gradcheck', true, @islogical);

params.parse(varargin{:});

%% Copy from params object
% -- General --
mask        = params.Results.mask;
init        = params.Results.init;
adam        = params.Results.adam;
% -- Batch sizes --
fsamples    = params.Results.fsamples;
fsampler    = params.Results.fsampler;
gsamples    = params.Results.gsamples;
gsampler    = params.Results.gsampler;
% -- Iterations/Epochs --
epochiters  = params.Results.epochiters;
maxepochs   = params.Results.maxepochs;
conv_cond   = params.Results.conv_cond;
% -- Rates --
rate        = params.Results.rate;
beta1       = params.Results.beta1;
beta2       = params.Results.beta2;
epsilon     = params.Results.epsilon;
% -- Loss function --
objfh       = params.Results.objfh;
gradfh      = params.Results.gradfh;
lowbound    = params.Results.lowbound;
% -- Renormalization/Decreasing rate --
renormalize = params.Results.renormalize;
dec_rate    = params.Results.dec_rate;
% -- Reporting --
verbosity   = params.Results.verbosity;
print_ftrue = params.Results.print_ftrue;
save_ftrue  = params.Results.save_ftrue;
gradcheck   = params.Results.gradcheck;

%% Welcome
if verbosity > 10
    fprintf('\n-----\nWelcome to CP-ADAM\n\n');
    fprintf('# f-samples: %d\n', fsamples);
    fprintf('# g-samples: %d\n', gsamples);
end

%% Initialize factor matrices and moment matrices
% Initialize factor matrices (and normalize if needed)
if iscell(init)
    Uinit = init;
    M = ktensor(Uinit);
else
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