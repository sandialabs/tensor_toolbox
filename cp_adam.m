function [M,info] = cp_adam(X,r,varargin)
%CP_ADAM Fits a generalized CP model to a tensor via stochastic
%optimization with adaptive moment estimation (Adam).
%
%   P = CP_ADAM(X,R) computes an estimate of the best rank-R generalized
%   CP model of a tensor X using stochastic optimization with adaptive
%   moment estimation (Adam). The input X can be a tensor, sptensor,
%   ktensor, or ttensor. The result P is a ktensor.
%
%   P = CP_ADAM(X,R,'param',value,...) specifies optional parameters and
%   values. Valid parameters and their default values are:
%   -- General --
%   'mask' - Tensor indicating missing data (0 = missing, 1 = present). {[]}
%   'init' - Initial guess [{'random'}|'nvecs'|cell array]
%   -- Batch sizes --
%   'fsamples' - Batch size for calculating the function value {1000}
%   'gsamples' - Batch size for calculating the gradient {1}
%   'gsample_gen' - Function to generate sample indices {@randi}
%   -- Iterations/Epochs --
%   'epochiters' - Number of iterations per epoch {1000}
%   'maxepochs' - Maximum number of epochs {100}
%   'conv_cond' - Convergence condition {@(f,fold) f > fold}
%   -- Rates --
%   'rate' - Step size {1e-3}
%   'beta1' - First moment decay {0.9}
%   'beta2' - Second moment decay {0.999}
%   'epsilon' - Small value to help with numerics in division {1e-8}
%   -- Loss function --
%   'objfh' - Loss function {@(x,m) (x-m).^2}
%   'gradfh' - Gradient of loss function {@(x,m) -2*(x-m)}
%   'lowbound' - Low bound constraint {-Inf}
%   -- Restart/Decreasing rate --
%   'restart' - Restart moment estimates and normalize at each epoch. {false}
%   'dec_rate' - Halve the step size at each epoch where the function value
%                increases. {false}
%   -- Reporting --
%   'verbosity' - Verbosity level {11}
%   'print_ftrue' - Print the true objective function value at each epoch {false}
%   'save_ftrue' - Save the true objective function value at each epoch {false}
%   'gradcheck' - Trigger error if the gradient is ever infinite {true}
%
%   [P,out] = CP_ADAM(...) also returns a structure with the trace of the
%   function value.
%
%   Examples:
%   X = sptenrand([5 4 3], 10);
%   P = cp_adam(X,2);
%   P = cp_adam(X,2, ...
%               'objfh',@(x,m) log(m+1)-x.*log(m+1e-7), ...
%               'gradfh',@(x,m) 1./(m+1)-x./(m+1e-7), ...
%               'lowbound',1e-6);
%
%   See also KTENSOR, TENSOR, SPTENSOR, TTENSOR, CP_SGD, CP_ALS, CP_OPT,
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
params.addParameter('verbosity', 11);
params.addParameter('init', 'random');
params.addParameter('gsamples', 1);
params.addParameter('fsamples', 1000);
params.addParameter('epochiters', 1000);
params.addParameter('maxepochs',100);
params.addParameter('rate', 1e-3);
params.addParameter('gsample_gen', @randi, @(x) isa(x,'function_handle'));
params.addParameter('print_ftrue', false, @islogical);
params.addParameter('save_ftrue', false, @islogical);
params.addParameter('conv_cond',@(f,fold) f > fold,@(c) isa(c,'function_handle'));
params.addParameter('beta1', 0.9);
params.addParameter('beta2', 0.999);
params.addParameter('epsilon', 1e-8);
params.addParameter('gradcheck', true, @islogical);
params.addParameter('objfh', @(x,m) (x-m).^2, @(f) isa(f,'function_handle'));
params.addParameter('gradfh', @(x,m) -2*(x-m), @(f) isa(f,'function_handle'));
params.addParameter('lowbound', -Inf, @isnumeric);
params.addParameter('restart',false, @islogical);
params.addParameter('dec_rate',false,@islogical);
params.addParameter('mask',[],@(mask) isa(mask,'sptensor') || isa(mask,'tensor'));
params.parse(varargin{:});

%% Copy from params object
verbosity   = params.Results.verbosity;
init        = params.Results.init;
gsamples    = params.Results.gsamples;
fsamples    = params.Results.fsamples;
epochiters  = params.Results.epochiters;
maxepochs   = params.Results.maxepochs;
rate        = params.Results.rate;
gsample_gen = params.Results.gsample_gen;
print_ftrue = params.Results.print_ftrue;
save_ftrue  = params.Results.save_ftrue;
conv_cond   = params.Results.conv_cond;
beta1       = params.Results.beta1;
beta2       = params.Results.beta2;
epsilon     = params.Results.epsilon;
gradcheck   = params.Results.gradcheck;
objfh       = params.Results.objfh;
gradfh      = params.Results.gradfh;
lowbound    = params.Results.lowbound;
restart     = params.Results.restart;
dec_rate    = params.Results.dec_rate;
mask        = params.Results.mask;

%% Welcome
if verbosity > 10
    fprintf('\n-----\nWelcome to CP-ADAM\n\n');
    fprintf('# f-samples: %d\n', fsamples);
    fprintf('# g-samples: %d\n', gsamples);
end

%% Initialize factor matrices and momentum matrices
if iscell(init)
    Uinit = init;
else
    Uinit = cell(n,1);
    for k = 1:n
        Uinit{k} = rand(sz(k),r);
    end
end
M = ktensor(Uinit);
M = M * (norm(X)/norm(M));
M = normalize(M,0);

m = cell(n,1);
for k = 1:n
    m{k} = zeros(sz(k),r);
end
v = cell(n,1);
for k = 1:n
    v{k} = zeros(sz(k),r);
end

%% Extract indices from mask if not sparse tensor
if ~isempty(mask) && ~isa(mask,'sptensor')
    fprintf('Extracting indices from mask...');
    mask_idx = find(double(mask));
    fprintf('done!\n');
end

%% Extract samples for estimating function value
% TBD: This version assumes that we allows for _repeats_ which may or may
% not be a good thing.
if ~isempty(mask)
    if isa(mask,'sptensor')
        fidx  = randi(nnz(mask), fsamples, 1);
        fsubs = mask.subs(fidx,:);
    else
        fidx  = randi(length(mask_idx), fsamples, 1);
        fsubs = tt_ind2sub(sz,mask_idx(fidx));
    end
else
    fidx  = randi(prod(sz), fsamples, 1);
    fsubs = tt_ind2sub(sz, fidx);
end
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
nepoch = 0; niters = 0;
while nepoch < maxepochs
    nepoch = nepoch + 1;
    
    for iter = 1:epochiters
        niters = niters + 1;
        % Select subset for stochastic gradient
        if ~isempty(mask)
            if isa(mask,'sptensor')
                gidx = gsample_gen(nnz(mask), gsamples, 1);
                gsubs = mask.subs(gidx,:);
            else
                gidx = gsample_gen(length(mask_idx), gsamples, 1);
                gsubs = tt_ind2sub(sz,mask_idx(gidx));
            end
        else
            gidx = gsample_gen(prod(sz), gsamples, 1);
            gsubs = tt_ind2sub(sz, gidx);
        end
        
        % Compute gradients and moments for each mode and take a step
        [~,Gest] = fg_est(M,X,gsubs,'objfh',objfh,'gradfh',gradfh,'IgnoreLambda',true);
        if gradcheck && any(any(isinf(cell2mat(Gest))))
            error('Infinite gradient reached! (epoch = %g, iter = %g)',nepoch,iter);
        end
        
        m = cellfun(@(mk,gk) beta1*mk + (1-beta1)*gk,m,Gest,'UniformOutput',false);
        v = cellfun(@(vk,gk) beta2*vk + (1-beta2)*gk.^2,v,Gest,'UniformOutput',false);
        mhat = cellfun(@(mk) mk/(1-beta1^niters),m,'UniformOutput',false);
        vhat = cellfun(@(vk) vk/(1-beta2^niters),v,'UniformOutput',false);
        M.u = cellfun(@(uk,mhk,vhk) max(lowbound,uk-rate*mhk./(sqrt(vhk)+epsilon)),M.u,mhat,vhat,'UniformOutput',false);
    end

    festold = fest;
    fest = fg_est(M,X,fsubs,'xvals',fvals,'objfh',objfh,'gradfh',gradfh,'IgnoreLambda',true);
    info.fest = [info.fest fest];
    
    % Not the cleanest way to print trace. TODO: Clean this up.
    if verbosity > 10
        fprintf(' Epoch %2d: fest = %e', nepoch, fest);
    end
    if verbosity > 10 && print_ftrue
        fprintf(' ftrue = %e',collapse(tenfun(objfh,X,full(M))));
    end
    if verbosity > 20
        fprintf(' (%4.3g seconds, %4.3g seconds elapsed)', ...
            etime(clock,last_time),etime(clock,start_time));
        last_time = clock;
    end
    if verbosity > 10
        fprintf('\n');
    end
    
    if save_ftrue
        info.ftrue = [info.ftrue collapse(tenfun(objfh,X,full(M)))];
    end

    %M = normalize(M,0);
    % Restart
    %if restart && festold <= fest
    if restart
        for k = 1:n
            m{k} = zeros(sz(k),r);
        end
        for k = 1:n
            v{k} = zeros(sz(k),r);
        end
        niters = 0;
        M = normalize(M,0);
    end
    if dec_rate && (fest > festold)
        %beta1 = 1-(1-beta1)/2;
        %beta2 = 1-(1-beta2)/2;
        rate = rate/2;
    end

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
