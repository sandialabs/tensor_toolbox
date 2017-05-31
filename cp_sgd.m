function [M,info] = cp_sgd(X,r,varargin)
%CP_SGD Stochastic gradient descent for CP

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
params.addParameter('rate', 1e-2);
params.addParameter('gsample_gen', @randi, @(x) isa(x,'function_handle'));
params.addParameter('print_ftrue', false, @islogical);
params.addParameter('conv_cond',@(f,fold) f >= fold,@(c) isa(c,'function_handle'));
params.addParameter('gradcheck', true, @islogical);
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
conv_cond   = params.Results.conv_cond;
gradcheck   = params.Results.gradcheck;

%% Welcome
if verbosity > 10
    fprintf('\n-----\nWelcome to CP-SGD\n\n');
    fprintf('# f-samples: %d\n', fsamples);
    fprintf('# g-samples: %d\n', gsamples);
end

%% Initialize factor matrices
if iscell(init)
    Uinit = init;
else
    Uinit = cell(n,1);
    for k = 1:n
        Uinit{k} = rand(sz(k),r);
    end
end
M = ktensor(Uinit);

%% Extract samples for estimating function value
% TBD: Handle missing data. Currently assumes input is complete. Also note
% that this version assumes that we allows for _repeats_ which may or may
% not be a good thing.
fidx  = randi(prod(sz), fsamples, 1);
fsubs = tt_ind2sub(sz, fidx);
fvals = X(fsubs);

%% Initial function value
fest = fg_est(M,X,fsubs,'xvals',fvals);

if verbosity > 10
    fprintf('Initial f-est: %e\n', fest);
end
if verbosity > 20
    start_time = clock;
    last_time = clock;
end

%% Main loop
nepoch = 0;
info.fest = [];
while nepoch < maxepochs
    nepoch = nepoch + 1;
    
    for iter = 1:epochiters
        
        % Select subset for stochastic gradient
        gidx = gsample_gen(prod(sz), gsamples, 1);
        gsubs = tt_ind2sub(sz, gidx);
        
        % Compute gradients for each mode and take step
        [~,Gest] = fg_est(M,X,gsubs);
        if gradcheck && any(isinf(cell2mat(Gest)))
            error('Infinite gradient reached! (epoch = %g, iter = %g)',nepoch,iter);
        end
        M.u = cellfun(@(u,g) u-rate*g,M.u,Gest,'UniformOutput',false);
        
    end
    
    festold = fest;
    fest = fg_est(M,X,fsubs,'xvals',fvals);
    info.fest = [info.fest fest];
    
    % Not the cleanest way to print trace. TODO: Clean this up.
    if verbosity > 10
        fprintf(' Epoch %2d: fest = %e', nepoch, fest);
    end
    if verbosity > 10 && print_ftrue
        fprintf(' ftrue = %e',2*fg(M,X,'Type','G','IgnoreLambda',true));
    end
    if verbosity > 20
        fprintf(' (%4.3g seconds, %4.3g seconds elapsed)', ...
            etime(clock,last_time),etime(clock,start_time));
        last_time = clock;
    end
    if verbosity > 10
        fprintf('\n');
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