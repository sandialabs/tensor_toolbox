function [M,info] = cp_adam(X,r,varargin)
%CP_ADAM Adaptive Momentum Estimation Stochastic gradient descent for CP

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
params.addParameter('conv_cond',@(f,fold) f >= fold,@(c) isa(c,'function_handle'));
params.addParameter('beta1', 0.9);
params.addParameter('beta2', 0.999);
params.addParameter('epsilon', 1e-8);
params.addParameter('gradcheck', true, @islogical);
params.addParameter('objfh', @(x,m) (x-m).^2, @(f) isa(f,'function_handle'));
params.addParameter('gradfh', @(x,m) -2*(x-m), @(f) isa(f,'function_handle'));
params.addParameter('lowbound', -Inf, @isnumeric);
params.addParameter('restart',false, @islogical);
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

m = cell(n,1);
for k = 1:n
    m{k} = zeros(sz(k),r);
end
v = cell(n,1);
for k = 1:n
    v{k} = zeros(sz(k),r);
end

%% Extract samples for estimating function value
% TBD: Handle missing data. Currently assumes input is complete. Also note
% that this version assumes that we allows for _repeats_ which may or may
% not be a good thing.
fidx  = randi(prod(sz), fsamples, 1);
fsubs = tt_ind2sub(sz, fidx);
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
        gidx = gsample_gen(prod(sz), gsamples, 1);
        gsubs = tt_ind2sub(sz, gidx);
        
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
    
    % Restart
    if restart && festold <= fest
        for k = 1:n
            m{k} = zeros(sz(k),r);
        end
        for k = 1:n
            v{k} = zeros(sz(k),r);
        end
        niters = 0;
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