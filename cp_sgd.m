function M = cp_sgd(X,r,varargin)
%CP_SGD Stochastic gradient descent for CP

%% Extract number of dimensions and norm of X.
n = ndims(X);
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
params.parse(varargin{:});

%% Copy from params object
verbosity = params.Results.verbosity;
init = params.Results.init;
gsamples = params.Results.gsamples;
fsamples = params.Results.fsamples;
epochiters = params.Results.epochiters;
maxepochs = params.Results.maxepochs;
rate = params.Results.rate;
gsample_gen = params.Results.gsample_gen;
print_ftrue = params.Results.print_ftrue;

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
fidx = randi(prod(sz), fsamples, 1);
fsubs = tt_ind2sub(sz, fidx);

%% Initial function value
fest = fg_est(M,X,fsubs);

if verbosity > 10
    fprintf('Initial f-est: %e\n', fest);
end

%% Main loop
nepoch = 0;
while nepoch < maxepochs
    
%     U = renormalize(U);    
    
    for iter = 1:epochiters
    
        % Select subset for stochastic gradient
        gidx = gsample_gen(prod(sz), gsamples, 1);
        gsubs = tt_ind2sub(sz, gidx);
        
        % Compute gradients for each mode and take step
        % This isn't using any of the ability of fg_est to keep values and
        % also creates a ktensor from U each time. Prob fairly inefficient.
        [~,Gest] = fg_est(M,X,gsubs);
        if any(isinf(cell2mat(Gest)))
            error('Infinite gradient reached!')
        end
        M.u = cellfun(@(u,g) u+rate*g,M.u,Gest,'UniformOutput',false);
        
    end
    
    festold = fest;
    % This isn't using any of the ability of fg_est to keep values and
    % also creates a ktensor from U each time. Prob fairly inefficient.
    fest = fg_est(M,X,fsubs);
    if fest >= festold
        break;
    end
    
    nepoch = nepoch + 1;
    
    if verbosity > 10
        if print_ftrue
            fprintf(' Epoch %2d: fest = %e, ftrue = %e\n', nepoch, fest, ...
                fg(M,X,'Type','G','IgnoreLambda',true));
        else
            fprintf(' Epoch %2d: fest = %e\n', nepoch, fest);
        end
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


function U = renormalize(n, r, U)
%RENORMALIZE Absord all the column weights into first factor matrix
for n = 2:N
    tmp = sqrt(sum(U{n}.^2,1));
    U{n} = bsxfun(@rdivide,Y,tmp);
    U{1} = bsxfun(@times,U{1},tmp);
end




