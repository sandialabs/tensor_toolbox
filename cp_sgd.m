function M = cp_sgd(X,r,varargin)
%CP_SGD Stochastic gradient descent for CP

%% Extract number of dimensions and norm of X.
n = ndims(X);
sz = size(X);
normX = norm(X);

%% Set algorithm parameters from input or by using defaults
params = inputParser;
params.addParameter('verbosity', 11);
params.addParameter('init', 'random');
params.addParameter('gsamples', 1);
params.addParameter('fsamples', 1000);
params.addParameter('epochiters', 1000);
params.addParameter('maxepochs',100);
params.addParameter('rate', 1e-2);
params.parse(varargin{:});

%% Copy from params object
verbosity = params.Results.verbosity;
init = params.Results.init;
gsamples = params.Results.gsamples;
fsamples = params.Results.fsamples;
epochiters = params.Results.epochiters;
maxepochs = params.Results.maxepochs;
rate = params.Results.rate;

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
U = Uinit;

%% Extract samples for estimating function value
% TBD: Handle missing data. Currently assumes input is complete. Also note
% that this version assumes that we allows for _repeats_ which may or may
% not be a good thing.
fidx = randi(prod(sz), fsamples, 1);
fsubs = tt_ind2sub(sz, fidx);
fxvals = X(fidx);

%% Initial function value
fest = festimate(n, r, fsamples, fxvals, fsubs, U);

if verbosity > 10
    fprintf('Initial f-est: %e\n', fest);
end

%% Main loop
nepoch = 0;
Uexplode = cell(n,1);
while nepoch < maxepochs
    
    U = renormalize(U);
    
    for iter = 1:epochiters
    
        gidx = randi(prod(sz), gsamples, 1);
        gsubs = tt_ind2sub(sz, gidx);
        for k = 1:n
            Uexplode{k} = U(gsubs(k,:),:);
        end
        
        % Compute gvals
        xvals = X(gidx);
        tmp = Uexplode{1};
        for k = 2:n
            tmp = tmp .* Uexplode{k};
        end
        mvals = sum(tmp,2);
        gvals = xvals - mvals;
        
        % Compute krvecs + gradients for each mode and take step
        
        
    end
    
    festold = fest;
    fest = festimate(n, r, fsamples, fxvals, fsubs, U);
    if fest >= festold
        break;
    end    
    
end



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




