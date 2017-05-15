function [fest,Gest,info] = fg_est(M, X, subs, varargin)
%FG_EST Estimate mean element function value from a few samples

%%
params = inputParser;
params.addParameter('xvals',[]);
params.addParameter('Uexplode',[]);
params.parse(varargin{:});

%%
d = ndims(X);
n = size(X);
r = ncomponents(M);

%% Extra entries specified by subs
xvals = params.Results.xvals;
if isempty(xvals)
    xvals = X(subs);
end

%% Grab appropriate rows of each factor matrix
Uexplode = params.Results.xvals;
if isempty(Uexplode)
    Uexplode = cell(d,1);
    for k = 1:d
        Uexplode{k} = M.u{k}(subs(:,k),:);
    end 
end

%% Calculate model values
% TODO: Note that we've ignored M.lambda here, assuming that it's all ones.
% Probably need to fix this or check!
tmp = Uexplode{1};
for k = 2:d
    tmp = tmp .* Uexplode{k};
end
mvals = sum(tmp,2);

%% Compute function value
fest = mean( (xvals - mvals).^2 );

%% Compute gradient 
% This is inherently doing an MTTKRP, but we avoid forming the G tensor or
% calling the MTTKRP function. THis is probably inefficient and also
% ignores the lambdas (i.e., assumes they are one).
gvals = 2 * (xvals - mvals);

KRexplode = cell(size(M.u));
for k = 1:d
    KRexplode{k} = ones(size(Uexplode{1}));
end
for k = 1:d 
    for kalt = 1:d
        if k ~= kalt
            KRexplode{kalt} = KRexplode{kalt} .* Uexplode{k};
        end
    end    
end

Gest = cell(d,1);
for k = 1:d
    m = zeros(n(k),r);
    mvals = bsxfun(@times,KRexplode{k},gvals);
    msubs = subs(:,k);
    for r = 1:ncomponents(M)
        m(:,r) = accumarray(msubs,mvals(:,r),[n(k),1]);
    end
    Gest{k} = m;        
end
    

%%
info.xvals = xvals;
info.mvals = mvals;