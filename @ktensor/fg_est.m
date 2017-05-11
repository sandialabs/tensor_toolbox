function [fest,info] = fg_est(M, X, subs, varargin)
%FG_EST Estimate mean element function value from a few samples

%%
params = inputParser;
params.addParameter('xvals',[]);
params.addParameter('Uexplode',[]);
params.parse(varargin{:});

%%
xvals = params.Results.xvals;
if isempty(xvals)
    xvals = X(subs);
end

%%
Uexplode = params.Results.xvals;
if isempty(Uexplode)
    Uexplode = cell(size(M.u));
    for k = 1:ndims(M)
        Uexplode{k} = M.u{k}(subs(:,k),:);
    end 
end

%% Calculate model values
tmp = Uexplode{1};
for k = 2:numel(Uexplode)
    tmp = tmp .* Uexplode{k};
end
mvals = sum(tmp,2);
fest = mean( (xvals - mvals).^2 );
%%
info.xvals = xvals;
info.mvals = mvals;