function [subs,wvals,meta] = cp_adam_prop(nsamples,sz,mask,meta,X,offset)

% Only handle fully-sampled data for now
if ~isempty(mask) && ~isfield(meta,'prop_missing_flag')
    warning('Proportional sampling not yet supported for missing data. Using uniform.');
    [subs,wvals,meta] = cp_adam_unif(nsamples,sz,mask,meta);
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

subs = zeros(nsamples,n);
wvals = ones(nsamples,1);
for k = 1:n
    [~,~,subs(:,k)] = histcounts(rand(nsamples,1),meta.cprobs{k});
    wvals = wvals./meta.probs{k}(subs(:,k));
end

end