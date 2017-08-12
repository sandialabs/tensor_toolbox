function [subs,wvals,meta] = cp_adam_prop(nsample,sz,mask,meta,X,offset)
%CP_ADAM_PROP Subindices to sample nsample entries of the tensor sampled
%with probability proportional to the product of the marginal sums (with an
%added offset)
%
%   Namely, entry (i,j,k) of a 3-way tensor X is sampled with probability
%   proportional to
%
%     collapse(X(i,:,:))*collapse(X(:,j,:))*collapse(X(:,:,k)) + offset
%
%   Offset can be used to make the probabilities generally more balanced.
%
%   wvals provides inverse probabilities that can be used in a weighted sum
%   to provide an unbiased estimate of the unweighted sum.
%
%   To use in cp_adam, pass into 'gsampler' as:
%
%     @(nsample,sz,mask,meta) cp_adam_prop(nsample,sz,mask,meta,X,offset)
%
%   where X is the data tensor and offset is the chosen offset (e.g., 1e3).
%
%   See also CP_ADAM.

% Only handle fully-sampled data for now
if ~isempty(mask) && ~isfield(meta,'prop_missing_flag')
    warning('Proportional sampling not yet supported for missing data. Using uniform.');
    [subs,wvals,meta] = cp_adam_unif(nsample,sz,mask,meta);
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