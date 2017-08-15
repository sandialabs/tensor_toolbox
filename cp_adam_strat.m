function [subs,wvals,meta] = cp_adam_strat(nsample,sz,mask,meta,X,prop_nz,num_cand)
%CP_ADAM_STRAT Subindices to sample nsample entries of a sparse tensor with
%a fixed number of zeros and fixed number of nonzeros.
%
%   wvals provides inverse probabilities that can be used in a weighted sum
%   to provide an unbiased estimate of the unweighted sum.
%
%   To use in cp_adam, pass into 'gsampler' as:
%
%     @(nsample,sz,mask,meta) cp_adam_prop(nsample,sz,mask,meta,X,prop_nz,num_cand)
%
%   where X is the data tensor, prop_nz is the proportion of samples that
%   should be nonzero and num_cand is the number of nonzero candidates to
%   pull ahead of time.
%
%   See also CP_ADAM.

% Only handle fully-sampled data for now
if ~isempty(mask) && ~isfield(meta,'strat_missing_flag')
    warning('Stratified sampling not yet supported for missing data. Using uniform.');
    [subs,wvals,meta] = cp_adam_unif(nsample,sz,mask,meta);
    meta.strat_missing_flag = true;
    return;
end

% Calculate sizes
nsample_nz = floor(nsample*prop_nz);
nsample_z  = nsample-nsample_nz;

% Get linear indices of nonzeros
if ~isfield(meta,'nz_list')
    meta.nz_list = tt_sub2ind(sz,find(X));
end

% Generate large list of zeros
if ~isfield(meta,'z_list') || (meta.z_idx + nsample_z - 1) > length(meta.z_list)
    meta.z_list = randi(prod(sz),num_cand,1);
    meta.z_list = meta.z_list(~ismember(meta.z_list,meta.nz_list));
    meta.z_idx  = 1;
end

% Choose nonzeros
idx_nz  = meta.nz_list(randi(length(meta.nz_list),nsample_nz,1));
prob_nz = nsample_nz/nsample * 1/length(meta.nz_list);

% Choose zeros
idx_z      = meta.z_list(meta.z_idx:(meta.z_idx+nsample_z-1));
meta.z_idx = meta.z_idx + nsample;
prob_z     = nsample_z/nsample * 1/(prod(sz)-length(meta.nz_list));

% Compile values
subs  = tt_ind2sub(sz,[idx_nz; idx_z]);
wvals = [1/prob_nz*ones(nsample_nz,1); 1/prob_z*ones(nsample-nsample_nz,1)];

end