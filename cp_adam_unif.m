function [subs,wvals,meta] = cp_adam_unif(nsample,sz,mask,meta)
%CP_ADAM_ALL Subindices to sample nsample entries of known entries of the
%tensor sampled uniformly at random (with replacement).
%
%   See also CP_ADAM.

if isempty(mask) % No missing entries
    subs = zeros(nsample,length(sz));
    for k = 1:length(sz)
        subs(:,k) = randi(sz(k),nsample,1);
    end
else % Missing entries
    if isa(mask,'sptensor')
        idx = randi(nnz(mask), nsample, 1);
        subs = mask.subs(idx,:);
    else
        if ~isfield(meta,'lin_idx') % Extract linear indices for dense
            meta.lin_idx = find(double(mask));
        end
        idx = randi(length(meta.lin_idx),nsample,1);
        subs = tt_ind2sub(sz,meta.lin_idx(idx));
    end
end

wvals = [];

end