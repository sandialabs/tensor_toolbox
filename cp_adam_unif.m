function [subs,wvals,meta] = cp_adam_unif(nsamples,sz,mask,meta)

if isempty(mask) % No missing entries
    subs = zeros(nsamples,length(sz));
    for k = 1:length(sz)
        subs(:,k) = randi(sz(k),nsamples,1);
    end
else % Missing entries
    if isa(mask,'sptensor')
        idx = randi(nnz(mask), nsamples, 1);
        subs = mask.subs(idx,:);
    else
        if ~isfield(meta,'lin_idx') % Extract linear indices for dense
            meta.lin_idx = find(double(mask));
        end
        idx = randi(length(meta.lin_idx),nsamples,1);
        subs = tt_ind2sub(sz,meta.lin_idx(idx));
    end
end

wvals = [];

end