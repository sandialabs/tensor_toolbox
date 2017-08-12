function [subs,wvals,meta] = cp_adam_all(~,sz,mask,meta)
%CP_ADAM_ALL Subindices to sample all known entries of the tensor.
%
%   See also CP_ADAM.

if ~isfield(meta,'allsubs')
    if isempty(mask) % No missing entries
        meta.allsubs = tt_ind2sub(sz,1:prod(sz));
    else % Missing entries
        meta.allsubs = find(mask);
    end
end

subs = meta.allsubs;
wvals = [];
meta = [];

end