function [subs,wvals,meta] = cp_adam_all(~,sz,~,~)

subs = tt_ind2sub(sz,1:prod(sz));
wvals = [];
meta = [];

end