function T = sthosvd(X,tol)
%STHOVD Compute sequentially-truncated higher-order SVD (Tucker).
%
%    T = STHOSVD(X,TOL) computes a Tucker decomposition with error
%    specified by tol, i.e., ||X-T||/||X|| <= TOL. The decomposition
%    automatically determines the appropriate ranks of the Tucker
%    decomposition.
%
%MATLAB Tensor Toolbox.
%Copyright 2017, Sandia Corporation.

n = size(X);
d = ndims(X);
normXsqr = collapse(X.^2);
cutoff = tol.^2 * normXsqr / d

r = zeros(d,1);
U = cell(d,1);
for k = 1:d
    Xk = double(tenmat(X,k));
    Y = Xk*Xk';
    [V,D] = eig(Y);
    [dvec,pi] = sort(diag(D),'descend');
    tmp = cumsum(dvec,'reverse')
    r(k) = find(tmp >= cutoff, 1, 'last');
    U{k} = V(:,pi(1:r(k)));
    X = ttm(X,U{k}',k);
end
G = X;
T = ttensor(G,U);
