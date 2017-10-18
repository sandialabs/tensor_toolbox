function [P,Uinit,output] = cp_als_rand(X,R,varargin)
%CPRAND Compute a CP decomposition of a dense tensor using randomized least squares.
%
%   P = CP_ALS_RAND(X,R) computes an estimate of the best rank-R
%   CP model of a tensor X using a randomized alternating least-squares
%   algorithm. The input tensor is preprocessed using a mixing algorithm.
%   The input X can be a tensor, ktensor, or ttensor. The result P is a ktensor.
%
%   P = CP_ALS_RAND(X,R,'mix',0) computes the above but without the preprocessing step.
%   This is suitable for some data, and requires less initialization time and space. 
%
%   P = CP_ALS_RAND(X,R,'param',value,...) specifies optional parameters and
%   values. Valid parameters and their default values are:
%   o 'tol'           - Tolerance on difference in fit {0}
%   o 'maxiters'      - Maximum number of iterations {100}
%   o 'dimorder'      - Order to loop through dimensions {1:ndims(A)}
%   o 'init'          - Initial guess [{'random'}|'nvecs'|cell array]
%   o 'printitn'      - Print fit every n iterations; 0 for no printing {1}
%   o 'desired_fit'   - Terminate when fit reaches this threshold {1}
%   o 'mix'           - Include  {10Rlog(R)}
%   o 'num_samples'   - Number of least-squares samples taken in each iteration 
%                       {10Rlog(R)}
%   o 'window'        - Maximum number of iterations to perform without seeing 
%                       any fit improvement; 0 to ignore this condition {0}
%   o 'fit_samples'   - Number of samples to use when computing approximate fit 
%                       {min(nnz(X),2^14)}
%
%   [P,U0] = CP_ALS_RAND(...) also returns the initial guess.
%
%   [P,U0,out] = CP_ALS_RAND(...) also returns additional output that contains
%   the input parameters.
%
%   Note: The exact "fit" is defined as 1 - norm(X-full(P))/norm(X) and is
%   loosely the proportion of the data described by the CP model, i.e., a
%   fit of 1 is perfect.
%
%   Examples:
%   X = tenrand([5 4 3], 10);
%   P = cp_als_rand(X,2);
%   P = cp_als_rand(X,2,'dimorder',[3 2 1]);
%   P = cp_als_rand(X,2,'dimorder',[3 2 1],'init','nvecs');
%   U0 = {rand(5,2),rand(4,2),[]}; %<-- Initial guess for factors of P
%   [P,U0,out] = cp_als_rand(X,2,'dimorder',[3 2 1],'init',U0);
%   P = cp_als_rand(X,2,out.params); %<-- Same params as previous run
%   P = cp_als_rand(X,2, 'num_samples', 10, 'printitn', 50, 'init', 'random', 
%       'maxiters', 1000, 'desired_fit', 0.98, 'mix', 1);
%   See also KTENSOR, TENSOR, SPTENSOR, TTENSOR.
%
%MATLAB Tensor Toolbox.
%Copyright 2015, Sandia Corporation.

% This is the MATLAB Tensor Toolbox by T. Kolda, B. Bader, and others.
% http://www.sandia.gov/~tgkolda/TensorToolbox.
% Copyright (2015) Sandia Corporation. Under the terms of Contract
% DE-AC04-94AL85000, there is a non-exclusive license for use of this
% work by or on behalf of the U.S. Government. Export of this data may
% require a license from the United States Government.
% The full license terms can be found in the file LICENSE.txt

    N = ndims(X);
    normX = norm(X);
    sz = size(X);
    num_elements = prod(sz);

    %% Set algorithm parameters from input or by using defaults
    params = inputParser;
    params.addParameter('init', 'random', @(x) (iscell(x) || ismember(x,{'random','nvecs'})));
    params.addParameter('dimorder', 1:N, @(x) isequal(sort(x),1:N));
    params.addParameter('printitn', 10, @isscalar);
    params.addParameter('mix', true, @islogical);
    params.addParameter('num_samples', ceil(10*R*log2(R)));
    params.addParameter('maxiters', 1000);
    params.addParameter('fit_samples', 2^14);
    params.addParameter('tol', 0, @isscalar);
    params.addParameter('desired_fit', 1, @(x) isscalar(x) & x > 0 & x <= 1);
    params.addParameter('epoch', 50)
    params.addParameter('window', 5);
    params.addParameter('truefit', false, @islogical);

    params.parse(varargin{:});

    %% Copy from params object
    fitchangetol = params.Results.tol;
    maxiters = params.Results.maxiters;
    dimorder = params.Results.dimorder;
    init = params.Results.init;
    printitn = params.Results.printitn;
    desired_fit = params.Results.desired_fit;   % cprand will terminate if this fit is reached (default 1)
    do_fft = params.Results.mix;
    window = params.Results.window;     
    num_samples = params.Results.num_samples;
    fit_samples = params.Results.fit_samples;
    epochsize = params.Results.epoch;
    truefit = params.Results.truefit;
    

 
    %% Error checking 
    %% Set up and error checking on initial guess for U.
    if iscell(init)
        Uinit = init;
        if numel(Uinit) ~= N
            error('OPTS.init does not have %d cells',N);
        end
        for n = dimorder(2:end)
            if ~isequal(size(Uinit{n}),[size(X,n) R])
                error('OPTS.init{%d} is the wrong size',n);
            end
        end
    else
        % Observe that we don't need to calculate an initial guess for the
        % first index in dimorder because that will be solved for in the first
        % inner iteration.
        if strcmp(init,'random')
            Uinit = cell(N,1);
            for n = dimorder(2:end)
                Uinit{n} = rand(sz(n),R);
            end
        elseif strcmp(init,'nvecs') || strcmp(init,'eigs') 
            Uinit = cell(N,1);
            for n = dimorder(2:end)
                Uinit{n} = nvecs(X,n,R);
            end
        else
            error('The selected initialization method is not supported');
        end
    end

    %% Set up for iterations - initializing U and the fit.
    U = Uinit;
    U_mixed = Uinit;
    fit = 0;
    diag_flips = [];

    if printitn>0
        if(do_fft)
            fprintf('CP-RAND-FFT: \n');
        else
            fprintf('CP_RAND: \n');
        end
    end

    %% Sample input tensor for stopping criterion
    fitsamples = min(nnz(X),fit_samples);
    [Xfit_subs, ~] = sample_all_modes(fitsamples, sz);
    Xfit_vals = X(Xfit_subs);

    if (do_fft)
        % Compute random diagonal D_n for each factor
        diag_flips = cell(N,1);
        for n = 1:N
            diag_flips{n} = (rand(sz(n),1)<0.5)*2-1;
        end

        X_mixed = X.data;
        % Mixing is equivalent to a series of TTMs with D_n, F_n
        % However, we can use bsxfun and fft to avoid matricizing.
        for n = N:-1:1
            % Reshape the diagonal flips into a 1*...*sz(n)*...*1 tensor
            % This lets us use bsxfun along the nth dimension.
            bsxdims = ones(1,N);
            bsxdims(n) = sz(n);
            flips = reshape(diag_flips{n},bsxdims);
            % fft(...,[],n) operates fiber-wise on dimension n
            X_mixed = fft(bsxfun(@times, X_mixed, flips),[],n);
        end
        X_mixed = tensor(X_mixed);
    else
        X_mixed = X; % no mixing
    end
    
    if (do_fft)
        % Mix factor matrices: U{i} = F{i}*D{i}*U{i}
        for i = 2:N
            U_mixed{i} = fft(bsxfun(@times,U{i},diag_flips{i}));
        end
    end
    %% Main Loop: Iterate until convergence
    maxfit = 0;
    niwi = 0; % num meta-iterations without improvement
    % ALS Loop
    for iter = 1:maxiters
        fitold = fit;
        
        for e = 1:epochsize            
            % Iterate over all N modes of the tensor
            for n = dimorder(1:end)
                
                mixinfo.dofft = do_fft;
                mixinfo.signflips = diag_flips;
                [Unew, ~, ~]= dense_sample_mttkrp(X_mixed,U_mixed,n,num_samples,mixinfo);
                
                if issparse(Unew)
                    Unew = full(Unew);   % for the case R=1
                end
                
                % Normalize each vector to prevent singularities in coefmatrix
                if iter == 1
                    lambda = sqrt(sum(abs(Unew).^2,1))'; %2-norm
                else
                    lambda = max( max(abs(Unew),[],1), 1 )'; %max-norm
                end
                
                Unew = bsxfun(@rdivide, Unew, lambda');
                U_mixed{n} = Unew;
                if (do_fft)
                    U{n} = real(bsxfun(@times, ifft(Unew), diag_flips{n}));
                else
                    U{n} = Unew;
                end
            end
        end

        P = ktensor(lambda, U);
        if normX == 0
            fit = norm(P)^2 - 2 * innerprod(X,P);
        else
            Pfit_vals = sample_ktensor(P, Xfit_subs);
            elem_mean = mean((Xfit_vals - Pfit_vals).^2);
            normDiff = sqrt(elem_mean * num_elements); % Approximate!
            fit = 1 -  normDiff / normX;
        end
                
        if fit > maxfit + fitchangetol
            niwi = 0;
            maxfit = fit;
            Psave = P; % Keep the best one seen so far!
        else
            niwi = niwi + 1;
        end
        
        if (fit > desired_fit) || (niwi >= window) 
            flag = 0;
        else
            flag = 1;
        end
        
        if (mod(iter,printitn)==0) || ((printitn>0) && (flag==0))
            fprintf(' Iter %2dx%d: f = %e niwi = %d\n', iter, epochsize, fit, niwi);
        end
        
        % Check for convergence
        if (flag == 0)
            break;
        end
    end
    %% Clean up final result
    % Arrange the final tensor so that the columns are normalized.
    P = Psave;
    P = arrange(P);
    P = fixsigns(P); % Fix the signs

    if truefit
        if normX == 0
            fit = norm(P)^2 - 2 * innerprod(X,P);
        else
            normresidual = sqrt( normX^2 + norm(P)^2 - 2 * innerprod(X,P) );
            fit = 1 - (normresidual / normX);%fraction explained by model
            Pfit_vals = sample_ktensor(P, Xfit_subs);
            Xfit_mean = mean((Xfit_vals - Pfit_vals).^2);
            testfit = 1 - sqrt(Xfit_mean*num_elements)/normX;
        end
        if printitn > 0
            fprintf(' Final fit = %e Final estimated fit = %e \n', fit, testfit);
        end
    else
        fit = -1;
    end

    output = struct;
    output.params = params.Results;
    output.iters = iter;
    output.fit = fit; 
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Sub-functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% MTTKRP That performs sampling after transforming X and the KR-product with an FFT
% And then solves using normal equations
function [V, Xsamp, Zsamp] = dense_sample_mttkrp(X,U,n,num_samples,mixinfo)

    N = ndims(X);

    if (N < 2) 
        error('MTTKRP is invalid for tensors with fewer than 2 dimensions'); 
    end

    if (length(U) ~= N) 
        error('Cell array is the wrong length'); 
    end

    if n == 1 
        R = size(U{2},2);
    else 
        R = size(U{1},2); 
    end

    for i = 1:N
        if i == n, continue; end
        if (size(U{i},1) ~= size(X,i)) || (size(U{i},2) ~= R)
            error('Entry %d of cell array is wrong size', i);
        end
    end
  
    dims = size(X);

    % Compute uniform samples for tensor and factor matrices
    [tensor_idx, factor_idx] = sample_mode_n(num_samples, dims, n);

    % Reshape the sampled tensor
    Xsamp = reshape(X(tensor_idx), dims(n), []);

    % Perform a sampled KRP
    Zsamp = skr(U{[1:n-1,n+1:N]}, factor_idx);
    
    % alternative
    V = Xsamp / Zsamp.';
    if (mixinfo.dofft)
       V = real(bsxfun(@times,ifft(V),mixinfo.signflips{n}));
       V = fft(bsxfun(@times,V,mixinfo.signflips{n}));
    end
    
    return;
end


% Sample Khatri-Rao Product of a cell array of factors
% Without forming the full KR Product
function P = skr(varargin)
    if iscell(varargin{1}) % Input is a single cell array
        A = varargin{1};
    else % Input is a sequence of matrices
        A = varargin(1:end-1);
    end

    numfactors = length(A);
    matorder = numfactors:-1:1;
    idxs = varargin{end};

    %% Error check on matrices and compute number of rows in result 
    ndimsA = cellfun(@ndims, A);
    if(~all(ndimsA == 2))
        error('Each argument must be a matrix');
    end

    ncols = cellfun(@(x) size(x, 2), A);
    if(~all(ncols == ncols(1)))
        error('All matrices must have the same number of columns.');
    end

    P = A{matorder(1)}(idxs(:,matorder(1)),:);
    for i = matorder(2:end)
        %P = P .*A{i}(idxs(:,i),:);
        P = bsxfun(@times, P, A{i}(idxs(:,i),:));
    end
end


% Random sample fibers in mode n from tensor X
% Generate the corresponding indices for the factor matrices as a tuple
function [tensor_idx, factor_idx] = sample_mode_n(num_samples, dims, n)
    D = length(dims);
    tensor_idx = zeros(num_samples, D);     % Tuples that index fibers in original tensor

    tensor_idx(:,n) = ones(num_samples, 1);
    for i = [1:n-1,n+1:D]
        % Uniformly sample w.r. in each dimension besides n
        tensor_idx(:,i) = randi(dims(i), num_samples, 1);
    end

    % Save indices to sample from factor matrices
    factor_idx = tensor_idx(:,[1:n-1,n+1:D]);

    % Expand tensor_idx so that every fiber element is included
    %tensor_idx = repelem(tensor_idx,dims(n),1); % not portable
    tensor_idx = kron(tensor_idx,ones(dims(n),1)); % portable
    tensor_idx(:,n) = repmat((1:dims(n))',num_samples,1);
    tensor_idx = tt_sub2ind(dims, tensor_idx);
end


% Random sample fibers in mode n from tensor X
% Generate the corresponding indices for the factor matrices as a tuple
function [subs, idxs] = sample_all_modes(num_samples, dims)
    D = length(dims);
    subs = zeros(num_samples, D);     % Tuples that index fibers in original tensor

    for i = 1:D
        % Uniformly sample w.r. in each dimension
        subs(:,i) = randi(dims(i), num_samples, 1);
    end

    % subs can be used to sample from factor matrices as well as the tensor
    subs = unique(subs,'rows'); %todo: do this more efficiently
    idxs = tt_sub2ind(dims, subs);
end


% Random sample fibers in mode n from tensor X
% Generate the corresponding indices for the factor matrices as a tuple
function [data] = sample_ktensor(P, subs)
    data = skr(P.u, subs) * P.lambda;
end