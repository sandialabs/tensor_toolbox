function [best_score, A, flag, best_perm] = score_new(A,B,varargin)
%SCORE Checks if two ktensors match except for permutation.
%   
%   SCORE(A,B) returns the score of the match between A and B where
%   A is trying to be matched against B.
%  
%   We define matching as follows. If A and B are single component ktensors 
%   that have been normalized so that their weights are lambda_a and
%   lambda_b, then the score is defined as
%   
%      score = penalty * (a1'*b1) * (a2'*b2) * ... * (aR'*bR),
%     
%   where the penalty is defined by the lambda values such that
%
%      penalty = 1 - abs(lambda_a - lambda_b) / max(lamdba_a, lambda_b).
%
%   The score of multi-components ktensors is a normalized sum of the
%   scores across the best permutation of the components of A. A can have
%   more components than B --- any extra components are ignored in terms of
%   the matching score.     
%
%   [SCORE,A] = SCORE(...) also returns A which has been normalized
%   and permuted to best match B. 
%
%   [SCORE,A,FLAG] = SCORE(...) also returns a boolean to indicate
%   a match according to a user-specified threshold.
%
%   [SCORE,A,FLAG,PERM] = SCORE(...) also returns the permutation
%   of the components of A that was used to best match B. 
%
%   SCORE(A,B,'param',value,...) takes the following parameters...
%
%      'lambda_penalty' - Boolean indicating whether or not to consider the
%      lambda values in the calculations. Default: true
%
%      'threshold' - Threshold specified in the formula above for
%      determining a match. Default: 0.99^N where N = ndims(A)
%
%      'greedy' - Boolean indicating whether or not to solve the problem
%      exactly or just do a greedy matching. Default: false
%      The exact algorithm uses a weighted bipartite matching local
%      function.
%
%   Examples
%   A = ktensor([2; 1; 2], rand(3,3), rand(4,3), rand(5,3));
%   B = ktensor([2; 4], ones(3,2), ones(4,2), ones(5,2));
%   score(A, B) %<--score(B,A) does not work: B has more components than A
%   score(A, B, 'greedy', false) %<--Check all permutations
%   score(A, B, 'lambda_penalty', false) %<--Without lambda penalty
%
%   This method is described in G. Tomasi and R. Bro, A Comparison of
%   Algorithms for Fitting the PARAFAC Model, Computational Statistics &
%   Data Analysis, Vol. 50, No. 7, pp. 1700-1734, April 2006,
%   doi:10.1016/j.csda.2004.11.013.
%  
%   See also KTENSOR.
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

%  E. Acar & T. Kolda, 2010.
%  Update by G. Ballard, B. Cobb, & S. Ma 2018

%% Make sure A and B are ktensors
if ~isa(A,'ktensor')
    A = ktensor(A);
end
if ~isa(B,'ktensor')
    B = ktensor(B);
end

%% Error checking
if ~isequal(size(A),size(B))
    error('Size mismatch');
end

%% Set-up
N = ndims(A);
RA = ncomponents(A);
RB = ncomponents(B);

%% We're matching components in A to B
if (RA < RB)
    error('Tensor A must have at least as many components as tensor B.');
end

%% Parse parameters
params = inputParser;
params.addParameter('lambda_penalty', true, @islogical);
params.addParameter('greedy', false, @islogical);
params.addParameter('threshold', 0.99^N, @(x)(x<1));
params.parse(varargin{:});

%% Make sure columns of factor matrices in A and B are normalized
A = normalize(A);
B = normalize(B);

%% Compute all possible vector-vector congruences.

% Compute every pair for each mode
Cbig = tenzeros([RA,RB,N]);
for n = 1:N
    Cbig(:,:,n) = abs(A.u{n}' * B.u{n});
end

% Collapse across all modes using the product
C = double(collapse(Cbig,3,@prod));

%% Calculate penalty based on differences in the Lambda's
% Note that we are assuming the the lambda value are positive because the
% ktensor's were previously normalized.
if (params.Results.lambda_penalty)
    P = zeros(RA,RB);
    for ra = 1:RA
        la = A.lambda(ra);
        for rb = 1:RB
            lb = B.lambda(rb);
            P(ra,rb) = 1 - (abs(la-lb) / max(abs(la),abs(lb)));
        end
    end
    C = P.*C;
end

%% Option to do greedy matching
if (params.Results.greedy)
    
    best_perm = zeros(1,RA);
    best_score = 0;
    for r = 1:RB
        [~,idx] = max(C(:));
        [i,j] = ind2sub([RA RB], idx);
        best_score = best_score + C(i,j);
        C(i,:) = -10;
        C(:,j) = -10;
        best_perm(j) = i;
    end
    best_score = best_score / RB;
    flag = 1;
    
    % Rearrange the components of A according to the best matching
    foo = 1:RA;
    tf = ismember(foo,best_perm);
    best_perm(RB+1:RA) = foo(~tf);
    A = arrange(A, best_perm);
    return;


%% Option to perform optimal matching in polynomial time
else
    
    % compute the best matching and scores using local function
    [matching,scores] = wbm(C); 
    best_score = sum(scores);
    
    % add unmatched indices to best permutation
    best_perm = [matching(:,1)',find(~ismember(1:RA,matching(:,1)))];

    % check if minimum score in matching is greater than threshold
    if min(scores) >= params.Results.threshold
        flag = 1;
    else
        flag = 0;
    end
    % Rearrange the components of A according to the best matching
    A = arrange(A, best_perm);
    best_score=best_score/RB;
    return;
end



end

function [matching, weights] = wbm(LR,varargin)
%WBM solves the weighted bipartite matching problem
%   
%   WBM(LR) returns the pairs of vertex indices of the optimal matching of
%   the bipartite graph with biadjacency matrix LR (vertex indices of 
%   matching correspond to row and column indices of LR)
%
%   [matching, weights] = WBM(...) also returns the weights of the edges in
%   the matching
%
%   WBM(LR,'param',value) takes the following parameters...
%
%      'max' - Boolean indicating whether to compute the maximum matching
%      or the minimum matching. Default: true
%
%   This method is described in textbook (pp. 404-414) by Jon Kleinberg and 
%   Eva Tardos. 2005. Algorithm Design. Addison-Wesley Longman, USA.
%
%   The idea of the algorithm is to build a bipartite digraph with edge
%   weights specified by LR.  We add a source vertex s and a sink vertex t,
%   and initially all edges are directed from s to L to R to t.  Each step
%   of the algorithm consists of finding the shortest augmenting path from 
%   s to t to increase the size of the matching by 1.  Edges currently in
%   the matching are directed from R to L and have negative weight.
%   Matched vertices are also disconnected from s and t.  Because some
%   edges are negative, the Bellman-Ford algorithm is used inside MATLAB's
%   built-in shortestpath function.  The worst-case running time in theory  
%   is O( min( m^3n, mn^3 ) ), which can be improved by maintaining all
%   non-negative edge weights (and using Dijkstra's algorithm), but in 
%   practice the time is dominated by manipulation of the graph rather than
%   finding the shortest paths.

    %% Parse parameters
    params = inputParser;
    params.addParameter('max', true, @islogical);
    params.parse(varargin{:});

    % preprocess if max matching desired (alg designed for min matching)
    if params.Results.max
        % negate entries and add constant value to make positive
        maxVal = max(max(LR));
        LR = maxVal - LR + 1; 
    end
    
    % check to make sure that all entries are positive
    if any(LR <= 0)
        error('WBM: expected input matrix to have positive entries')
    end

    % extract size of the input
    [n,m]=size(LR);

    % initialize the adjacency matrix with 2 extra nodes
    A = zeros(n+m+2);
    s = m+n+1;
    t = m+n+2;
    A(1:n,n+1:n+m) = LR;
    A(s,1:n) = 1;
    A(n+1:n+m,t) = 1;

    % create bipartite graph to use MATLAB's built-in graph functions
    G = digraph(A);

    % repeatedly find shortest augmenting path and add to matching
    for j = 1:min(n,m)

        % get shortest path
        v = shortestpath(G,s,t);

        % get indices of edges along path (ignoring s and t)
        idx = findedge(G,v(2:end-2),v(3:end-1));

        % negate the edges along path
        G.Edges.Weight(idx) = -G.Edges.Weight(idx);

        % flip direction of edges along path
        G = flipedge(G,idx);

        % remove edges between s and t and newly matched vertices
        G = rmedge(G,[s,v(end-1)],[v(2) , t]);
    end

    % extract the indices of edges in matching (all have negative weight)
    idx = find(G.Edges.Weight < 0);

    % extract matching from G based upon idx 
    matching = G.Edges.EndNodes(idx,:);
    
    % flip order and adjust index to be row by col
    temp = matching(:,2);
    matching(:,2) = matching(:,1) - n;
    matching(:,1) = temp;

    % return original (positive) weights of edges in matching
    weights = -G.Edges.Weight(idx);

    % for max case readjust cost based on preprocessing values
    if params.Results.max
        weights = maxVal - weights + 1;
    end
        
end



