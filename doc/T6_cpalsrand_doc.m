%% Randomized least squares for fitting CP to sparse count data

%% Set up a sample problem
% Pick the size, rank, noise level, and collinearity
sz = [400 350 300];
R = 10;
ns = 0.01;
coll = 0.9;

info = create_problem('Size', sz, 'Num_Factors', R, 'Noise', ns, 'Factor_Generator', @(m,n) matrandcong(m,n,coll), 'Lambda_Generator', @ones);

% Extract data and solution
X = info.Data;
M_true = info.Soln;

%% Compare CP-ALS, CPRAND, and CPRAND-MIX

% Compute solutions
tic; [M_als, Uinit, out] = cp_als(X,R,'maxiters',500,'printitn',10); als_time = toc

% Random methods should achieve similar fit in less time
tic; [M_rand, Uinit, out] = cp_als_rand(X,R,'mix',0,'maxiters',500,'printitn',10,'window',10); rand_time = toc
tic; [M_randmix, Uinit, out] = cp_als_rand(X,R,'mix',1,'maxiters',500,'printitn',10,'window',10); randmix_time = toc

% Score the solutions
als_score = score(M_als, M_true, 'greedy', true, 'lambda_penalty', false)
rand_score = score(M_rand, M_true, 'greedy', true, 'lambda_penalty', false)

% Mixing isn't typically needed for randomly generated data
randmix_score = score(M_randmix, M_true, 'greedy', true, 'lambda_penalty', false)


tic; [M_randmix, Uinit, out] = cp_als_rand(X,R,'num_samples',100,'maxiters',500,'printitn',10,'window',10); toc;
tic; [M_randmix, Uinit, out] = cp_als_rand(X,R,'num_samples',1000,'maxiters',500,'printitn',10,'window',10); toc;
