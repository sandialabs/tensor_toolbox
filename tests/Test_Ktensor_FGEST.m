%%
info = create_problem('Size',[100 100 100],'Num_Factors',5, 'Noise', 0);
M = info.Soln;
M = normalize(M,0);
X = info.Data;

%% Compute error directly

ftrue = collapse( (full(M) - X).^2 );
relerr = sqrt(ftrue) / norm(X); % Should be equal to the noise level 

%% Compute error using ktensor/fg

%falt = 2*fg(M,X,'Type','G')

%% Compute error using ktensor/fg_est
sz = size(X);
n = 1000;
idx = randi(prod(sz), n, 1);
subs = tt_ind2sub(sz, idx);
[fmean,stuff] = fg_est(M,X,subs);
fest = fmean * prod(sz);


fprintf('True function value      = %e\n', ftrue);
fprintf('Estimated function value = %e (with %d samples)\n', fest, n);
fprintf('Relative error of true   = %e\n', relerr);
fprintf('Relative error of est    = %e\n', sqrt(fest) / norm(X));
fprintf('Samples                  = %f%%\n', 100*n / prod(sz));
