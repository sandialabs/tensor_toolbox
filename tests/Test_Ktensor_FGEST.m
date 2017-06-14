%%
info = create_problem('Size',[100 100 100],'Num_Factors',5, 'Noise', 0.01);
M = info.Soln;
M = normalize(M,0);
X = info.Data;

%% Compute error directly

ftrue = collapse( (full(M) - X).^2 );
relerr = sqrt(ftrue) / norm(X); % Should be equal to the noise level 

%% Compute error using ktensor/fg

[f,G] = fg(M,X,'Type','G','IgnoreLambda',true);
f = 2*f;
G = cellfun(@(g) 2*g,G,'UniformOutput',false);

%% Compute error using ktensor/fg_est
sz = size(X);
n = 1000;
idx = randi(prod(sz), n, 1);
subs = tt_ind2sub(sz, idx);
[fest,Gest,stuff] = fg_est(M,X,subs,'IgnoreLambda',true);

fprintf('True function value      = %e\n', ftrue);
fprintf('Estimated function value = %e (with %d samples)\n', fest, n);
fprintf('Relative error of true   = %e\n', relerr);
fprintf('Relative error of est    = %e\n', sqrt(fest) / norm(X));
fprintf('Samples                  = %f%%\n', 100*n / prod(sz));


%% Compute error using ktensor/fg with a mask
% Values should match those from ktensor/fg_est.

W = tensor(sptensor(subs,1,size(X)));
% W = sptensor(subs,1,size(X));  % Strangely gives something different
[fmask,Gmask] = fg(M,X,'Type','G','IgnoreLambda',true,'Mask',W);
fmask = 2 * (fmask / n * prod(sz));
Gmask = cellfun(@(g) 2*(g / n * prod(sz)),Gmask,'UniformOutput',false);

fprintf('Estimated function value = %e (using fg w/ mask)\n', fmask);
fprintf('Gradient cosine angle    = %e (b/w fg_est and fg w/ mask)\n', ...
    sum(sum(cell2mat(Gest).*cell2mat(Gmask)))/(norm(cell2mat(Gest),'fro')*norm(cell2mat(Gmask),'fro')));
fprintf('Gradient cosine angle    = %e (b/w fg_est and G)\n', ...
    sum(sum(cell2mat(Gest).*cell2mat(G)))/(norm(cell2mat(Gest),'fro')*norm(cell2mat(G),'fro')));