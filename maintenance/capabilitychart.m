%% Create capability chart for all the tensor toolbox classes

%% Analyze files to find what we have
D = dir('../');
nd = length(D);

tf_class = false(nd,1);
for i = 1:nd
    if D(i).name(1) == '@'
        tf_class(i) = true;
    end
end

tf_func = false(nd,1);
for i = 1:nd
    if (length(D(i).name) > 3) && strcmp(D(i).name(end-1:end),'.m')
        tf_func(i) = true;
    end
end

%% Create table of functions per class

% number of classes
nclasses = sum(tf_class);

% classnames
classnames = cell(nclasses,1);
[classnames{:}] = deal(D(tf_class).name);
for i = 1:nclasses
    classnames{i} = classnames{i}(2:end);
end
pi = [8 3 6 1 9 5 4 7 2];
classnames = classnames(pi);

% get directory contents for each class (omitting constructor)
classmembers = cell(nclasses,1);
functionnames = {};
for i = 1:nclasses
    C = dir(['../@' classnames{i}]);
    nc = length(C);
    tf_tmp = false(nd,1);
    for j = 1:nc
        if (length(C(j).name) > 3) && strcmp(C(j).name(end-1:end),'.m') ...
                && ~strcmp(C(j).name(1:end-2),classnames{i})
            tf_tmp(j) = true;
        end
    end
    C = C(tf_tmp);
    C = arrayfun(@(x) x.name(1:end-2), C, 'UniformOutput', false);
    classmembers{i} = C;
    functionnames = union(functionnames,C);
end

% get membership array
nfunctions = length(functionnames);
tf = false(nfunctions, nclasses);
for i = 1:nclasses
    tf(:,i) = ismember(functionnames, classmembers{i});
end

%% Print out results
cnl = cellfun(@length,classnames);
fprintf('function     ');
for i = 1:nclasses
    fprintf('%s ', classnames{i});
end
fprintf('\n');
for i = 1:nfunctions
    fprintf('%-12s', functionnames{i});
    for j = 1:nclasses
        for k = 1:4
            fprintf(' ');
        end
        if tf(i,j)
            fprintf('X');
        else
            fprintf('-');
        end
        for k = 5:cnl(j)
            fprintf(' ');
        end
    end
    fprintf('\n');
end
        
        

