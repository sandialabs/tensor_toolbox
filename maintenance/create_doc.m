%% Create HTML documentation
clear

%% Get list of files
fprintf('\nReading the contents of the directory.\n');
files = dir('../doc');
if (numel(files) == 0)
    error('ERROR: Running in wrong directory!');
end
addpath('../doc');

%% Find the documentation files
fprintf('\nFinding the documentation files that should be processed.\n');
docs = struct; k = 1;
for i = 1:numel(files)
    if regexp(files(i).name,'.*_doc.m$')
        fprintf('File %d: %s\n',k,files(i).name);
        docs(k).name = files(i).name;
        k = k+1;
    end
end

%% Extract titles
fprintf('\nExtracing the document titles.\n');
for i = 1:numel(docs)
    % Open file
    fid = fopen(['../doc/' docs(i).name]);
    if (fid == -1)
        error('Unable to open file');
    end
    % Find title line
    while 1
        tline = fgetl(fid);
        if regexp(tline,'^%%.*')
            foo = regexp(tline,'\w+.*','match');
            docs(i).title = foo{1};
            fprintf('Title for %-30s: %s\n',docs(i).name,docs(i).title);
            break;
        end
    end
    fclose(fid);
end

%% Create tensor_toolbox_product_page.html
fprintf('\nGenerating a new tensor_toolbox_product_page.html.\n');
infid = fopen('tensor_toolbox_product_page_template.html','r');
outfid = fopen('../tensor_toolbox_product_page.html','w');
fprintf(outfid,'<!-- DO NOT MODIFY THIS FILE. IT IS AUTOMATICALLY GENERATED. -->\n');
while 1
    tline = fgetl(infid);
    if tline == -1
        break;
    elseif  regexp(tline,'INSERT LIST HERE')
        fprintf(outfid','<ul>\n');
        for i = 1:numel(docs)
            fprintf(outfid,'<li><a href="doc/html/%s.html">%s</a></li>\n',docs(i).name(1:end-2),docs(i).title);
        end
        fprintf(outfid','</ul>\n');

    else
        fprintf(outfid,'%s\n',tline);
    end
end
fclose(infid);
fclose(outfid);
%% Create helptoc.xml
fprintf('\nGenerating a new helptoc.xml.\n');
infid = fopen('helptoc_template.xml','r');
outfid = fopen('../helptoc.xml','w');
while 1
    tline = fgetl(infid);
    if tline == -1
        break;
    elseif  regexp(tline,'INSERT LIST HERE')
        for i = 1:numel(docs)
            fprintf(outfid,'<tocitem image="$toolbox/matlab/icons/pageicon.gif"\n');
            fprintf(outfid,' target="doc/html/%s.html">%s</tocitem>\n',docs(i).name(1:end-2),docs(i).title);
        end

    else
        fprintf(outfid,'%s\n',tline);
    end
end
fclose(infid);
fclose(outfid);
%% Publish the HTML 
% (run last because it creates a lot of variables in the working space)  
fprintf('\nPublishing the documentation.\n');
for iii = 1:numel(docs)
    %name = ['../doc/' docs(iii).name];
    name = [docs(iii).name];
    fprintf('Publishing file %s ...\n',name);
    html = publish(name);
    fprintf('File has been published to %s\n',html);
    keep iii docs
end
