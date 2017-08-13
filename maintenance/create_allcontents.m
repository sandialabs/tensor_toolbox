%% Create Content Files
clear
clc

%% Open main contents file
fidmain = fopen('../Contents.m','w');
fprintf(fidmain,'%% Tensor Toolbox (Sandia National Labs)\n'); 
fprintf(fidmain,'%% Version 3.0-dev %s\n', date); 
%fprintf(fidmain,'%%\n');
%fprintf(fidmain,'%% By Tamara G. Kolda, Brett W. Bader, and others.\n');
%fprintf(fidmain,'%% Also including...\n');
%fprintf(fidmain,'%% MET: Memory Efficient Tucker (Tamara Kolda and Jimeng Sun)\n');
%fprintf(fidmain,'%% \n'); 
fprintf(fidmain,'%% Tensor Toolbox for dense, sparse, and decomposed n-way arrays.\n'); 
fprintf(fidmain,'%% \n'); 
fprintf(fidmain,'%% Tensor Toolbox Classes:\n');
fprintf(fidmain,'%%   tensor     - Dense tensor.\n');
fprintf(fidmain,'%%   sptensor   - Sparse tensor.\n');
fprintf(fidmain,'%%   symtensor  - Symmetric tensor.\n');
fprintf(fidmain,'%%   ktensor    - Kruskal decomposed tensor.\n');
fprintf(fidmain,'%%   symktensor - Kruskal decomposed symmetric tensor.\n');
fprintf(fidmain,'%%   sumtensor  - Sum of different types of tensors.\n');
fprintf(fidmain,'%%   ttensor    - Tucker decomposed tensor.\n');
fprintf(fidmain,'%%   tenmat     - Tensor as matrix.\n');
fprintf(fidmain,'%%   sptenmat   - Sparse tensor as matrix.\n');
fprintf(fidmain,'%% \n'); 

%% Go through subdirs
subdir = {'@tensor','@sptensor','@ttensor','@ktensor','@tenmat','@sptenmat','@sumtensor','@symtensor','@symktensor'};
for j = 1:numel(subdir)
    
    % Extract contents
    C = create_dircontents(['../' subdir{j}]);
    
    % Write to subdir Contents file
    if isequal(subdir{j},'met')
        fidsub = fopen(['../' subdir{j} '/Contents.m'],'w');
        title = 'MET: Memory Efficient Tucker';
        for i = 1:numel(C)
            fprintf(fidsub,'%%   %s\n',C{i});
        end
        fclose(fidsub);
    end
    
    % Write to main class file
    if isequal(subdir{j}(1),'@')
        classname = subdir{j}(2:end);
        fprintf('Need to replace %s\n', classname);
        fidold = fopen(['../' subdir{j} '/' classname '.m'], 'r');
        fidnew = fopen(['../' subdir{j} '/tmp_' classname '.m'], 'w');

        % state == 0   => Looking for class header line.
        % state == 1   => Just found header line.
        % state == 2   => Found function line, nothing left to do.
        % state == 3   => Successful completion.
        state = 0;
        ptrn = sprintf('^%%%s', classname);
        while 1
            
            oldline = fgetl(fidold);
            
            % Check for end of file.
            if ~ischar(oldline)
                if (state < 2), error('Never found function line.'); end 
                state = 3;
                break
            end
            
            % Check for the class header line.
            if (state == 0) && (~isempty(regexpi(oldline,ptrn)))
                fprintf(fidnew, '%s\n', oldline);
                state = 1;
                
            % Check for the function line.
            elseif (state < 2) && (~isempty(regexp(oldline,'^function.*', 'once')))

                if state == 0
                    fprintf(fidnew,'%%%s\n', upper(classname));
                end
                fprintf(fidnew,'%%\n');
                fprintf(fidnew,'%%%s Methods:\n',upper(classname));
                for i = 1:numel(C)
                    fprintf(fidnew,'%%   %s\n',C{i});
                end
                fprintf(fidnew, '%%\n%% See also\n%%   TENSOR_TOOLBOX\n\n');
                fprintf(fidnew, '%s\n', oldline);
                state = 2;
                
            % Just copy the line directly.    
            elseif (state == 2)            
                fprintf(fidnew, '%s\n', oldline);
            end
        end
        
        fclose(fidold);
        fclose(fidnew);
    end
    %[s1,mess1,messid1] = movefile(['../' subdir{j} '/' classname '.m'], ['../' subdir{j} '/tmp_old_' classname '.m']);
    [s2,mess2,messid2] = movefile(['../' subdir{j} '/tmp_' classname '.m'], ['../' subdir{j} '/' classname '.m']);
end

%% Get contents of main directory
fprintf(fidmain,'%% Tensor Toolbox Functions:\n');
C = create_dircontents('..');
for i = 1:numel(C)
    fprintf(fidmain,'%%   %s\n',C{i});
end


%% Close main contents file
fclose(fidmain);
