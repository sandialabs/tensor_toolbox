classdef Test_Ktensor_FG < matlab.unittest.TestCase

    properties (TestParameter)
        
    end
    
    methods (Test)
        
        function FgDense(testCase)
            
            % Contruct a problem
            sz = [20 15 10];
            r = 4;
            M = ktensor(@rand, sz, r);
            X = tensor(@(x) double(rand(x) > 0.5), sz);
            
            % Compute just function value
            f = fg(M,X);
            testCase.verifyTrue(isscalar(f));

            % Compute function value & all gradients
            [f,G] = fg(M,X);
            testCase.verifyTrue(isscalar(f));
            testCase.verifyTrue(isa(G,'ktensor'));
            testCase.verifyEqual(size(G), size(M));
            testCase.verifyEqual(ncomponents(G), ncomponents(M));
            %
            [f,G] = fg(M,X,'GradVec',true);
            testCase.verifyTrue(isscalar(f));
            testCase.verifyTrue(isnumeric(G));
            vecsz = [r*(sum(sz)+1) 1];
            testCase.verifyEqual(size(G),vecsz);
            
            
            % Compute function value & all gradients
            [f,G] = fg(M,X,'IgnoreLambda',true);
            testCase.verifyTrue(isscalar(f));
            testCase.verifyTrue(iscell(G));
            for k = 1:ndims(X)                
                testCase.verifyEqual(size(G{k},1), size(M,k));
                testCase.verifyEqual(size(G{k},2), r);
            end
            %
            [f,G] = fg(M,X,'IgnoreLambda',true,'GradVec',true);
            testCase.verifyTrue(isscalar(f));
            testCase.verifyTrue(isnumeric(G));
            vecsz = [r*sum(sz) 1];
            testCase.verifyEqual(size(G),vecsz);

            % Compute function value & lambda gradient
            [f,G] = fg(M,X,'GradMode',0);
            testCase.verifyTrue(isscalar(f));
            testCase.verifyEqual(size(G), [r 1]);
            %
            [f,G] = fg(M,X,'GradMode',0,'GradVec',true);
            testCase.verifyTrue(isscalar(f));
            testCase.verifyEqual(size(G), [r 1]);
            
            % Compute function value & mode gradient
            for k = 1:ndims(X)
                [f,G] = fg(M,X,'GradMode',k);
                testCase.verifyTrue(isscalar(f));
                testCase.verifyEqual(size(G), [sz(k) r]);
                [f,G] = fg(M,X,'GradMode',k,'IgnoreLambda',true);
                testCase.verifyTrue(isscalar(f));
                testCase.verifyEqual(size(G), [sz(k) r]);
                %
                [f,G] = fg(M,X,'GradMode',k,'GradVec',true);
                testCase.verifyTrue(isscalar(f));
                testCase.verifyEqual(size(G), [sz(k)*r 1]);
            end
        end
        
        
        
        function FgSparse(testCase)
            
            % Contruct a problem
            sz = [20 15 10];
            r = 4;
            M = ktensor(@rand, sz, r);
            X = sptensor(@ones, sz,.2);
            W = sptensor(@ones, sz,.2);
            
            % Compute just function value
            f = fg(M,X,'Mask',W);
            testCase.verifyTrue(isscalar(f));
            
            % Compute function value & all gradients
            [f,G] = fg(M,X,'Mask',W);
            testCase.verifyTrue(isscalar(f));
            testCase.verifyTrue(isa(G,'ktensor'));
            testCase.verifyEqual(size(G), size(M));
            testCase.verifyEqual(ncomponents(G), ncomponents(M));
            %
            [f,G] = fg(M,X,'Mask',W,'GradVec',true);
            testCase.verifyTrue(isscalar(f));
            testCase.verifyTrue(isnumeric(G));
            vecsz = [r*(sum(sz)+1) 1];
            testCase.verifyEqual(size(G),vecsz);
            
            
            % Compute function value & all gradients
            [f,G] = fg(M,X,'Mask',W,'IgnoreLambda',true);
            testCase.verifyTrue(isscalar(f));
            testCase.verifyTrue(iscell(G));
            for k = 1:ndims(X)
                testCase.verifyEqual(size(G{k},1), size(M,k));
                testCase.verifyEqual(size(G{k},2), r);
            end
            %
            [f,G] = fg(M,X,'Mask',W,'IgnoreLambda',true,'GradVec',true);
            testCase.verifyTrue(isscalar(f));
            testCase.verifyTrue(isnumeric(G));
            vecsz = [r*sum(sz) 1];
            testCase.verifyEqual(size(G),vecsz);
            
            % Compute function value & lambda gradient
            [f,G] = fg(M,X,'Mask',W,'GradMode',0);
            testCase.verifyTrue(isscalar(f));
            testCase.verifyEqual(size(G), [r 1]);
            %
            [f,G] = fg(M,X,'Mask',W,'GradMode',0,'GradVec',true);
            testCase.verifyTrue(isscalar(f));
            testCase.verifyEqual(size(G), [r 1]);
            
            % Compute function value & mode gradient
            for k = 1:ndims(X)
                [f,G] = fg(M,X,'Mask',W,'GradMode',k);
                testCase.verifyTrue(isscalar(f));
                testCase.verifyEqual(size(G), [sz(k) r]);
                [f,G] = fg(M,X,'Mask',W,'GradMode',k,'IgnoreLambda',true);
                testCase.verifyTrue(isscalar(f));
                testCase.verifyEqual(size(G), [sz(k) r]);
                %
                [f,G] = fg(M,X,'Mask',W,'GradMode',k,'GradVec',true);
                testCase.verifyTrue(isscalar(f));
                testCase.verifyEqual(size(G), [sz(k)*r 1]);
            end
        end

    end
end