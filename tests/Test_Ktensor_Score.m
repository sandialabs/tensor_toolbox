% Testing @ktensor/score function
classdef Test_Ktensor_Score < matlab.unittest.TestCase
    methods (Test)

        function Same(testCase)
            K=ktensor({randn(10,10),randn(12,10),randn(3,10),randn(5,10)});
            sGreedy = score(K,K,'greedy',true); % greedy should work here
            sExact = score(K,K,'greedy',false);
            testCase.verifyEqual(1, sGreedy, 'RelTol', 1e-14);
            testCase.verifyEqual(1, sExact, 'RelTol', 1e-14);
        end
        
        function Permuted(testCase)
            n = 5;
            A = randn(6,n); B = randn(9,n); C = randn(4,n);
            P = eye(n); perm = randperm(n); P = P(perm,:);
       
            K1 = ktensor({A,B,C});
            K2 = ktensor({A*P,B*P,C*P}); % make K2 permuted version of K1
            [sGreedy,~,~,pGreedy] = score(K1,K2,'greedy',true); % greedy should work here
            [sExact,~,~,pExact] = score(K1,K2,'greedy',false);
            testCase.verifyEqual(1, sGreedy, 'RelTol', 1e-14);
            testCase.verifyEqual(1, sExact, 'RelTol', 1e-14);
            testCase.verifyEqual(pGreedy,pExact);
            pt(perm) = 1:n; % get transpose of original permutation
            testCase.verifyEqual(pExact,pt);
            K1p = arrange(K1,pExact);
            [~,~,~,pExact] = score(K1p,K2); % after arranging, expect to return identity perm
            testCase.verifyEqual(pExact,1:n);
        end
        
        function Recovered(testCase)
            rng(1);
            m = 10; n = 8;
            K=ktensor({randn(10,m),randn(12,m),randn(15,m),randn(17,m)});
            % cp_als should recover n of the m original factors, up to a 
            % threshold of .9 or so
            S=cp_als(full(K),n,'init','nvecs','printitn',0);
            [scr,~,flag,~] = score(K,S,'greedy',false,'threshold',.9);
            testCase.verifyEqual(1, flag); % threshold specifies min compontent-wise score
            testCase.verifyGreaterThanOrEqual(scr, .98); % scr is a weighted average of comp-wise scores
        end
            
        
    end
end