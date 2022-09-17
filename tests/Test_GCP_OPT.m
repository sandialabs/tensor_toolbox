% Testing gcp_opt
classdef Test_GCP_OPT < matlab.unittest.TestCase
    methods (Test)

        function SparseBinary(testCase)
            [output,X] = evalc('create_problem_binary([10 15 20],3);');
            [output,M] = evalc('gcp_opt(X,10,''type'',''binary'',''maxiters'',2);');
            testCase.verifyTrue(contains(output, 'End Main Loop'));
        end % SparseBinary
        
    end % methods
end % classdef Test_GCP_OPT
