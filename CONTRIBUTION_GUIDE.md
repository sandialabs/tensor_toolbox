# Tensor Toolbox Contribution Guide

Contributions are welcome in the form of opening issues and merge requests with 
changes to the code or documentation itself.

## Submitting an Issue
Please use the GitLab issue tracking to submit any issues. Please provide as much
detailed information as possible so that the issue can be recreated by others.
If you also have the solution to the issues, please create a fork or branch with 
a solution per the instructions below.

## Changing to source code

1. Create an issue
2. Create a branch from the issue itself, if you're a developer. Otherwise, create a fork.
3. Make whatever changes you like.
4. Add cognizant test functions in the test directory, especially if you are fixing a bug or adding new functions.
4. If you are creating new functions
   1. Be sure to add a new m-file doc directory. The first two letters of the file control the order that the help is displayed. Use 'ZZ' until othewise advised.
   2. Publish the help in the doc/html directory.
   3. Update the main page and the XML via the create_doc command in the maintenance directory.
5. Submit a merge request with 'WIP' in the title


