# Tensor Toolbox Contribution Guide

## Checklist

- [ ] Submit an issue for the change, providing as much detailed information as possible. For bug reports, please provide enough information to reproduce the problem.
- [ ] Create a fork of the code.
- [ ] At any point, create a work-in-progress merge request with this checklist and WIP in the header.
- [ ] Make your contribution.
- [ ] Create or update comments for the m-files, following the style of the existing files. Be sure to explain all code options.
- [ ] For any major new functionality, add HTML documentation in the `doc` directory. The first two letters of the file control the order that the help is displayed. Use 'ZZ' until othewise advised. Use the MATLAB `publish` command to create a new file in `doc\html`. Go to `maintenance` directory and run `create_doc` to add the new HTML page to the Tensor Toolbox HTML and XML help files.
- [ ] Create or update tests in the `tests` directory, especially for bug fixes or newly added code.
- [ ] For any new files, go to `maintenance` directory and run `create_allcontents` to add the files to the appropriate class file or main contents page.
- [ ] Update `RELEASE_NOTES.txt` with any significant bug fixes or additions.
- [ ] Confirm that all tests (including existing tests) pass in `tests` directory.



