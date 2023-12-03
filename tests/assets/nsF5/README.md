# Creating nsF5 test images in Matlab

As reference, we use the [Matlab simulation](https://dde.binghamton.edu/download/nsf5simulator/) provided by the Binghamton DDE lab.

Because we could not find a numpy equivalent to Matlab's `randperm`, our nsF5 method converts the permutation to Matlab's column-major ordering and stores this permutation to a file, which is then loaded and used by our modified Matlab implementation. See `nsF5.m` for details.
