# Unit tests

## Creating UERD test images in Matlab

As reference, we use the [Matlab simulation](https://codeocean.com/capsule/7800700/tree/v4) from Remi Cogranne.

We slightly modified the code such that the user can provide a seed to the random number generator. The array of random numbers is transposed in order to match the Python implementation. You can find the updated code in `assets/uerd/uerd.m`.

## Creating nsF5 test images in Matlab

As reference, we use the [Matlab simulation](https://dde.binghamton.edu/download/nsf5simulator/) provided by the Binghamton DDE lab.

Because we could not find a numpy equivalent to Matlab's `randperm`, our nsF5 method converts the permutation to Matlab's column-major ordering and stores this permutation to a file, which is then loaded and used by our modified Matlab implementation. See `assets/nsF5/nsF5.m` for details.

## Running the tests

The tests should be run from the project root directory.

```bash
python3 -m unittest
```

You can also run a specific test file.

```bash
python -m unittest test.test_uerd
```