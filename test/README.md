# Unit tests

## Creating UERD test images in Matlab

As reference, we use the [Matlab simulation](https://codeocean.com/capsule/7800700/tree/v4) from Rémi Cogranne.

We slightly modified the code such that the user can provide a seed to the random number generator. The array of random numbers is transposed in order to match the Python implementation. You can find the updated code in `assets/uerd/uerd.m`.

## Creating nsF5 test images in Matlab

As reference, we use the [Matlab simulation](https://dde.binghamton.edu/download/nsf5simulator/) provided by the Binghamton DDE lab.

Because we could not find a numpy equivalent to Matlab's `randperm`, our nsF5 method converts the permutation to Matlab's column-major ordering and stores this permutation to a file, which is then loaded and used by our modified Matlab implementation. See `assets/nsF5/nsF5.m` for details.

## Creating J-UNIWARD test images in C++

As reference, we use the [Matlab implementation](http://dde.binghamton.edu/download/stego_algorithms/download/J-UNIWARD_matlab_v11.zip) and the [C++ implementation](http://dde.binghamton.edu/download/stego_algorithms/download/J-UNIWARD_linux_make_v11.tar.gz) provided by the Binghamton DDE lab.
To compare the costmap between the original C++ and our Python implementation, we modified the C++ implementation to store the adjusted costmap in a file.

```cpp
// Compute the costmap
base_cost_model* model = (base_cost_model *)new cost_model(coverStruct, config);

// Dump to costmap to a text file
fs::path costmapPath(stegoPath.parent_path() / stegoPath.stem());
costmapPath += ".costmap";

std::ofstream cost_output;
cost_output.open(costmapPath.c_str(), std::ios::out | std::ios::binary);
cost_output.write(reinterpret_cast<const char*>(&model->costs[0]), sizeof(float) * 3 * model->rows * model->cols);
cost_output.close();
```

## Running the tests

The tests should be run from the project root directory.

```bash
python3 -m unittest
```

You can also run a specific test file.

```bash
python -m unittest test.test_uerd
```