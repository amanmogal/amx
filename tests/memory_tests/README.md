# Memory Tests

This test suite contains pipelines, which are executables. 
Memory tests measuring memory required for the use cases and fail when memory
usage exceeds a pre-defined level.

## Prerequisites

To build the time tests, you need to have OpenVINO™ installed or build from source.

## Measure Time

To build and run the tests, open a terminal, set OpenVINO™ environment and run
the commands below:

1. Build tests:
``` bash
mkdir build && cd build
cmake .. && make memory_tests
```

If you don't have OpenVINO™ installed you need to have the `build` folder, which
is created when you configure and build OpenVINO™ from sources:

``` bash
cmake .. -DInferenceEngine_DIR=$(realpath ../../../build) && make memory_tests
```

2. Install tests:
``` bash
make install tests
```

3. Run test:
``` bash
./scripts/run_memorytest.py tests/memtest_infer -m model.xml -d CPU
```

4. Run several configurations using `pytest`:
``` bash
pytest ./test_runner/test.py --exe tests/memorytest_infer

# For parse_stat testing:
pytest ./scripts/run_memorytest.py
```
