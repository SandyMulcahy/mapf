# External solvers

C++ baselines live here as git submodules, so the exact version used for each
result is recorded in the repo history. Built binaries are git-ignored.

## LaCAM (lacam3)

On a Mac you need Apple's compiler and CMake once:

```
xcode-select --install     # skip if already installed
brew install cmake
```

Then, from the repo root:

```
git submodule add https://github.com/Kei18/lacam3 external/lacam3
git submodule update --init --recursive
cmake -S external/lacam3 -B external/lacam3/build -DCMAKE_BUILD_TYPE=Release
cmake --build external/lacam3/build -j
```

`mapf_gt.solvers.LaCAM` finds `external/lacam3/build/main` automatically, or
uses the path in the `LACAM_BIN` environment variable.

Check it works:

```
pytest tests/test_lacam.py
```

When cloning this repo somewhere new, run
`git submodule update --init --recursive` and build again.
