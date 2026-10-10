# External solvers

C++ baselines live here as git submodules, so the exact version used for each
result is recorded in the repo history. Built binaries are git-ignored.

```
git submodule add https://github.com/Kei18/lacam3 external/lacam3
cmake -S external/lacam3 -B external/lacam3/build
cmake --build external/lacam3/build
```

`mapf_gt.solvers.LaCAM` looks for `external/lacam3/build/main`, or the path in
the `LACAM_BIN` environment variable. Follow the build steps in the lacam3
README if these differ.
