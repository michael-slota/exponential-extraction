# Data Folder

This folder contains the input `.dat` files used by the notebook workflow.

The current analysis expects:

- Tab-separated text files
- A first column containing delay values in ns
- A second column containing complex-valued echo data written with `i` notation

Example:

```text
600    418568.71+90376.138i
632    795365.41-15357.282i
664    692311.66-1341.3526i
```

Temperatures are extracted from file names of the form `_*K`, for example `010_Tm_8K.dat`.
