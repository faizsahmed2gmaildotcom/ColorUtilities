# ColorUtilities

Color and pattern detection for images of fabrics

Run the following in terminal for NVIDIA GPUs on Fedora:

    export LD_LIBRARY_PATH=\$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

## training-data folder structure
```text
training-data/
├── shirt/
│   ├── main/
│   ...
├── jacket/
│   ├── main/
│   ...
└── weave/
    └── main/
```
Every deepest directory contains directories of classes, which each contain the training data.