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
Every deepest directory contains directories of classes. Each class directory contains that class's training data.
This hierarchical structure used to train multiple models works best due to the irregular size of my dataset.

## Training Results
### Main Classifier
![Main Classifier](<plots/models shirting mainfinal.png>)
### Check
![Check Classifier](<plots/models shirting check mainfinal.png>)
### Stripes
![Stripes Classifier](<plots/models shirting stripes mainfinal.png>)
### Dots
![Dots Classifier](<plots/models shirting dots mainfinal.png>)
^Need to gather more training data for dots

## Data Sources
pinterest.com

tartanregister.gov.uk

hollandandsherry.com

divij.com
