# itm
Dataset and code for Intelligent Traffic Monitoring with Distributed Acoustic Sensing.

## Structure
```
├── CNN1
│   ├── data
│   │   ├── img  # inputs of dataset1 (linear features with noise, 1024x1024)
│   │   │   ├── 1.dat
│   │   │   ├── 2.dat
│   │   │   ├── 3.dat
│   │   │   .
│   │   │   .
│   │   │   .
│   │   │   └── 200.dat
│   │   └── lbl  # labels of dataset1 (clean straight lines, 1024x1024)
│   │       ├── 1.dat
│   │       ├── 2.dat
│   │       ├── 3.dat
│   │       .
│   │       .
│   │       .
│   │       └── 200.dat
│   └── src
│       ├── dataset.py
│       ├── train.py
│       └── unet.py
└── CNN2
    ├── data
    │   ├── htimg  # inputs of dataset2 (point-like features with noise, 1451x301)
    │   │   ├── 1.dat
    │   │   ├── 2.dat
    │   │   ├── 3.dat
    │   │   .
    │   │   .
    │   │   .
    │   │   └── 200.dat
    │   └── htlbl  # labels of dataset2 (separate points, 1451x301)
    │       ├── 1.dat
    │       ├── 2.dat
    │       ├── 3.dat
    │       .
    │       .
    │       .
    │       └── 200.dat
    └── src
        ├── dataset.py
        ├── train.py
        └── unet.py
```

## Dependencies
- python      3.9.18
- pytorch     2.1.0
- numpy       1.26.2
- matplotlib  3.8.0
