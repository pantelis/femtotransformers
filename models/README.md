# Triton Model Repository

This directory contains models for NVIDIA Triton Inference Server.

## Directory Structure

Place your models in subdirectories following Triton's model repository structure:

```
models/
├── model1/
│   ├── config.pbtxt
│   └── 1/
│       └── model.pt
└── model2/
    ├── config.pbtxt
    └── 1/
        └── model.onnx
```

## Model Configuration

Each model needs a `config.pbtxt` file. Example for PyTorch model:

```
name: "my_model"
platform: "pytorch_libtorch"
max_batch_size: 8
input [
  {
    name: "input__0"
    data_type: TYPE_FP32
    dims: [ 3, 224, 224 ]
  }
]
output [
  {
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ 1000 ]
  }
]
```

## Getting Started

1. Add your trained models to this directory
2. Create appropriate config.pbtxt files
3. Start the services: `docker-compose up`
4. Access Triton at http://localhost:8000
