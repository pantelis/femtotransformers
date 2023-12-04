# FemtoTransformer package

This package attempts to teach foundational concepts of transformers and attention. The goal is to provide a simple and easy to understand implementation of a transformer model. The package is designed to be used as a teaching tool only and at this point only contains largely [Karpathy's tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY) and in addition we have used some utilities from Hugging Face transformer's library. 

## Installation

To use this package with GPU support, you'll need to install CUDA and cuDNN. Make sure you have the appropriate GPU drivers and CUDA toolkit installed. This repo at this time is tested only to work with the pytorch docker container. In VS Code you can launch the repo with remote container support. After the container is launched youto install femtotransformer, run the following command:

```bash
make build-install
```

This package will be updated to support pip install in the future and pushed to pypi. 

