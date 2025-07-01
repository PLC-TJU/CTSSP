# Common Temporal-Spectral-Spatial Patterns (CTSSP)

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2020b%2B-orange.svg)](https://www.mathworks.com/products/matlab.html)

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Quick Start](#-quick-start)
- [Related Research Resources](#-related-research-resources)
- [Data Availability](#-data-availability)
- [Citation](#-citation)
- [Contact](#-contact)
- [License and Attribution](#-license-and-attribution)

## Introduction

**CTSSP** 

The Common Temporal-Spectral-Spatial Patterns (CTSSP) algorithm is a unified framework for decoding motor imagery (MI) EEG signals by jointly optimizing temporal, spectral, and spatial feature extraction. It addresses key challenges in MI-BCI, including signal non-stationarity and cross-session variability, through three core innovations:

- **Multi-Scale Temporal Modeling**: Overlapping time windows capture dynamic neural transitions (e.g., ERD/ERS evolution) during motor imagery.

- **Adaptive Spectral Filtering**: Finite impulse response ([FIR](https://en.wikipedia.org/wiki/Finite_impulse_response)) filters suppress noise (e.g., EMG/EOG artifacts) while amplifying task-relevant rhythms (8–30 Hz mu/beta bands).

- **Low-Rank Spatial Projection**: Regularized spatial filters compress redundant features to mitigate overfitting and enhance generalization.

CTSSP achieves state-of-the-art classification accuracy while maintaining neurophysiological interpretability, as validated by its alignment with motor cortex activation patterns. Its robustness to parameter variations makes it suitable for real-world BCI applications.

**Key Features**: Brain-computer interface ([BCI](https://en.wikipedia.org/wiki/Brain%E2%80%93computer_interface)), motor imagery ([MI](https://en.wikipedia.org/wiki/Motor_imagery)), electroencephalography ([EEG](https://en.wikipedia.org/wiki/Electroencephalography)), Joint temporal-spectral-spatial optimization and cross-session.

## 📁 Project Structure
```plaintext
CTSSP/
├── matlab_version/                 # MATLAB implementation
│   ├── ctssp_modeling.m            # Model training
│   └── ctssp_classify.m            # Sample classification
├── ctssp.py                        # Core CTSSP algorithm implementation
├── main_cross_session_parallel.py  # Cross-session classification pipeline (parallelized)
├── main_within_subject_parallel.py # Within-subject classification pipeline (parallelized)

```

## 🔧 Installation & Setup

To install and run the project, please follow these steps:

### Python Version
1. Clone the repository locally
```bash
git clone https://github.com/PLC-TJU/CTSSP.git
cd CTSSP
```

2. Install the necessary dependencies
```bash
pip install -r requirements.txt
```

3. Install NeuroDecKit Toolbox
```bash
git clone https://github.com/PLC-TJU/NeuroDecKit.git
cd NeuroDecKit
python setup.py install
```

### MATLAB Version
1. Clone the repository locally
```bash
git clone https://github.com/PLC-TJU/CTSSP.git
cd CTSSP
```

2. Add `matlab_version` to the MATLAB path
```matlab
addpath(genpath('matlab_version'));
```

## 🚀 Quick Start

### Within-Subject Classification Example
```bash
python main_within_subject_parallel.py 
```

### Cross-Session Classification Example
```bash
python main_cross_session_parallel.py 
```

## 📚 Related Research Resources

We express our gratitude to the open-source community, which facilitates the broader dissemination of research by other researchers and ourselves. The coding style in this repository is relatively rough. We welcome anyone to refactor it to make it more efficient. Our model codebase is largely based on the following repositories:


- [<img src="https://img.shields.io/badge/GitHub-MOABB-b31b1b"></img>](https://github.com/NeuroTechX/moabb) An open science project aimed at establishing a comprehensive benchmark for BCI algorithms using widely available EEG datasets.
- [<img src="https://img.shields.io/badge/GitHub-NeuroDeckit-b31b1b"></img>](https://github.com/PLC-TJU/NeuroDeckit) A Python toolbox for EEG signal processing and BCI applications. It includes various preprocessing methods, feature extraction techniques, and classification algorithms.
- [<img src="https://img.shields.io/badge/GitHub-SBLEST-b31b1b"></img>](https://github.com/EEGdecoding/Code-SBLEST) Sparse Bayesian Learning for End-to-End Spatio-Temporal-Filtering-Based Single-Trial EEG Classification.
- [<img src="https://img.shields.io/badge/GitHub-Braindecode-b31b1b"></img>](https://github.com/braindecode/braindecode) Contains several deep learning models such as EEGNet, ShallowConvNet, and DeepConvNet, designed specifically for EEG signal classification. Braindecode aims to provide an easy-to-use deep learning toolbox.
- [<img src="https://img.shields.io/badge/GitHub-CSPNet-b31b1b"></img>](https://github.com/GeometricBCI/Tensor-CSPNet-and-Graph-CSPNet) Contains Tensor-CSPNet and Graph-CSPNet, two deep learning models for MI-EEG signal classification.
- [<img src="https://img.shields.io/badge/GitHub-LMDANet-b31b1b"></img>](https://github.com/MiaoZhengQing/LMDA-Code) A deep learning-based network for EEG signal classification. LMDA-Net combines various advanced neural network architectures to enhance classification accuracy.

## 📊 Data Availability

We used the following public datasets:

**Table 1** Details of all public datasets

| Dataset                                                |     Classes     | Sessions | Trials | Channels | Duration (s) | Subjects |
| :----------------------------------------------------- | :-------------: | :------: | :----: | :------: | :----------: | :------: |
| [BNCI2014001](https://doi.org/10.3389/fnins.2012.00055)| left/right hand |    2     |  288   |    22    |      4       |    9     |
| [Lee2019](https://doi.org/10.1093/gigascience/giz002)  | left/right hand |    2     |  200   |    62    |      4       |    54    |
| [Pan2023](https://doi.org/10.1088/1741-2552/ad0a01)    | left/right hand |    2     |  240   |    28    |      4       |    14    |
| [BNCI2014002](https://doi.org/10.1515/bmt-2014-0117)   | right hand/feet |    1     |  160   |    15    |      5       |    14    |
| [Cho2017](https://doi.org/10.1093/gigascience/gix034)  | left/right hand |    1     |  200   |    64    |      3       |    52    |
| **Total:**                                             |                 |          |        |          |              |  **143** |


## 📜 Citation
If you use this code, please cite:  

```bibtex
@article{pan2025ctssp,
  title={CTSSP: A Temporal-Spectral-Spatio Joint Optimization Algorithm for Motor Imagery EEG Decoding}, 
  author={Lincong, Pan and Kun, Wang and  Weibo, Yi and  Yang, Zhang and Minpeng, Xu and Dong, Ming},
  journal={TechRxiv},
  year={2025},
  doi={10.36227/techrxiv.174431208.89304915/v1}
}
```

## 🤝 Contact

If you have any questions or concerns, please contact us at:  
 - Authors: Lincong Pan
 - Institution: Tianjin University
 - Email: panlincong@tju.edu.cn

## 📝 License and Attribution

© 2024 Lincong Pan. MIT License.  
Please refer to the [LICENSE](./LICENSE) file for details on the licensing of our code.
