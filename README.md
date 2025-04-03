# Common Temporal-Spectral-Spatial Patterns (CTSSP)

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![GitHub](https://img.shields.io/badge/GitHub-NeuroDecKit-red.svg)](https://github.com/PLC-TJU/NeuroDeckit)

## Introduction

**CTSSP** 
Official implementation of the CTSSP algorithm for motor imagery EEG decoding.  
**Key Features**: Brain-computer interface (BCI), motor imagery (MI), electroencephalography (EEG), Joint temporal-spectral-spatial optimization and cross-session.

## 📁 Project Structure
```plaintext
CTSSP/
├── ctssp.py                        # Core CTSSP algorithm implementation
├── main_cross_session_parallel.py  # Cross-session classification pipeline (parallelized)
├── main_within_subject_parallel.py # Within-subject classification pipeline (parallelized)

```

## ⚙️ Installation & Setup

To install and run the project, please follow these steps:

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

## 🚀 Quick Start

### Within-Subject Classification Example
```bash
python main_within_subject_parallel.py 
```

### Cross-Session Classification Example
```bash
python main_cross_session_parallel.py 
```

## 🤝 Related Research Resources

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

```
@article{pan2025ctssp,
  title={Common Temporal-Spectral-Spatial Patterns for Motor Imagery EEG Decoding}, 
  author={Lincong, Pan and Kun, Wang and Minpeng, Xu and Dong, Ming},
  journal={...},
  year={2025},
  volume={...},
  pages={...},
  doi={...},
}
```

## License and Attribution

© 2024 Lincong Pan. MIT License.

Please refer to the [LICENSE](./LICENSE) file for details on the licensing of our code.

