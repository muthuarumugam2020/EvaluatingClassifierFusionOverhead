# EvaluatingClassifierFusionOverhead
Evaluating Classifier Fusion Overhead and Feature Redundancy in Static Android Malware Detection


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MATLAB Version](https://img.shields.io/badge/MATLAB-Online-blue.svg)](https://matlab.mathworks.com/)

This repository contains the official MATLAB source code and implementation details accompanying the conference paper:

> **"Evaluating Classifier Fusion Overhead and Feature Redundancy in Static Android Malware Detection"**  
> **Authors:** Muthukumar A, Dr. A. Devi  

---

## Repository Structure

drebin215.csv      				# Data File
CheaterColumnIdentifier.m     	# Utility program to identify the cheater column when processing data
ComputationalOverheadTable.m	# This program process and creates the data for the Computational Overhead
DetectionPerformanceGraphV2.m	# This program generates graph on performance detection
DetectionPerformanceTableV2.m   # This program generates table on Performance detection
ReliefFgraphV3.m                # This program generates graph on ReliefF
LICENSE                 		# MIT open-source license file
README.md               		# Project documentation (this file)
						

---

## System Requirements & Dependencies

* **Software Platform:** MATLAB (Tested and developed on Matlab Online-blue).
* **Required MATLAB Toolboxes:** {'MATLAB'}    {'Deep Learning Toolbox'}    {'Statistics and Machine Learning Tool'}
* Deep Learning Toolbox (for constructing and training the ABiLSTM network)
* Statistics and Machine Learning Toolbox (for statistical evaluation and data manipulation)

---

## Dataset Preparation

The framework utilizes benchmark Android malware repositories (Drebin-215 Dataset). Due to file size constraints, the full raw dataset is not hosted directly in this repository. 

**Full Dataset Execution:** Download the complete dataset from the [https://doi.org/10.6084/m9.figshare.5854653]. 
Preprocessed files is added xlsx file is in the folder, file name drebin215.xlsx .

---

## Step-by-Step Execution Guide

1. Login to MATLAB Online 
2. Import the project into Matlab Online
3. The code is setup to execute individual steps 
4. Database also in csv format
5. Execute the required step, each code loads the databsae
6. When prompted select "Change Folder" option to execute from that folder 
7. The code will process the data and will generate tables, and graphs

---

## Citation

If you use this software, algorithm implementation, or dataset structure in your academic research, please cite our work using the following BibTeX entry:

```bibtex
@misc{Muthukumar2026GBOA,
  author    = {Muthukumar A and Dr. A. Devi},
  title     = {Evaluating Classifier Fusion Overhead and Feature Redundancy in Static Android Malware Detection},
  year      = {2026},
  note      = {},
  url       = {}
}
```

---

## License

This project is open-source and licensed under the terms of the **MIT License**. See the [LICENSE](LICENSE) file for more information.
