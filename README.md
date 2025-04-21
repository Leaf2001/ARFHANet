# ARFC-WAHNet: Adaptive Receptive Field Convolution and Wavelet-Attentive Hierarchical Network for Infrared Small Target Detection
## Introduction
Infrared small target detection (ISTD) is critical in both civilian and military applications. However, the limited texture and structural information in infrared images makes
accurate detection particularly challenging. Although recent deep learning-based methods have improved performance, their use of conventional convolution kernels limits adaptability to complex scenes and diverse targets. Moreover, pooling operations often cause feature loss and insufficient exploitation of image information. To address these issues, we propose an adaptive receptive field convolution and wavelet-attentive hierarchical network for infrared small target detection (ARFC-WAHNet). This network incorporates a multi-receptive field feature interaction convolution (MRFFIConv) module to adaptively extract discriminative features by integrating multiple convolutional branches with a gated unit. A wavelet frequency enhancement downsampling (WFED) module leverages Haar wavelet transform and frequency-domain reconstruction to enhance target features and suppress background noise. Additionally, we introduce a high-low feature fusion (HLFF) module for integrating low-level details with high-level semantics, and a global median enhancement attention (GMEA) module to improve feature diversity and expressiveness via global attention. Experiments on public datasets SIRST, NUDT-SIRST, and IRSTD-1k demonstrate that ARFC-WAHNet outperforms recent state-of-the-art methods in both detection accuracy and robustness, particularly under complex backgrounds.

## Overview
![](https://github.com/Leaf2001/ARFC-WAHNet/blob/main/assert/overview.png)  

## Visual Results
![](https://github.com/Leaf2001/ARFC-WAHNet/blob/main/assert/visual%20result.png)
