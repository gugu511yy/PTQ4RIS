# PTQ4RIS: Post-Training Quantization for Referring Image Segmentation

## Abstract

[//]: # (This repository contains the code for the paper **"PTQ4RIS: An Effective and Efficient Post-Training Quantization Framework for Referring Image Segmentation"**.)

Referring Image Segmentation (RIS), aims to segment the object referred by a given sentence in an image by understanding both visual and linguistic information. However, existing RIS methods tend to explore top-performance models, disregarding considerations for practical applications on resources-limited edge devices. This oversight poses a significant challenge for on-device RIS inference. To this end, we propose an effective and efficient post-training quantization framework termed PTQ4RIS.
Specifically, we first conduct an in-depth analysis of the root causes of performance degradation in RIS model quantization and propose dual-region quantization (DRQ) and reorder-based outlier-retained quantization (RORQ) to address the quantization difficulties in visual and text encoders. Extensive experiments on three benchmarks with different bits settings (from 8 to 4 bits) demonstrates its superior performance. Importantly, we are the first PTQ method specifically designed for the RIS task, highlighting the feasibility of PTQ in RIS applications. 

[//]: # (![PTQ4RIS Framework]&#40;image.png&#41;)

## Code Availability

The code for this framework will be made publicly available upon acceptance of our paper. We will provide detailed instructions for installation and usage, as well as support for any issues that arise. Stay tuned for updates!

[//]: # (## Getting Started)

[//]: # ()
[//]: # (Instructions for setting up the environment, installing dependencies, and running the code will be provided in the final release. For now, please refer to the following sections:)

[//]: # ()
[//]: # (- [Installation]&#40;#installation&#41; - How to set up the environment)

[//]: # (- [Usage]&#40;#usage&#41; - How to use the code)

[//]: # (- [Contributing]&#40;#contributing&#41; - How to contribute to the project)

[//]: # (- [)
