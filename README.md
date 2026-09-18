# spMAE: A spatially guided masked autoencoder for scalable spatial multi-omics integration
## Overview
In this work, we introduce spMAE, a self-supervised framework for spatial multi-omics integration that decouples spatial context modeling from graph-based propagation. spMAE leverages spatially guided masked reconstruction to implicitly encode spatial structure, uses feature-level modality fusion to capture heterogeneous regulatory signals, and incorporates contrastive learning to enhance representation consistency across modalities.
## Installation
We tested our code on a server running Ubuntu 22.04.5 LTS, equipped with NVIDIA 3090 GPUs.

git clone https://github.com/IgniteOmics/spMAE
cd spMAE
conda create -n spMAE python=3.10.20
conda activate spMAE
pip install -r requirements.txt
