# EGPFNet 

This project provides the code for 'A Novel EPI-Guided Network with Progressive Fusion for Light Field Reconstruction', ESWA, 2025. paper link is
https://www.sciencedirect.com/science/article/pii/S0957417425025813

# Requirements
1. Python 3.8
2. Pytorch 2.3.1
3. Cuda 11.8

# Training
- Run:
    `python train.py`

# Test
We provide the pre-trained model for real-world and synthetic scenes in `./log/EGPF_Lytro_2x2-7x7.pth.tar` and `./log/EGPF_HCI_2x2-7x7.pth.tar`.
- Run:
    `python test_on_datasets.py`

# Citation

`@article{EGPFNet2025, 
    title={A Novel EPI-Guided Network with Progressive Fusion for Light Field Reconstruction}, 
    author={Wang, Baoshuai and Chen, Yilei and Huang Xinpeng and An Ping},  
    journal={Expert Systems with Applications},
    year={2025}
}`
