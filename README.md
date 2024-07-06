# VA-Count

# [ECCV 2024]Zero-shot Object Counting with Good Exemplars
## News
[[Project page]([https://github.com/HopooLinZ/VA-Count/] 
An officical implementation of "Zero-shot Object Counting with Good Exemplars" (Accepted by ECCV 2024).   
## Overview
Our work proposes a domain-adaptive framework for crowd counting based on optimal transport (OT).
The training process explains the acquisition of 𝑀𝑇0 and 𝑀𝑇, respectively. The inference process is divided into two stages. In the individual-level measurement stage (stage 1), region information from both the source and target domains is collected. The source domain model 𝑀𝑆 and target domain model 𝑀𝑇0 trained with pseudo-labels from 𝑀𝑆 are used for distribution perception, yielding the source domain distribution 𝐷𝑆 and target domain distribution 𝐷𝑇. The distance matrix 𝐶 is calculated using SSIM to measure the distance between each distribution and extended to form the cost matrix 𝐶. In the domain-level alignment stage (stage 2), we use the Sinkhorn algorithm with iterative updates to obtain the optimal transfer matrix solution 𝑃 and the final simulated distribution. We fine-tune the initial model 𝑀𝑇0 using the simulated distribution to obtain the target domain model 𝑀𝑇.
![vis](![image]()
)
Our code will be available soon！
