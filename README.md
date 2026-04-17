# 2210fcd_public
**Fusion Strategies and Performance–Efficiency Trade-offs in 3D Multi-Modal Brain MRI Classification for Focal Cortical Dysplasia**

ABSTRACT

Focal cortical dysplasia (FCD) is a major cause of drug-resistant epilepsy, but MRI-based detection remains chal-
lenging because lesions are often subtle. Multi-sequence three-dimensional MRI models can improve classification
by combining complementary information from T1-weighted and FLAIR sequences, but this also increases compu-
tational cost, memory footprint, and inference latency. This project presented a hardware-aware analysis of fusion
strategies for 3D brain MRI classification for FCD detection under constrained GPU resources. Using a 3D ResNet-
based pipeline on a public presurgical MRI dataset, I compared single-modality baselines, early fusion, late fusion, and
a memory-optimized late-fusion variant across full-resolution and reduced-resolution inputs, as well as FP32 and au-
tomatic mixed precision (AMP) execution. Evaluation included both predictive performance and system-level metrics,
including training time, validation time, data time, compute time, peak training GPU memory usage, and inference
latency.

The experiments showed that fusion strategy was both a modeling decision and a systems decision. Among unimodal
baselines, FLAIR consistently outperformed T1 in AUC in most settings. Among multimodal models, early fusion
with AMP provided the best overall trade-off between predictive quality and efficiency. System-level analysis showed
consistent efficiency gains from AMP and input size reduction, with input resolution emerging as the dominant factor
in computational cost. Late fusion achieved competitive performance in some configurations but incurred much higher
runtime and memory cost. In the full-resolution FP32 setting, standard late fusion was infeasible at batch size 2 due to
out-of-memory failure; a memory-optimized version restored feasibility but remained expensive. Overall, early fusion
with AMP at full resolution provided the best practical operating point on a single L40S GPU.

ACKNOWLEGEMENT AND NEW CONTRIBUTION

This work builds upon a baseline implementation developed by Daniel Rafique at the Intelligent Medical Informatics Computing Systems (IMICS) Lab at the Hospital for Sick Children and the University of Toronto, which includes MRI image pre-processing and a baseline early-fusion two-channel (T1 + FLAIR) 3D ResNet model. 

For this project, I significantly extended the baseline with a focus on hardware-aware optimization and performance analysis. I developed a flexible experimental framework supporting multiple model configurations, including single-modality models (T1-only and FLAIR-only), standard late fusion, and a memory-optimized late fusion architecture that incorporates activation checkpointing and selective layer freezing to reduce GPU memory usage. In addition, I introduced configurable system-level optimizations such as mixed precision training (AMP vs FP32), variable input resolutions, and batch size scaling, enabling systematic evaluation of accuracy–efficiency trade-offs under constrained hardware settings. I implemented enhanced checkpointing strategies, evaluation through dynamic threshold selection using Youden’s J statistic, as well as detailed runtime profiling, including measurement of training/validation time, data loading vs compute time, peak GPU memory usage, and inference latency, allowing for comprehensive characterization of system performance. Finally, I conducted experiments, analyzed results, and wrote the report.
