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
