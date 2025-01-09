Faster and Better Quantum Software Testing through Specification Reduction and Projective Measurements, published in [TOSEM, 2025].
-
### **Keywords**: Test case reduction, Quantum software testing, Program specification, Projective measurements

This repository contains:

- **[experiments/](./experiments/)** - all related algorithms and source code for conducting the experiments:
- **[results/](./results/)** - all the raw data and postprocessed results from the experiments:
- **[data_analysis/](./data_analysis/)** - the analysis scripts and source files for creating figures and tables:

Description:
-
Quantum computing promises polynomial and exponential speedups in many domains, such as unstructured search and prime number factoring. However, quantum programs yield probabilistic outputs from exponentially growing distributions and are vulnerable to quantum-specific faults. Existing quantum software testing (QST) approaches treat quantum superpositions as classical distributions. This leads to two major limitations when applied to quantum programs: (1) an exponentially growing sample space distribution and (2) failing to detect quantum-specific faults such as phase flips. To overcome these limitations, we introduce a QST approach, which applies a reduction algorithm to a quantum program specification. The reduced specification alleviates the limitations (1) by enabling faster sampling through quantum parallelism and (2) by performing projective measurements in the mixed Hadamard basis. Our evaluation of 143 quantum programs across four categories demonstrates significant improvements in test runtimes and fault detection with our reduction approach. Average test runtimes improved from 169.9s to 11.8s, with notable enhancements in programs with large circuit depths (383.1s to 33.4s) and large program specifications (464.8s to 7.7s). Furthermore, our approach increases mutation scores from 54.5% to 74.7%, effectively detecting phase flip faults that non-reduced specifications miss. These results underline our approach's importance to improve QST efficiency and effectiveness.

ReduQt Overview:
-

Prerequisites:
-
