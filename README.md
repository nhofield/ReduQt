Quantum Software Testing Experiments
This repository contains the replication package for the experiments and analyses described in the paper: 
Faster and Better Quantum Software Testing through Specification Reduction and Projective Measurements, published in [TOSEM, 2025].

Directory Structure
experiments/

Experiment1/: Contains the source files for Experiment 1 and the study subjects.
Experiment2/: Contains the source files for Experiment 2 and a mutation generator.
results/

Stores the output from Experiment 1 and Experiment 2, along with postprocessing files.
data_analysis/

RQ1/: Contains source files for figures, tables, and data analysis for Research Question 1.
RQ2/: Contains source files for Research Question 2.
RQ3/: Contains source files for Research Question 3.
Procedure to Perform a New Run
Step 1: Experiment 1
Navigate to the Experiment1 directory and run:

bash
Kopier kode
python experiment1_main.py
This will execute the greedy approach for Experiment 1.

Copy the generated result files from:

bash
Kopier kode
Experiment1/results_experiment1/
to:

bash
Kopier kode
results/results_experiment1_greedy/
Repeat steps 1 and 2 for the random approach.

Step 2: Experiment 2
Navigate to the Experiment2 directory and run:

bash
Kopier kode
python mutant_generator.py
This generates mutants for Experiment 2.

Execute the greedy approach for Experiment 2 by running:

bash
Kopier kode
python experiment2_main.py
Copy the results from:

bash
Kopier kode
Experiment2/results_experiment2/
to:

bash
Kopier kode
results/results_experiment2_greedy/
Repeat steps 5 and 6 for the random approach.

Step 3: Postprocessing
Run the following scripts in the results/postProcessing/ directory:
bash
Kopier kode
python add_median_runtimes_combine_results.py
python create_run_all_data.py
Step 4: Data Analysis
Analyze data for each research question by running the following scripts in data_analysis/:

RQ1:

bash
Kopier kode
python full_tables_reduction_rate.py
python full_tables_runtime.py
RQ2:

bash
Kopier kode
python full_tables_runtime.py
RQ3:

bash
Kopier kode
python full_tables_mutation_score.py
python summary_tables_mutation_score.py