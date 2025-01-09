import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

def Vargha_Delaney_A( R_1, m_number, n_number  ):
    """
    Takes---
    R_1: rank sum of the first data group we are comparing
    m_number: number of observations in the first sample
    n_number: number of observations in the second sample
    Returns---
    VD_A_effect_size: probability that running algorithm A
    gives a higher performance measure M, value between
    0 and 1.
    For instance, given a performance measure M.
    VD_A_effect_size = 0.7 says that we would get higher M
    values from running A 70& of the time
    From Arcuri - Briand: A practical guide for using statistical
    tests to assess randomized algorithms in software engineering.
    """
    return ( R_1/m_number - (m_number + 1)/2 )/n_number

def check_VA_scaled(VA_A):
    abs_VA_scaled = abs( (VA_A - 0.5)*2 )

    if abs_VA_scaled < 0.147:
        # negligible
        category = "(N)"
    elif abs_VA_scaled >= 0.147 and abs_VA_scaled <= 0.33:
        # small
        category = "(S)"
    elif abs_VA_scaled >= 0.33 and abs_VA_scaled <= 0.474:
        # medium
        category = "(M)"
    else:
        # large
        category = "(L)"

    return category
df_pass_counts_filename = os.path.join("tables", "reductionRate", "Summary", "reduction_rate_pass_counts.csv")
df_pass_counts = pd.read_csv( df_pass_counts_filename )

df_pass_counts_filename_r = os.path.join("tables", "reductionRate", "Summary", "reduction_rate_pass_counts_r.csv")
df_pass_counts_r = pd.read_csv( df_pass_counts_filename_r )

#filename = r"optimal_bases.txt"

current_directory = os.getcwd()
grandparent_directory = os.path.dirname(os.path.dirname(current_directory))
dir_to_results = os.path.join(grandparent_directory, "results", "postProcessing")
dir_to_results_experiment1 = os.path.join(grandparent_directory, "results")


filename_factoring = os.path.join( dir_to_results_experiment1, r"results_experiment1_greedy", "greedy_bases.txt" )
filename_random    = os.path.join( dir_to_results_experiment1, r"results_experiment1_random", "random_bases.txt" )

repetitions = 100
program_categories = ["all", "aamp", "qwalk", "var", "gs"]
qubit_ranges = [ [2, 15], [6, 9], [3, 5], [2, 8], [3, 15] ]
depth_ranges = [ [3, 684], [30, 684], [14, 154], [3, 70], [5, 17] ]
alpha = 0.05

df_factoring = pd.read_csv( filename_factoring, delimiter = ", ", engine='python')
df_random    = pd.read_csv( filename_random, delimiter = ", ", engine='python')

df_factoring['reduction'] = df_factoring['reduction'].apply(lambda x: 100*(1 - x) )
df_random['reduction']    = df_random['reduction'].apply(lambda x: 100*(1 - x) )

df_table_total = pd.DataFrame()

df_table = pd.DataFrame(index = ["\\textbf{Category}", "\\textbf{\\#Qubits}", "\\textbf{Depth}", "\\textbf{Factoring [\\%]}", "\\textbf{Random [\\%]}", "\\textbf{\\%(NN)}", "\\textbf{\\%(S)}", "\\textbf{\\%(M)}", "\\textbf{\\%(L)}", "\\textbf{\\% fail or (N)}"], columns = range(len(program_categories)))

filename_latex_table = os.path.join( "tables", "reductionRate", "Summary", "latex_summary_table_reduction.txt" )

with open( filename_latex_table, "w") as outfile:

    for program_category_index in range( len(program_categories) ):

        curr_program_category = program_categories[ program_category_index ]
        curr_qubit_range = qubit_ranges[ program_category_index ]
        curr_depth_range = depth_ranges[ program_category_index ]

        if curr_program_category != "all":
            df_curr_factoring = df_factoring[ df_factoring['program'].str.startswith( curr_program_category ) ]
            df_curr_random    = df_random[ df_random['program'].str.startswith( curr_program_category ) ]
        else:
            df_curr_factoring = df_factoring
            df_curr_random    = df_random

        #if curr_program_category == "gs":
            #unique_progs = list(df_curr_factoring['program'].unique())[:-1]
        #else:
        unique_progs = list(df_curr_factoring['program'].unique())


        prog_to_exclude1 = "aamp_q6_6_R"
        prog_to_exclude2 = "aamp_q7_5_R"
        if prog_to_exclude1 in unique_progs:
            unique_progs.remove(prog_to_exclude1)

        if prog_to_exclude2 in unique_progs:
            unique_progs.remove(prog_to_exclude2)

        # to count the number of passes
        number_of_passes = 0
        number_of_S_s = 0
        number_of_M_s = 0
        number_of_L_s = 0
        number_of_N_s = 0

        number_of_non_zero_iqr_factoring = 0
        number_of_non_zero_sd_factoring  = 0
        number_of_non_zero_iqr_random = 0
        number_of_non_zero_sd_random  = 0


        reduction_array_factoring = np.array( df_curr_factoring["reduction"] )
        reduction_array_random = np.array( df_curr_random["reduction"] )

        bin_counts_factoring = np.zeros(11)
        bin_counts_random = np.zeros(11)

        bin_counts_factoring[0] = np.sum( np.where(reduction_array_factoring == 0)[0] )
        bin_counts_random[0] = np.sum( np.where(reduction_array_random == 0)[0] )
        interval_value = 10

        for bin_index in range(10):

            bin_counts_factoring[bin_index + 1] = np.sum( np.where((reduction_array_factoring > interval_value - 10) & (reduction_array_factoring <= interval_value))[0] )
            bin_counts_random[bin_index + 1]    = np.sum( np.where((reduction_array_random > interval_value - 10) & (reduction_array_random <= interval_value))[0] )
            interval_value += 10




        # fetch average reduction rate
        reduction_rate_average_factoring = df_curr_factoring["reduction"].mean()
        reduction_rate_average_random    = df_curr_random["reduction"].mean()


        # Percent of non-zero reductions
        non_zero_count = (df_curr_factoring['reduction'] != 0).sum()
        print(f'Percent of non-zero reductions for {curr_program_category}: {non_zero_count / len(df_curr_factoring["reduction"]) * 100:.2f}%')

        # Calculating and printing the percentage of reductions greater than specific thresholds
        thresholds = [50, 60, 70, 80, 90]
        for threshold in thresholds:
            count_above_threshold = (df_curr_factoring['reduction'] > threshold).sum()
            percent_above_threshold = count_above_threshold / len(df_curr_factoring['reduction']) * 100
            print(f'Percent of reductions > {threshold} for {curr_program_category}: {percent_above_threshold:.2f}%')

        # Assuming df_curr_random is another DataFrame you're working with
        # Percent of non-zero reductions
        non_zero_count_r = (df_curr_random['reduction'] != 0).sum()
        print(f'Percent of non-zero reductions for {curr_program_category}, in random: {non_zero_count_r / len(df_curr_random["reduction"]) * 100:.2f}%')

        # Repeating the process for df_curr_random
        for threshold in thresholds:
            count_above_threshold_r = (df_curr_random['reduction'] > threshold).sum()
            percent_above_threshold_r = count_above_threshold_r / len(df_curr_random['reduction']) * 100
            print(f'Percent of reductions > {threshold} for {curr_program_category}, in random: {percent_above_threshold_r:.2f}%')


        # fetch stds
        reduction_rate_std_factoring = df_curr_factoring["reduction"].std()
        reduction_rate_std_random    = df_curr_random["reduction"].std()


        # assign values to table dataframe
        df_table.at["\\textbf{\\#Qubits}", program_category_index] = f"{curr_qubit_range}"

        # assign depth
        df_table.at["\\textbf{Depth}", program_category_index] = f"{curr_depth_range}"

        # assign program category
        df_table.at["\\textbf{Category}", program_category_index] = f"{curr_program_category}"

        # assign average and std
        df_table.at["\\textbf{Factoring [\\%]}", program_category_index] = f"{reduction_rate_average_factoring:.1f} $\\pm$ {reduction_rate_std_factoring:.1f}"
        df_table.at["\\textbf{Random [\\%]}", program_category_index] = f"{reduction_rate_average_random:.1f} $\\pm$ {reduction_rate_std_random:.1f}"

        reduction_distribution_factoring = np.array(df_curr_factoring["reduction"])
        reduction_distribution_random = np.array(df_curr_random["reduction"])


        # Perform the Mann-Whitney U test
        u_statistic, p_value = stats.mannwhitneyu(reduction_distribution_factoring, reduction_distribution_random)

        # Combine the samples
        combined = np.concatenate((reduction_distribution_factoring, reduction_distribution_random))

        # Rank the combined array
        ranks = stats.rankdata(combined)

        # Sum the ranks for the first sample
        R1 = np.sum( ranks[ :len( reduction_distribution_factoring ) ] )

        # fetch Vargha_Delaney_A effect size
        VA_A = Vargha_Delaney_A( R1, len(reduction_distribution_factoring), len(reduction_distribution_random) )

        # fetch effect size category
        VA_category = check_VA_scaled(VA_A)

        # increment category counter
        if VA_category == "(N)":
            number_of_N_s += 1

        total_rows = len(unique_progs)

        # see if test passed (1) or failed (0)
        if p_value <= alpha and VA_category != "(N)":
            bin_category = "(NN)"

            test_result = f"p-val $\leq$ {alpha}, {bin_category}"
            number_of_passes += 1

            if VA_category == "(S)":
                number_of_S_s += 1

            if VA_category == "(M)":
                number_of_M_s += 1

            if VA_category == "(L)":
                number_of_L_s += 1


        else:
            bin_category = "(N)"
            test_result = f"p-val $>$ {alpha}, {bin_category}"

        number_of_NN = df_pass_counts.loc[program_category_index, "\%(NN)"]
        number_of_S  = df_pass_counts.loc[program_category_index, "\%(S)"]
        number_of_M  = df_pass_counts.loc[program_category_index, "\%(M)"]
        number_of_L  = df_pass_counts.loc[program_category_index, "\%(L)"]

        number_of_NN_r = df_pass_counts_r.loc[program_category_index, "\%(NN)"]
        number_of_S_r  = df_pass_counts_r.loc[program_category_index, "\%(S)"]
        number_of_M_r  = df_pass_counts_r.loc[program_category_index, "\%(M)"]
        number_of_L_r  = df_pass_counts_r.loc[program_category_index, "\%(L)"]

        df_table.at["\\textbf{\\%(NN)}", program_category_index] = f'{number_of_NN}/{number_of_NN_r}'
        df_table.at["\\textbf{\\%(S)}", program_category_index]  = f'{number_of_S}/{number_of_S_r}'
        df_table.at["\\textbf{\\%(M)}", program_category_index]  = f'{number_of_M}/{number_of_M_r}'
        df_table.at["\\textbf{\\%(L)}", program_category_index]  = f'{number_of_L}/{number_of_L_r}'
        # add # fails or (N)
        df_table.at["\\textbf{\\% fail or (N)}", program_category_index]  = df_pass_counts.loc[program_category_index, "Total Not Pass or (N)"]

        print("------------------")
        print("| Category =", program_categories[program_category_index], "|")
        print("------------------")

        df_table_transposed = df_table.T


    df_latex = df_table_transposed.to_latex(index=False, escape=False)

    outfile.write(df_latex + "\n")
