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

def f_complexity(n_qubits):
    return int(round( n_qubits*( n_qubits + 1 )/2 ) )

#filename = r"optimal_bases.txt"
df_pass_counts_filename = os.path.join("tables", "runtime", "Summary", "runtime_pass_counts.csv")
df_pass_counts = pd.read_csv( df_pass_counts_filename )

df_pass_counts_filename_r = os.path.join("tables", "runtime", "Summary", "runtime_pass_counts_r.csv")
df_pass_counts_r = pd.read_csv( df_pass_counts_filename_r )


b_add = "\\textbf{"
b_end = "}"

current_directory = os.getcwd()
grandparent_directory = os.path.dirname(os.path.dirname(current_directory))
dir_to_results = os.path.join(grandparent_directory, "results", "postProcessing")
dir_to_results_experiment1 = os.path.join(grandparent_directory, "results")


filename_factoring = os.path.join( dir_to_results_experiment1, r"results_experiment1_factoring", "factoring_runtimes.txt" )
filename_random    = os.path.join( dir_to_results_experiment1, r"results_experiment1_random", "random_runtimes.txt" )

repetitions = 100
program_categories = ["all", "aamp", "qwalk", "var", "gs"]
qubit_ranges = [ [2, 15], [6, 9], [3, 5], [2, 8], [3, 15] ]
depth_ranges = [ [3, 684], [30, 684], [14, 154], [3, 70], [5, 17] ]
alpha = 0.05

df_factoring = pd.read_csv( filename_factoring, delimiter = ", ", engine='python')
df_random    = pd.read_csv( filename_random, delimiter = ", ", engine='python')

df_factoring['run time'] = df_factoring['run time'].apply(lambda x: 1000*x )
df_random['run time']    = df_random['run time'].apply(lambda x: 1000*x )


df_table_total = pd.DataFrame()

df_table = pd.DataFrame(index = ["\\textbf{Category}", "\\textbf{\\#Qubits}", "\\textbf{Depth}", "\\textbf{Factoring [ms]}", "\\textbf{Random [ms]}", "\\textbf{\\%(NN)}", "\\textbf{\\%(S)}", "\\textbf{\\%(M)}", "\\textbf{\\%(L)}", "\\textbf{\\% fail or (N)}"], columns = range(len(program_categories)))

filename_latex_table = os.path.join( "tables", "runtime", "Summary", "latex_summary_table_runtime.txt" )

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
        number_of_effects_less_50 = 0
        number_of_Ls = 0
        number_of_Ms = 0
        number_of_Ss = 0


        # fetch average runtime rate
        runtime_rate_average_factoring = df_curr_factoring["run time"].mean()
        runtime_rate_average_random    = df_curr_random["run time"].mean()

        # fetch stds
        runtime_rate_std_factoring = df_curr_factoring["run time"].std()
        runtime_rate_std_random    = df_curr_random["run time"].std()


        # fetch #T
        number_of_transformations = int( df_curr_factoring["number_of_transformations"].max() )


        runtime_distribution_factoring = np.array(df_curr_factoring["run time"])
        runtime_distribution_random = np.array(df_curr_random["run time"])


        # Perform the Mann-Whitney U test
        u_statistic, p_value = stats.mannwhitneyu(runtime_distribution_factoring, runtime_distribution_random)


        # Combine the samples
        combined = np.concatenate((runtime_distribution_factoring, runtime_distribution_random))

        # Rank the combined array
        ranks = stats.rankdata(combined)

        # Sum the ranks for the first sample
        R1 = np.sum( ranks[ :len( runtime_distribution_factoring ) ] )

        # fetch Vargha_Delaney_A effect size
        VA_A = Vargha_Delaney_A( R1, len(runtime_distribution_factoring), len(runtime_distribution_random) )

        # fetch effect size category
        VA_category = check_VA_scaled(VA_A)


        # see if test passed (1) or failed (0)
        if p_value <= alpha and VA_category != "(N)":
            bin_category = "(NN)"

            test_result = f"p-val $\leq$ {alpha}, {bin_category}"
            number_of_passes += 1

            if VA_A <= 0.5:
                number_of_effects_less_50 += 1

            if VA_category == "(S)":
                number_of_Ss += 1

            if VA_category == "(M)":
                number_of_Ms += 1

            if VA_category == "(L)":
                number_of_Ls += 1

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


        # assign values to table dataframe
        df_table.at["\\textbf{\\#Qubits}", program_category_index] = f"{curr_qubit_range}"

        # assign depth
        df_table.at["\\textbf{Depth}", program_category_index] = f"{curr_depth_range}"

        # assign program category
        df_table.at["\\textbf{Category}", program_category_index] = f"{curr_program_category}"

        # assign #T and f(#Q)
        #df_table.at["\\textbf{\\#T}", program_category_index] = f"{number_of_transformations}"
        #df_table.at["\\textbf{f(\\#Q)}", program_category_index] = f"{f_complexity( int( number_of_qubits) )}"


        # assign average and std
        df_table.at["\\textbf{Factoring [ms]}", program_category_index] = f"{runtime_rate_average_factoring:.1f} $\\pm$ {runtime_rate_std_factoring:.1f}"
        df_table.at["\\textbf{Random [ms]}", program_category_index] = f"{runtime_rate_average_random:.1f} $\\pm$ {runtime_rate_std_random:.1f}"



        df_table_transposed = df_table.T



        print("------------------")
        print("| Category =", program_categories[program_category_index], "|")
        print("------------------")



    df_latex = df_table_transposed.to_latex(index=False, escape=False)

    outfile.write(df_latex + "\n")
