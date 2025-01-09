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


def perform_mann_whitney_and_get_effect_size(df1, df2, pval_tag, program_category_index):

    # Perform the Kruskal-Wallis H test
    statistic, p_value = stats.mannwhitneyu( df1, df2 )

    # Combine the samples
    combined = np.concatenate((df1, df2))

    # Rank the combined array
    ranks = stats.rankdata(combined)

    # Sum the ranks for the first sample
    R1 = np.sum( ranks[ :len( df1 ) ] )

    # fetch Vargha_Delaney_A effect size
    VA_A = Vargha_Delaney_A( R1, len(df1), len(df2) )

    # fetch effect size category
    VA_category = check_VA_scaled(VA_A)

    pass_or_fail_variable = "pass"

    if p_value <= alpha and VA_category != "(N)":



        pass_or_fail_variable = "fail"

        return pass_or_fail_variable

    else:


        return pass_or_fail_variable


def perform_kruskal_wallis(df1, df2, df3, program_category_index, pass_or_fail):
    # Perform the Kruskal-Wallis H test
    statistic, p_value = stats.kruskal( df1, df2, df3 )

    if pass_or_fail == "fail":

        if p_value <= 0.001:
            rounded_pval = "{:.1e}".format(p_value)
            df_table.at["\\textbf{p-value} DFR", program_category_index] = b_add + f"{rounded_pval}" + b_end

        else:
            df_table.at["\\textbf{p-value} DFR", program_category_index] = b_add + f"{np.round(p_value, 5)}" + b_end

    else:

        if p_value <= 0.001:
            rounded_pval = "{:.1e}".format(p_value)
            df_table.at["\\textbf{p-value} DFR", program_category_index] = f"{rounded_pval}"

        else:
            df_table.at["\\textbf{p-value} DFR", program_category_index] = f"{np.round(p_value, 5)}"


b_add = "\\textbf{"
b_end = "}"

b_add = ""
b_end = ""


alpha = 0.05

current_directory = os.getcwd()
grandparent_directory = os.path.dirname(os.path.dirname(current_directory))

full_or_summary = "Summary"

program_categories = ["all", "aamp", "qwalk", "var", "gs"]

qubit_ranges = [ [2, 15], [6, 9], [3, 5], [2, 8], [3, 15] ]
depth_ranges = [ [3, 684], [30, 684], [14, 154], [3, 70], [5, 17] ]

filename_csv_passes = os.path.join( "tables", full_or_summary, "runtime_pass_counts.csv" )
df_pass_counts = pd.read_csv( filename_csv_passes, index_col = 0 )

filename_csv_passes_r = os.path.join( "tables", full_or_summary, "runtime_pass_counts_r.csv" )
df_pass_counts_r = pd.read_csv( filename_csv_passes_r, index_col = 0 )


filename_all_data = os.path.join( grandparent_directory, "results", r"results_run_all.txt" )

df_all_data = pd.read_csv( filename_all_data )

df_default   = df_all_data[ (df_all_data["Approach"] == "Greedy") & (df_all_data[" Type"] == True) ]
df_factoring = df_all_data[ (df_all_data["Approach"] == "Greedy") & (df_all_data[" Type"] == False) ]
df_random    = df_all_data[ (df_all_data["Approach"] == "Random") & (df_all_data[" Type"] == False) ]


filename_latex_table = os.path.join( "tables", full_or_summary, f"latex_summary_table_runtime.txt" )

df_table = pd.DataFrame(index = ["\\textbf{Category}", "\\textbf{\\#Qubits}", "\\textbf{Depth}", "\\textbf{Default [s]}", "\\textbf{Factoring [s]}", "\\textbf{Random [s]}", "\\textbf{\\%passed (NN) DF}", "\\textbf{\\%equal DF}", "\\textbf{\\%passed (NN) DR}","\\textbf{\\%equal DR}", "\\textbf{\\%passed (NN) FR}", "\\textbf{\\%equal FR}"], columns = range(len(program_categories)))


with open( filename_latex_table , "w") as outfile:


    for program_category_index in range( len(program_categories) ):

        curr_program_category = program_categories[ program_category_index ]

        curr_qubit_range = qubit_ranges[ program_category_index ]
        curr_depth_range = depth_ranges[ program_category_index ]

        if curr_program_category == "all":

            df_curr_default   = df_default
            df_curr_factoring = df_factoring
            df_curr_random    = df_random

        else:

            df_curr_default   = df_default[ df_default["category"] == curr_program_category ]
            df_curr_factoring = df_factoring[ df_factoring["category"] == curr_program_category ]
            df_curr_random    = df_random[ df_random["category"] == curr_program_category ]

        unique_progs = list(df_curr_default['Program var'].unique())

        prog_to_exclude1 = "aamp_q6_6"
        prog_to_exclude2 = "aamp_q7_5"

        if prog_to_exclude1 in unique_progs:
            unique_progs.remove(prog_to_exclude1)

        if prog_to_exclude2 in unique_progs:
            unique_progs.remove(prog_to_exclude2)


        # fetch number of qubits
        number_of_qubits = curr_qubit_range

        # fetch average runtime rate
        runtime_rate_average_default   = df_curr_default["  T_tot[s]"].mean()
        runtime_rate_average_factoring = df_curr_factoring["  T_tot[s]"].mean()
        runtime_rate_average_random    = df_curr_random["  T_tot[s]"].mean()

        # fetch stds
        runtime_rate_std_default   = df_curr_default["  T_tot[s]"].std()
        runtime_rate_std_factoring = df_curr_factoring["  T_tot[s]"].std()
        runtime_rate_std_random    = df_curr_random["  T_tot[s]"].std()


        # fetch depth
        depth = curr_depth_range

        runtime_distribution_default   = np.array(df_curr_default["  T_tot[s]"])
        runtime_distribution_factoring = np.array(df_curr_factoring["  T_tot[s]"])
        runtime_distribution_random    = np.array(df_curr_random["  T_tot[s]"])


        # between Default-Factoring
        DF_pass_or_fail = perform_mann_whitney_and_get_effect_size(runtime_distribution_default, runtime_distribution_factoring, "DF", program_category_index)

        # between Default-Random
        DR_pass_or_fail = perform_mann_whitney_and_get_effect_size(runtime_distribution_default, runtime_distribution_random, "DR", program_category_index)

        # between Factoring-Random
        FR_pass_or_fail = perform_mann_whitney_and_get_effect_size(runtime_distribution_factoring, runtime_distribution_random, "FR", program_category_index)


        if DF_pass_or_fail == "fail" or DR_pass_or_fail == "fail" or FR_pass_or_fail == "fail":

            #perform_kruskal_wallis(runtime_distribution_default, runtime_distribution_factoring, runtime_distribution_random, program_category_index, "fail")

            # assign values to table dataframe
            df_table.at["\\textbf{\\#Qubits}", program_category_index] = b_add + f"{number_of_qubits}" + b_end

            # assign depth
            df_table.at["\\textbf{Depth}", program_category_index] = b_add + f"{depth}" + b_end

            # assign program category
            df_table.at["\\textbf{Category}", program_category_index] = b_add + f"{curr_program_category}" + b_end

            # assign average and std
            df_table.at["\\textbf{Default [s]}", program_category_index] = b_add + f"{runtime_rate_average_default:.1f} $\\pm$ {runtime_rate_std_default:.1f}" + b_end
            df_table.at["\\textbf{Factoring [s]}", program_category_index] = b_add + f"{runtime_rate_average_factoring:.1f} $\\pm$ {runtime_rate_std_factoring:.1f}" + b_end
            df_table.at["\\textbf{Random [s]}", program_category_index] = b_add + f"{runtime_rate_average_random:.1f} $\\pm$ {runtime_rate_std_random:.1f}" + b_end

            passed_DF   = df_pass_counts.loc[curr_program_category, "\%passed (NN) DF"]
            passed_DF_r = df_pass_counts_r.loc[curr_program_category, "\%passed (NN) DF"]

            passed_DR = df_pass_counts.loc[curr_program_category, "\%passed (NN) DR"]
            passed_DR_r = df_pass_counts_r.loc[curr_program_category, "\%passed (NN) DR"]

            passed_FR = df_pass_counts.loc[curr_program_category, "\%passed (NN) FR"]
            passed_FR_r = df_pass_counts_r.loc[curr_program_category, "\%passed (NN) FR"]

            df_table.at["\\textbf{\\%passed (NN) DF}", program_category_index] = f"{passed_DF}/{passed_DF_r}"
            df_table.at["\\textbf{\\%passed (NN) DR}", program_category_index] = f"{passed_DR}/{passed_DR_r}"
            df_table.at["\\textbf{\\%passed (NN) FR}", program_category_index] = f"{passed_FR}/{passed_FR_r}"

            df_table.at["\\textbf{\\%equal DF}", program_category_index] = df_pass_counts.loc[curr_program_category, "\%equal DF"]
            df_table.at["\\textbf{\\%equal DR}", program_category_index] = df_pass_counts.loc[curr_program_category, "\%equal DR"]
            df_table.at["\\textbf{\\%equal FR}", program_category_index] = df_pass_counts.loc[curr_program_category, "\%equal FR"]


        else:

            #perform_kruskal_wallis(runtime_distribution_default, runtime_distribution_factoring, runtime_distribution_random, program_category_index, "pass")

            # assign values to table dataframe
            df_table.at["\\textbf{\\#Qubits}", program_category_index] = f"{number_of_qubits}"

            # assign depth
            df_table.at["\\textbf{Depth}", program_category_index] = f"{depth}"

            # assign program category
            df_table.at["\\textbf{Category}", program_category_index] = f"{curr_program_category}"


            # assign average and std
            df_table.at["\\textbf{Default [s]}", program_category_index] = f"{runtime_rate_average_default:.1f} $\\pm$ {runtime_rate_std_default:.1f}"
            df_table.at["\\textbf{Factoring [s]}", program_category_index] = f"{runtime_rate_average_factoring:.1f} $\\pm$ {runtime_rate_std_factoring:.1f}"
            df_table.at["\\textbf{Random [s]}", program_category_index] = f"{runtime_rate_average_random:.1f} $\\pm$ {runtime_rate_std_random:.1f}"

            passed_DF   = df_pass_counts.loc[curr_program_category, "\%passed (NN) DF"]
            passed_DF_r = df_pass_counts_r.loc[curr_program_category, "\%passed (NN) DF"]

            passed_DR = df_pass_counts.loc[curr_program_category, "\%passed (NN) DR"]
            passed_DR_r = df_pass_counts_r.loc[curr_program_category, "\%passed (NN) DR"]

            passed_FR = df_pass_counts.loc[curr_program_category, "\%passed (NN) FR"]
            passed_FR_r = df_pass_counts_r.loc[curr_program_category, "\%passed (NN) FR"]

            df_table.at["\\textbf{\\%passed (NN) DF}", program_category_index] = f"{passed_DF}/{passed_DF_r}"
            df_table.at["\\textbf{\\%passed (NN) DR}", program_category_index] = f"{passed_DR}/{passed_DR_r}"
            df_table.at["\\textbf{\\%passed (NN) FR}", program_category_index] = f"{passed_FR}/{passed_FR_r}"

            df_table.at["\\textbf{\\%equal DF}", program_category_index] = df_pass_counts.loc[curr_program_category, "\%equal DF"]
            df_table.at["\\textbf{\\%equal DR}", program_category_index] = df_pass_counts.loc[curr_program_category, "\%equal DR"]
            df_table.at["\\textbf{\\%equal FR}", program_category_index] = df_pass_counts.loc[curr_program_category, "\%equal FR"]


    df_table_transposed = df_table.T

    df_latex = df_table_transposed.to_latex(index=False, escape=False)

    outfile.write(df_latex + "\n")
