import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import seaborn as sns

# ----------
# | Inputs |
# ---------
number_of_repetitions = 30


def perform_mann_whitney_and_get_effect_size(df1, df2):

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

        # assign test result
        magnitude_result = f"{VA_category}"

        # assign Vargha_Delaney_A effect size
        A_12_effect_size = f"{np.round(VA_A, 3)}"

        if p_value <= 0.001:
            rounded_pval = "{:.1e}".format(p_value)
            pval_result = f"{rounded_pval}"

        else:
            pval_result = f"{np.round(p_value, 5)}"

        pass_or_fail_variable = "fail"

        return magnitude_result, pval_result, A_12_effect_size, pass_or_fail_variable

    else:

        # assign test result
        magnitude_result = f"{VA_category}"

        # assign Vargha_Delaney_A effect size
        A_12_effect_size = f"{np.round(VA_A, 3)}"

        if p_value <= 0.001:
            rounded_pval = "{:.1e}".format(p_value)
            pval_result = f"{rounded_pval}"

        else:
            pval_result = f"{np.round(p_value, 5)}"

        return magnitude_result, pval_result, A_12_effect_size, pass_or_fail_variable

def plot_density(df1, df2, column_name, label1='Default', label2='Factoring'):
    """
    Plots density plots for the specified column from two different DataFrames.

    Parameters:
    - df1: Pandas DataFrame containing the first dataset.
    - df2: Pandas DataFrame containing the second dataset.
    - column_name: The name of the column for which the density plot is to be generated.
    - label1: Label for the first dataset (for the legend).
    - label2: Label for the second dataset (for the legend).
    """
    # Set the style of the visualization
    sns.set(style="whitegrid")

    # Create a KDE plot for the specified column of the first DataFrame
    sns.kdeplot(data=df1[column_name], label=label1, shade=True)

    # Create a KDE plot for the specified column of the second DataFrame
    sns.kdeplot(data=df2[column_name], label=label2, shade=True)

    # Add some labels and a title for clarity
    plt.title('Density Plot of ' + column_name)
    plt.xlabel(column_name)
    plt.ylabel('Density')

    # Add a legend to distinguish between the plots
    plt.legend()

    # Save the figure
    save_filename = os.path.join("figures", "DF_all")
    plt.savefig(save_filename)

    # Clear the current figure to prevent overlap with future plots
    plt.clf()

def perform_kruskal_wallis(df1, df2, df3):

    combined_dfs = np.concatenate( [df1, df2, df3] )
    if not np.all( combined_dfs == combined_dfs[0] ):

        # Perform the Kruskal-Wallis H test
        statistic, p_value = stats.kruskal( df1, df2, df3 )
        print(p_value,"test")
        if p_value <= 0.001:
            rounded_pval = "{:.1e}".format(p_value)
            return f"{rounded_pval}"
        else:
            return f"{np.round(p_value, 5)}"
    else:
        return "id"

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


def compute_mutation_score_distribution( dataframe, unique_prog_list, number_of_reps, category  ):

    # Specify the columns from 'dataframe' that you want to include in 'df_out'
    included_columns = ['Qubits', ' depth', 'Program var', "Run Num", "category", "Approach"]

    # Verify the included columns exist in the dataframe to avoid KeyErrors
    included_columns = [col for col in included_columns if col in dataframe.columns]

    # Define new columns that will be added to 'df_out'
    new_columns = included_columns + ["ms all", "ms X", "ms Z", "ms Ry"]

    # Initialize 'df_out' with the selected columns from 'dataframe' and the new columns
    # For the existing columns, copy the structure and for the new columns initialize with NaN
    df_out = pd.DataFrame(columns=new_columns)

    current_approach = dataframe["Approach"][dataframe.index[0]]

    for prog_index in range( len(unique_prog_list) ):

        curr_program = unique_prog_list[prog_index]
        df_curr_program = dataframe[ dataframe["Program var"] == curr_program ]

        curr_program_number_of_qubits = df_curr_program["Qubits"][df_curr_program.index[0]]
        curr_program_depth = df_curr_program[" depth"][df_curr_program.index[0]]

        for run_index in range( number_of_reps ):

            df_curr_program_curr_run_all = df_curr_program[ df_curr_program[ "Run Num" ] == run_index ]


            df_curr_program_curr_run_X  = df_curr_program_curr_run_all[ (df_curr_program_curr_run_all[" Mutation"] == "1|0") | \
                                                                        (df_curr_program_curr_run_all[" Mutation"] == "1|1") | \
                                                                        (df_curr_program_curr_run_all[" Mutation"] == "1|2")]

            df_curr_program_curr_run_Z  = df_curr_program_curr_run_all[ (df_curr_program_curr_run_all[" Mutation"] == "2|0") | \
                                                                        (df_curr_program_curr_run_all[" Mutation"] == "2|1") | \
                                                                        (df_curr_program_curr_run_all[" Mutation"] == "2|2")]

            df_curr_program_curr_run_Ry  = df_curr_program_curr_run_all[ (df_curr_program_curr_run_all[" Mutation"] == "3|1") | \
                                                                        (df_curr_program_curr_run_all[" Mutation"] == "3|2") | \
                                                                        (df_curr_program_curr_run_all[" Mutation"] == "3|3") | \
                                                                        (df_curr_program_curr_run_all[" Mutation"] == "3|4") | \
                                                                        (df_curr_program_curr_run_all[" Mutation"] == "3|5") | \
                                                                        (df_curr_program_curr_run_all[" Mutation"] == "3|6") | \
                                                                        (df_curr_program_curr_run_all[" Mutation"] == "3|7") | \
                                                                        (df_curr_program_curr_run_all[" Mutation"] == "3|8") | \
                                                                        (df_curr_program_curr_run_all[" Mutation"] == "3|9") ]

            ms_all = len(df_curr_program_curr_run_all[ (df_curr_program_curr_run_all[" Outcome"] == "pdo") | (df_curr_program_curr_run_all[" Outcome"] == "woo") ] )/len( df_curr_program_curr_run_all )*100

            ms_X  = len(df_curr_program_curr_run_X[ (df_curr_program_curr_run_X[" Outcome"] == "pdo") | (df_curr_program_curr_run_X[" Outcome"] == "woo") ] )/len( df_curr_program_curr_run_X )*100
            ms_Z  = len(df_curr_program_curr_run_Z[ (df_curr_program_curr_run_Z[" Outcome"] == "pdo") | (df_curr_program_curr_run_Z[" Outcome"] == "woo") ] )/len( df_curr_program_curr_run_Z )*100
            ms_Ry = len(df_curr_program_curr_run_Ry[ (df_curr_program_curr_run_Ry[" Outcome"] == "pdo") | (df_curr_program_curr_run_Ry[" Outcome"] == "woo") ] )/len( df_curr_program_curr_run_Ry )*100

            new_row = {
            'Qubits': curr_program_number_of_qubits,
            " depth": curr_program_depth,
            "Program var": curr_program,
            "Run Num": run_index,
            "category": category,
            "Approach": current_approach,
            "ms all": ms_all,
            "ms X": ms_X,
            "ms Z": ms_Z,
            "ms Ry": ms_Ry
            }

            df_out = df_out.append(new_row, ignore_index=True)

    return df_out


b_add = "\\textbf{"
b_end = "}"

b_add = ""
b_end = ""

df_pass_counts = pd.DataFrame( index = ["all", "aamp", "qwalk", "var", "gs"], columns = ["\\%passed (NN) DF", "\\%passed (NN) DR", "\\%passed (NN) FR", "\\%passed (NN) Counts DF", "\\%passed (NN) Counts DR", "\\%passed (NN) Counts FR", "Total Counts"] )
df_pass_counts.index.name = "category"

alpha = 0.05
round_to = 1

current_directory = os.getcwd()
grandparent_directory = os.path.dirname(os.path.dirname(current_directory))

full_or_summary = "Summary"

program_categories = ["all", "aamp", "qwalk", "var", "gs"]
qubit_ranges = [ [2, 15], [6, 9], [3, 5], [2, 8], [3, 15] ]
depth_ranges = [ [3, 684], [30, 684], [14, 154], [3, 70], [5, 17] ]

approaches_list = ["default", "factoring", "random"]
approaches_list_uppercase = ["Default", "Factoring", "Random"]

filename_all_data = os.path.join( grandparent_directory, "results", r"results_run_all.txt" )

df_all_data = pd.read_csv( filename_all_data )

df_default   = df_all_data[ (df_all_data["Approach"] == "Greedy") & (df_all_data[" Type"] == True) ]
df_default["Approach"] = "Default"

df_factoring = df_all_data[ (df_all_data["Approach"] == "Greedy") & (df_all_data[" Type"] == False) ]
df_random    = df_all_data[ (df_all_data["Approach"] == "Random") & (df_all_data[" Type"] == False) ]


filename_latex_table = os.path.join( "tables", full_or_summary, f"latex_summary_table_mutation_score.txt" )

with open( filename_latex_table , "w") as outfile:

    df_table = pd.DataFrame(columns = ["\\textbf{Category}", "Mutant Type", "\\textbf{\\#Qubits}", "\\textbf{Depth}", "\\textbf{Default [\\%]}", "\\textbf{Greedy [\\%]}", "\\textbf{Random [\\%]}", "\\textbf{p-value} DFR", "\\textbf{p-value} DF", "$\\mathbf{\hat{A}}_{12}$ DF", "\\textbf{Magnitude} DF", "\\textbf{p-value} DR", "$\\mathbf{\hat{A}}_{12}$ DR", "\\textbf{Magnitude} DR", "\\textbf{p-value} FR", "$\\mathbf{\hat{A}}_{12}$ FR", "\\textbf{Magnitude} FR"] )

    for program_category_index in range( len(program_categories) ):

        curr_program_category = program_categories[ program_category_index ]
        curr_qubit_range = qubit_ranges[program_category_index]
        curr_depth_range = depth_ranges[program_category_index]

        if curr_program_category != "all":
            df_curr_default   = df_default[ df_default["category"] == curr_program_category ]
            df_curr_factoring = df_factoring[ df_factoring["category"] == curr_program_category ]
            df_curr_random    = df_random[ df_random["category"] == curr_program_category ]

        else:
            df_curr_default   = df_default
            df_curr_factoring = df_factoring
            df_curr_random    = df_random


        unique_progs = list(df_curr_default['Program var'].unique())

        prog_to_exclude1 = "aamp_q6_6"
        prog_to_exclude2 = "aamp_q7_5"

        number_fail_DF_all = 0
        number_fail_DF_X   = 0
        number_fail_DF_Z   = 0
        number_fail_DF_Ry  = 0

        number_fail_DR_all = 0
        number_fail_DR_X   = 0
        number_fail_DR_Z   = 0
        number_fail_DR_Ry  = 0

        number_fail_FR_all = 0
        number_fail_FR_X = 0
        number_fail_FR_Z = 0
        number_fail_FR_Ry = 0


        if prog_to_exclude1 in unique_progs:
            unique_progs.remove(prog_to_exclude1)

        if prog_to_exclude2 in unique_progs:
            unique_progs.remove(prog_to_exclude2)


        df_mutation_score_distribution_default   = compute_mutation_score_distribution(df_curr_default,   unique_progs, number_of_repetitions, curr_program_category)
        df_mutation_score_distribution_factoring = compute_mutation_score_distribution(df_curr_factoring, unique_progs, number_of_repetitions, curr_program_category)
        df_mutation_score_distribution_random    = compute_mutation_score_distribution(df_curr_random,    unique_progs, number_of_repetitions, curr_program_category)

        # write mutation score data for use in correlation table
        if curr_program_category != "all":
            df_mutation_score_distribution_factoring.to_csv( os.path.join("msDistributionsForCorrelation", f"mutation_score_distribution_{curr_program_category}_greedy.csv"), index=False)
            df_mutation_score_distribution_random.to_csv( os.path.join("msDistributionsForCorrelation", f"mutation_score_distribution_{curr_program_category}_random.csv"), index=False)
        else:
            pass

        if curr_program_category == "all":
            plot_density(df_mutation_score_distribution_default, df_mutation_score_distribution_factoring, "ms all")


        # fetch current program
        df_default_current_program   = df_mutation_score_distribution_default
        df_factoring_current_program = df_mutation_score_distribution_factoring
        df_random_current_program    = df_mutation_score_distribution_random

        # fetch number of qubits
        number_of_qubits = curr_qubit_range

        # fetch depth
        depth = curr_depth_range

        # -------------------------------- all
        # fetch average mutation score all
        ms_average_default   = round( df_default_current_program["ms all"].mean(), 1)
        ms_average_factoring = round( df_factoring_current_program["ms all"].mean(), 1)
        ms_average_random    = round( df_random_current_program["ms all"].mean(), 1)

        # fetch stds
        ms_std_default   = round( df_default_current_program["ms all"].std(), 1)
        ms_std_factoring = round( df_factoring_current_program["ms all"].std(), 1)
        ms_std_random    = round( df_random_current_program["ms all"].std(), 1)
        # ------------------------------- all

        # -------------------------------- X
        # fetch average mutation score X
        ms_average_default_X   = round( df_default_current_program["ms X"].mean(), 1)
        ms_average_factoring_X = round( df_factoring_current_program["ms X"].mean(), 1)
        ms_average_random_X    = round( df_random_current_program["ms X"].mean(), 1)

        # fetch stds
        ms_std_default_X   = round( df_default_current_program["ms X"].std(), 1)
        ms_std_factoring_X = round( df_factoring_current_program["ms X"].std(), 1)
        ms_std_random_X    = round( df_random_current_program["ms X"].std(), 1)
        # ------------------------------- X

        # -------------------------------- Z
        # fetch average mutation score Z
        ms_average_default_Z   = round( df_default_current_program["ms Z"].mean(), 1)
        ms_average_factoring_Z = round( df_factoring_current_program["ms Z"].mean(), 1)
        ms_average_random_Z    = round( df_random_current_program["ms Z"].mean(), 1)

        # fetch stds
        ms_std_default_Z   = round( df_default_current_program["ms Z"].std(), 1)
        ms_std_factoring_Z = round( df_factoring_current_program["ms Z"].std(), 1)
        ms_std_random_Z    = round( df_random_current_program["ms Z"].std(), 1)
        # ------------------------------- Z

        # -------------------------------- Ry
        # fetch average mutation score Ry
        ms_average_default_Ry   = round( df_default_current_program["ms Ry"].mean(), 1)
        ms_average_factoring_Ry = round( df_factoring_current_program["ms Ry"].mean(), 1)
        ms_average_random_Ry    = round( df_random_current_program["ms Ry"].mean(), 1)

        # fetch stds
        ms_std_default_Ry   = round( df_default_current_program["ms Ry"].std(), 1)
        ms_std_factoring_Ry = round( df_factoring_current_program["ms Ry"].std(), 1)
        ms_std_random_Ry    = round( df_random_current_program["ms Ry"].std(), 1)
        # ------------------------------- Ry


        pval_kruskal_all = perform_kruskal_wallis( df_default_current_program["ms all"], df_factoring_current_program["ms all"], df_random_current_program["ms all"] )
        pval_kruskal_X   = perform_kruskal_wallis( df_default_current_program["ms X"], df_factoring_current_program["ms X"], df_random_current_program["ms X"] )
        pval_kruskal_Z   = perform_kruskal_wallis( df_default_current_program["ms Z"], df_factoring_current_program["ms Z"], df_random_current_program["ms Z"] )
        pval_kruskal_Ry  = perform_kruskal_wallis( df_default_current_program["ms Ry"], df_factoring_current_program["ms Ry"], df_random_current_program["ms Ry"] )

        # between Default-Factoring
        magnitude_DF_all, pval_DF_all, A12_DF_all, pass_or_fail_DF_all = perform_mann_whitney_and_get_effect_size(df_default_current_program["ms all"], df_factoring_current_program["ms all"])
        magnitude_DF_X, pval_DF_X, A12_DF_X, pass_or_fail_DF_X         = perform_mann_whitney_and_get_effect_size(df_default_current_program["ms X"], df_factoring_current_program["ms X"])
        magnitude_DF_Z, pval_DF_Z, A12_DF_Z, pass_or_fail_DF_Z         = perform_mann_whitney_and_get_effect_size(df_default_current_program["ms Z"], df_factoring_current_program["ms Z"])
        magnitude_DF_Ry, pval_DF_Ry, A12_DF_Ry, pass_or_fail_DF_Ry     = perform_mann_whitney_and_get_effect_size(df_default_current_program["ms Ry"], df_factoring_current_program["ms Ry"])

        if pass_or_fail_DF_all == "fail":
            number_fail_DF_all += 1

        if pass_or_fail_DF_X == "fail":
            number_fail_DF_X += 1

        if pass_or_fail_DF_Z == "fail":
            number_fail_DF_Z += 1

        if pass_or_fail_DF_Ry == "fail":
            number_fail_DF_Ry += 1

        # between Default-Random
        magnitude_DR_all, pval_DR_all, A12_DR_all, pass_or_fail_DR_all = perform_mann_whitney_and_get_effect_size(df_default_current_program["ms all"], df_random_current_program["ms all"])
        magnitude_DR_X, pval_DR_X, A12_DR_X, pass_or_fail_DR_X       = perform_mann_whitney_and_get_effect_size(df_default_current_program["ms X"], df_random_current_program["ms X"])
        magnitude_DR_Z, pval_DR_Z, A12_DR_Z, pass_or_fail_DR_Z       = perform_mann_whitney_and_get_effect_size(df_default_current_program["ms Z"], df_random_current_program["ms Z"])
        magnitude_DR_Ry, pval_DR_Ry, A12_DR_Ry, pass_or_fail_DR_Ry    = perform_mann_whitney_and_get_effect_size(df_default_current_program["ms Ry"], df_random_current_program["ms Ry"])

        if pass_or_fail_DR_all == "fail":
            number_fail_DR_all += 1

        if pass_or_fail_DR_X == "fail":
            number_fail_DR_X += 1

        if pass_or_fail_DR_Z == "fail":
            number_fail_DR_Z += 1

        if pass_or_fail_DR_Ry == "fail":
            number_fail_DR_Ry += 1

        # between Factoring-Random
        magnitude_FR_all, pval_FR_all, A12_FR_all, pass_or_fail_FR_all = perform_mann_whitney_and_get_effect_size(df_factoring_current_program["ms all"], df_random_current_program["ms all"])
        magnitude_FR_X, pval_FR_X, A12_FR_X, pass_or_fail_FR_X       = perform_mann_whitney_and_get_effect_size(df_factoring_current_program["ms X"], df_random_current_program["ms X"])
        magnitude_FR_Z, pval_FR_Z, A12_FR_Z, pass_or_fail_FR_Z       = perform_mann_whitney_and_get_effect_size(df_factoring_current_program["ms Z"], df_random_current_program["ms Z"])
        magnitude_FR_Ry, pval_FR_Ry, A12_FR_Ry, pass_or_fail_FR_Ry    = perform_mann_whitney_and_get_effect_size(df_factoring_current_program["ms Ry"], df_random_current_program["ms Ry"])

        if pass_or_fail_FR_all == "fail":
            number_fail_FR_all += 1

        if pass_or_fail_FR_X == "fail":
            number_fail_FR_X += 1

        if pass_or_fail_FR_Z == "fail":
            number_fail_FR_Z += 1

        if pass_or_fail_FR_Ry == "fail":
            number_fail_FR_Ry += 1

        # add to df_table
        new_row_all = {
        "Mutant Type": "all",
        "\\textbf{Category}": curr_program_category,
        "\\textbf{\\#Qubits}": number_of_qubits,
        "\\textbf{Depth}": depth,
        "\\textbf{Default [\\%]}": f"{ms_average_default} $\\pm$ {ms_std_default}",
        "\\textbf{Greedy [\\%]}": f"{ms_average_factoring} $\\pm$ {ms_std_factoring}",
        "\\textbf{Random [\\%]}": f"{ms_average_random} $\\pm$ {ms_std_random}",
        "\\textbf{p-value} DFR": pval_kruskal_all,
        "\\textbf{p-value} DF": pval_DF_all,
        "$\\mathbf{\hat{A}}_{12}$ DF": A12_DF_all,
        "\\textbf{Magnitude} DF": magnitude_DF_all,
        "\\textbf{p-value} DR": pval_DR_all,
        "$\\mathbf{\hat{A}}_{12}$ DR": A12_DR_all,
        "\\textbf{Magnitude} DR": magnitude_DR_all,
        "\\textbf{p-value} FR": pval_FR_all,
        "$\\mathbf{\hat{A}}_{12}$ FR": A12_FR_all,
        "\\textbf{Magnitude} FR": magnitude_FR_all
        }


        # add to df_table
        new_row_X = {
        "Mutant Type": "X",
        "\\textbf{Category}": curr_program_category,
        "\\textbf{\\#Qubits}": number_of_qubits,
        "\\textbf{Depth}": depth,
        "\\textbf{Default [\\%]}": f"{ms_average_default_X} $\\pm$ {ms_std_default_X}",
        "\\textbf{Greedy [\\%]}": f"{ms_average_factoring_X} $\\pm$ {ms_std_factoring_X}",
        "\\textbf{Random [\\%]}": f"{ms_average_random_X} $\\pm$ {ms_std_random_X}",
        "\\textbf{p-value} DFR": pval_kruskal_X,
        "\\textbf{p-value} DF": pval_DF_X,
        "$\\mathbf{\hat{A}}_{12}$ DF": A12_DF_X,
        "\\textbf{Magnitude} DF": magnitude_DF_X,
        "\\textbf{p-value} DR": pval_DR_X,
        "$\\mathbf{\hat{A}}_{12}$ DR": A12_DR_X,
        "\\textbf{Magnitude} DR": magnitude_DR_X,
        "\\textbf{p-value} FR": pval_FR_X,
        "$\\mathbf{\hat{A}}_{12}$ FR": A12_FR_X,
        "\\textbf{Magnitude} FR": magnitude_FR_X
        }

        # add to df_table
        new_row_Z = {
        "Mutant Type": "Z",
        "\\textbf{Category}": curr_program_category,
        "\\textbf{\\#Qubits}": number_of_qubits,
        "\\textbf{Depth}": depth,
        "\\textbf{Default [\\%]}": f"{ms_average_default_Z} $\\pm$ {ms_std_default_Z}",
        "\\textbf{Greedy [\\%]}": f"{ms_average_factoring_Z} $\\pm$ {ms_std_factoring_Z}",
        "\\textbf{Random [\\%]}": f"{ms_average_random_Z} $\\pm$ {ms_std_random_Z}",
        "\\textbf{p-value} DFR": pval_kruskal_Z,
        "\\textbf{p-value} DF": pval_DF_Z,
        "$\\mathbf{\hat{A}}_{12}$ DF": A12_DF_Z,
        "\\textbf{Magnitude} DF": magnitude_DF_Z,
        "\\textbf{p-value} DR": pval_DR_Z,
        "$\\mathbf{\hat{A}}_{12}$ DR": A12_DR_Z,
        "\\textbf{Magnitude} DR": magnitude_DR_Z,
        "\\textbf{p-value} FR": pval_FR_Z,
        "$\\mathbf{\hat{A}}_{12}$ FR": A12_FR_Z,
        "\\textbf{Magnitude} FR": magnitude_FR_Z
        }

        # add to df_table
        new_row_Ry = {
        "Mutant Type": "$R_y$",
        "\\textbf{Category}": curr_program_category,
        "\\textbf{\\#Qubits}": number_of_qubits,
        "\\textbf{Depth}": depth,
        "\\textbf{Default [\\%]}": f"{ms_average_default_Ry} $\\pm$ {ms_std_default_Ry}",
        "\\textbf{Greedy [\\%]}": f"{ms_average_factoring_Ry} $\\pm$ {ms_std_factoring_Ry}",
        "\\textbf{Random [\\%]}": f"{ms_average_random_Ry} $\\pm$ {ms_std_random_Ry}",
        "\\textbf{p-value} DFR": pval_kruskal_Ry,
        "\\textbf{p-value} DF": pval_DF_Ry,
        "$\\mathbf{\hat{A}}_{12}$ DF": A12_DF_Ry,
        "\\textbf{Magnitude} DF": magnitude_DF_Ry,
        "\\textbf{p-value} DR": pval_DR_Ry,
        "$\\mathbf{\hat{A}}_{12}$ DR": A12_DR_Ry,
        "\\textbf{Magnitude} DR": magnitude_DR_Ry,
        "\\textbf{p-value} FR": pval_FR_Ry,
        "$\\mathbf{\hat{A}}_{12}$ FR": A12_FR_Ry,
        "\\textbf{Magnitude} FR": magnitude_FR_Ry
        }

        df_table = df_table.append( new_row_all, ignore_index = True )
        df_table = df_table.append( new_row_X, ignore_index = True )
        df_table = df_table.append( new_row_Z, ignore_index = True )
        df_table = df_table.append( new_row_Ry, ignore_index = True )

    df_latex = df_table.to_latex(index=False, escape=False)

    outfile.write(df_latex + "\n")
