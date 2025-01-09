import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import spearmanr


current_directory = os.getcwd()
grandparent_directory = os.path.dirname(os.path.dirname(current_directory))

program_categories = [ "all", "aamp", "qwalk", "var", "gs" ]
qubit_ranges = [ [2, 15], [6, 9], [3, 5], [2, 8], [3, 15] ]
depth_ranges = [ [3, 684], [30, 684], [14, 154], [3, 70], [5, 17] ]

df_table = pd.DataFrame(index = ["\\textbf{Category}", "\\textbf{\\#Qubits}", "\\textbf{Depth}", "\\textbf{Greedy Correlation}", "\\textbf{Random Correlation}", "\\textbf{Factoring p-value}", "\\textbf{Random p-value}"], columns = range( len( program_categories ) ) )

df_all = pd.DataFrame()

for df_reduction_index in range( len(program_categories[1:]) ):

    curr_program_category = program_categories[df_reduction_index + 1]
    filename_df_reduction_loc = os.path.join( grandparent_directory, "results", "postProcessing" )
    filename_df_reduction_factoring = os.path.join( filename_df_reduction_loc, f"df_reduction_run_all_{curr_program_category}_greedy.txt" )
    filename_df_reduction_random = os.path.join( filename_df_reduction_loc, f"df_reduction_run_all_{curr_program_category}_random.txt" )

    df_factoring = pd.read_csv( filename_df_reduction_factoring )
    df_random    = pd.read_csv( filename_df_reduction_random )

    df_factoring['Approach'] = 'Greedy'
    df_random['Approach'] = 'Random'

    df_all = pd.concat( [df_all, df_factoring] )
    df_all = pd.concat( [df_all, df_random] )


for category_index in range( len(program_categories) ):

    current_category = program_categories[ category_index ]
    curr_qubit_range = qubit_ranges[ category_index ]
    curr_depth_range = depth_ranges[ category_index ]

    if current_category == "all":

        df_curr_factoring = df_all[ df_all["Approach"] == "Greedy" ]
        df_curr_random    = df_all[ df_all["Approach"] == "Random" ]

    else:

        df_curr_factoring = df_all[ (df_all["Approach"] == "Greedy") & (df_all["category"] == current_category) ]
        df_curr_random    = df_all[ (df_all["Approach"] == "Random") & (df_all["category"] == current_category) ]

    reduction_rate_factoring_distribution = df_curr_factoring[ "reduction rate" ] * 100
    reduction_rate_random_distribution    = df_curr_random[ "reduction rate" ] * 100

    performance_factor_factoring_distribution = df_curr_factoring[ "reduction factor" ]
    performance_factor_random_distribution = df_curr_random[ "reduction factor" ]

    # compute the spearman correlation
    spearman_factoring, pvalue_factoring = spearmanr( reduction_rate_factoring_distribution, performance_factor_factoring_distribution )

    spearman_random, pvalue_random = spearmanr( reduction_rate_random_distribution, performance_factor_random_distribution )

    # assign values to table dataframe
    df_table.at["\\textbf{\\#Qubits}", category_index] =  f"{curr_qubit_range}"

    # assign depth
    df_table.at["\\textbf{Depth}", category_index] = f"{curr_depth_range}"

    # assign program category
    df_table.at["\\textbf{Category}", category_index] = f"{current_category}"

    # assign spearman
    df_table.at["\\textbf{Greedy Correlation}", category_index] = f"{np.round( spearman_factoring, 5 )}"

    # assign spearman
    df_table.at["\\textbf{Random Correlation}", category_index] = f"{np.round( spearman_random, 5 )}"

    # assign pvalue
    if pvalue_factoring <= 0.001:
        rounded_pval_f = "{:.1e}".format(pvalue_factoring)
        df_table.at["\\textbf{Factoring p-value}", category_index] = f"{rounded_pval_f}"

    else:
        df_table.at["\\textbf{Factoring p-value}", category_index] = f"{np.round(pvalue_factoring, 5)}"

    if pvalue_random <= 0.001:
        rounded_pval_r = "{:.1e}".format(pvalue_random)
        df_table.at["\\textbf{Random p-value}", category_index] = f"{rounded_pval_r}"

    else:
        df_table.at["\\textbf{Random p-value}", category_index] = f"{np.round(pvalue_random, 5)}"


df_table_transposed = df_table.T

df_latex = df_table_transposed.to_latex(index=False, escape=False)

filename_out = os.path.join( "tables", "correlation_table.txt" )

with open( filename_out, "w" ) as outfile:
    outfile.write( df_latex )
