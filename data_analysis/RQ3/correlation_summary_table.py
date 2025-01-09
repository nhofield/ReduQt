import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import spearmanr


current_directory = os.getcwd()
grandparent_directory = os.path.dirname(os.path.dirname(current_directory))

mutant_column_names = ["ms all", "ms X", "ms Z", "ms Ry"]
mutant_types_list = ["all", "X", "Z", "$R_y$"]
program_categories = [ "all", "aamp", "qwalk", "var", "gs" ]

qubit_ranges = [ [2, 15], [6, 9], [3, 5], [2, 8], [3, 15] ]
depth_ranges = [ [3, 684], [30, 684], [14, 154], [3, 70], [5, 17] ]

df_table = pd.DataFrame(columns = ["\\textbf{Category}", "\\textbf{Mutant Type}", "\\textbf{\\#Qubits}", "\\textbf{Depth}", "\\textbf{Greedy Correlation}", "\\textbf{Random Correlation}", "\\textbf{Greedy p-value}", "\\textbf{Random p-value}"] )

df_all = pd.DataFrame()
df_ms_all = pd.DataFrame()

for df_reduction_index in range( len(program_categories[1:]) ):

    curr_program_category = program_categories[df_reduction_index + 1]

    # to get reduction rate
    filename_df_reduction_loc = os.path.join( grandparent_directory, "results", "postProcessing" )
    filename_df_reduction_factoring = os.path.join( filename_df_reduction_loc, f"df_reduction_run_all_{curr_program_category}_greedy.txt" )
    filename_df_reduction_random = os.path.join( filename_df_reduction_loc, f"df_reduction_run_all_{curr_program_category}_random.txt" )

    df_factoring = pd.read_csv( filename_df_reduction_factoring )
    df_random    = pd.read_csv( filename_df_reduction_random )

    df_factoring['Approach'] = 'Greedy'
    df_random['Approach'] = 'Random'

    df_all = pd.concat( [df_all, df_factoring] )
    df_all = pd.concat( [df_all, df_random] )

    # to get mutation score
    filename_df_ms_loc = "msDistributionsForCorrelation"
    filename_df_ms_factoring = os.path.join( filename_df_ms_loc, f"mutation_score_distribution_{curr_program_category}_greedy.csv" )
    filename_df_ms_random = os.path.join( filename_df_ms_loc, f"mutation_score_distribution_{curr_program_category}_random.csv" )

    df_ms_factoring = pd.read_csv( filename_df_ms_factoring )
    df_ms_random    = pd.read_csv( filename_df_ms_random )

    df_ms_factoring['Approach'] = 'Greedy'
    df_ms_random['Approach'] = 'Random'

    df_ms_all = pd.concat( [df_ms_all, df_ms_factoring] )
    df_ms_all = pd.concat( [df_ms_all, df_ms_random] )


#  ---------------------
# | clean up df_ms_all |
# ---------------------

# create ampty df with the same columns as df_all
df_ms_all_clean = pd.DataFrame(columns=["Program var", "reduction rate", "Run Num"])

unique_programs_df_ms_all = df_ms_all["Program var"].unique()

for program_unique_index in range( len(unique_programs_df_ms_all) ):

    current_program = unique_programs_df_ms_all[ program_unique_index ]

    df_temp_greedy = df_all[ (df_all["Program Name"] == current_program) & (df_all["Approach"] == "Greedy") ]

    reduction_rate_greedy = df_temp_greedy.loc[ df_temp_greedy.index[0], "reduction rate"]

    category = df_temp_greedy.loc[ df_temp_greedy.index[0], "category"]

    for run_index in range( 30 ):

        new_row = pd.Series({
            "Program var": current_program,
            "reduction rate": reduction_rate_greedy,
            "Run Num": run_index,
            "Approach": "Greedy",
            "category": category
        })

        df_ms_all_clean = df_ms_all_clean.append(new_row, ignore_index=True)



for program_unique_index in range( len(unique_programs_df_ms_all) ):

    current_program = unique_programs_df_ms_all[ program_unique_index ]

    df_temp_random = df_all[ (df_all["Program Name"] == current_program) & (df_all["Approach"] == "Random") ]

    reduction_rate_random = df_temp_random.loc[ df_temp_random.index[0], "reduction rate"]

    category = df_temp_random.loc[ df_temp_random.index[0], "category"]

    for run_index in range( 30 ):

        new_row = pd.Series({
            "Program var": current_program,
            "reduction rate": reduction_rate_random,
            "Run Num": run_index,
            "Approach": "Random",
            "category": category
        })

        df_ms_all_clean = df_ms_all_clean.append(new_row, ignore_index=True)


df_ms_all = df_ms_all.reset_index(drop=True)
df_ms_all_clean = df_ms_all_clean.reset_index(drop=True)


df_ms_all_clean_factored = df_ms_all_clean[ df_ms_all_clean["Approach"] == "Greedy" ]
df_ms_all_clean_random = df_ms_all_clean[ df_ms_all_clean["Approach"] == "Random" ]

df_ms_all_clean_factored_aamp = df_ms_all_clean_factored[ df_ms_all_clean_factored["category"] == "aamp" ]
df_ms_all_clean_factored_qwalk = df_ms_all_clean_factored[ df_ms_all_clean_factored["category"] == "qwalk" ]
df_ms_all_clean_factored_var = df_ms_all_clean_factored[ df_ms_all_clean_factored["category"] == "var" ]
df_ms_all_clean_factored_gs = df_ms_all_clean_factored[ df_ms_all_clean_factored["category"] == "gs" ]

df_ms_all_clean_random_aamp = df_ms_all_clean_random[ df_ms_all_clean_random["category"] == "aamp" ]
df_ms_all_clean_random_qwalk = df_ms_all_clean_random[ df_ms_all_clean_random["category"] == "qwalk" ]
df_ms_all_clean_random_var = df_ms_all_clean_random[ df_ms_all_clean_random["category"] == "var" ]
df_ms_all_clean_random_gs = df_ms_all_clean_random[ df_ms_all_clean_random["category"] == "gs" ]

df_ms_all_clean = pd.concat( [df_ms_all_clean_factored_aamp, df_ms_all_clean_random_aamp, \
                              df_ms_all_clean_factored_qwalk, df_ms_all_clean_random_qwalk, \
                              df_ms_all_clean_factored_var, df_ms_all_clean_random_var, \
                              df_ms_all_clean_factored_gs, df_ms_all_clean_random_gs] )

df_ms_all_clean = df_ms_all_clean.reset_index(drop=True)


for program_index in df_ms_all.index:

    reduction_rate_value = df_ms_all_clean.loc[program_index, "reduction rate"]
    df_ms_all.loc[program_index, "reduction rate"] = reduction_rate_value




for category_index in range( len(program_categories) ):

    current_category = program_categories[ category_index ]
    curr_qubit_range = qubit_ranges[ category_index ]
    curr_depth_range = depth_ranges[ category_index ]

    if current_category == "all":

        df_curr_factoring = df_ms_all[ df_ms_all["Approach"] == "Greedy" ]
        df_curr_random    = df_ms_all[ df_ms_all["Approach"] == "Random" ]

    else:

        df_curr_factoring = df_ms_all[ (df_ms_all["Approach"] == "Greedy") & (df_ms_all["category"] == current_category) ]
        df_curr_random    = df_ms_all[ (df_ms_all["Approach"] == "Random") & (df_ms_all["category"] == current_category) ]


    reduction_rate_factoring_distribution = df_curr_factoring[ "reduction rate" ] * 100
    reduction_rate_random_distribution    = df_curr_random[ "reduction rate" ] * 100



    for mutant_type_index in range(4):

        current_mutant_type = mutant_types_list[mutant_type_index]
        current_mutant_column_name = mutant_column_names[mutant_type_index]

        mutation_score_factoring_distribution = df_curr_factoring[ current_mutant_column_name ]
        mutation_score_random_distribution = df_curr_random[ current_mutant_column_name ]

        # compute the spearman correlation
        spearman_factoring, pvalue_factoring = spearmanr( reduction_rate_factoring_distribution, mutation_score_factoring_distribution )

        spearman_random, pvalue_random = spearmanr( reduction_rate_random_distribution, mutation_score_random_distribution )


        # assign pvalue
        if pvalue_factoring <= 0.001:
            rounded_pval_f = "{:.1e}".format(pvalue_factoring)
            p_val_greedy_for_table = f"{rounded_pval_f}"

        else:
            p_val_greedy_for_table = f"{np.round(pvalue_factoring, 5)}"

        if pvalue_random <= 0.001:
            rounded_pval_r = "{:.1e}".format(pvalue_random)
            p_val_random_for_table = f"{rounded_pval_r}"

        else:
            p_val_random_for_table = f"{np.round(pvalue_random, 5)}"


        new_row = {
        "\\textbf{Category}": f"{current_category}",
        "\\textbf{Mutant Type}": f"{current_mutant_type}",
        "\\textbf{\\#Qubits}": f"{curr_qubit_range}",
        "\\textbf{Depth}": f"{curr_depth_range}",
        "\\textbf{Greedy Correlation}": f"{np.round( spearman_factoring, 5 )}",
        "\\textbf{Random Correlation}": f"{np.round( spearman_random, 5 )}",
        "\\textbf{Greedy p-value}": p_val_greedy_for_table,
        "\\textbf{Random p-value}": p_val_random_for_table
        }

        df_table = df_table.append( pd.Series(new_row), ignore_index=True )



df_latex = df_table.to_latex(index=False, escape=False)

filename_out = os.path.join( "tables", "correlation_table.txt" )

with open( filename_out, "w" ) as outfile:
    outfile.write( df_latex )
