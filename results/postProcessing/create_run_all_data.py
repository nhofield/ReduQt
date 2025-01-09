import pandas as pd
import numpy as np
import os

# ----------
# | Inputs |
# ---------
#Number of runs of experiment 2
number_of_runs = 30

def add_qubits(dataframe):
    dataframe["Qubits"] = 0
    for row_index in dataframe.index:
        program_filename = dataframe.loc[row_index, "Program"]
        num_qubits = program_filename.split("_")[1][1:]
        dataframe.loc[row_index, "Qubits"] = int(num_qubits)
    return dataframe

def add_program_variation(dataframe):

    for program_index in dataframe.index:
        filename = dataframe.loc[program_index, "Program"]
        filename_list = filename.split("_")
        if filename_list[0] == "gs":
            filename_variation = "_".join(filename_list[:2])
        else:
            filename_variation = "_".join(filename_list[:3])

        dataframe.loc[program_index, "Program var"] = filename_variation

    return dataframe


current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)


method_list = ["greedy", "random"]

# df to store all results
df_all_combined = pd.DataFrame()

for method_index in range( len(method_list) ):

    # get current method
    current_method = method_list[method_index]

    #  get median --------------
    filename_experiment1_runtimes = os.path.join( parent_directory, f"results_experiment1_{current_method}", f"{current_method}_runtimes.txt" )

    # Add median reduction runtimes
    df_method_runtimes = pd.read_csv( filename_experiment1_runtimes )

    # get unique programs
    unique_count = df_method_runtimes[' program'].nunique()

    # create empty dataframe for median runtimes
    df_median_runtimes = pd.DataFrame(columns=['program', 'median_run_time'])

    #print(unique_count) gives unique_count = 145
    # define index_value for runs 0-100 for each program
    index_value = 0

    for program_index in range( unique_count ):

        current_program_filename = df_method_runtimes[" program"][index_value]

        if "aamp_q6_6" not in current_program_filename and "aamp_q7_5" not in current_program_filename:

            current_runtimes = df_method_runtimes[" run time"][index_value:index_value + 100]
            median_value_runtime = current_runtimes.median()

            filename_mod_for_check = current_program_filename[:-2]
            df_median_runtimes.loc[program_index] = [filename_mod_for_check, median_value_runtime]

        else:
            pass

        # update index_value for indexing runs 0-100 for each program
        index_value = index_value + 100

    # --------------------------

    df_all = pd.DataFrame()


    for number_index in range( number_of_runs ):

        filename_results_run = os.path.join( parent_directory, f"results_experiment2_{current_method}", f"results_run_{number_index}.txt" )
        read_factoring = pd.read_csv( filename_results_run )
        read_factoring = add_qubits(read_factoring)
        read_factoring = add_program_variation(read_factoring)
        read_factoring["Run Num"] = number_index

        # iterate over each row in the dataframe
        for row_index in range( len(read_factoring) ):

            # if a row is for the factored version
            if read_factoring[" Type"][row_index] == False:

                # filename of the given program, removing mutant and _A extension
                filename_from_df = read_factoring["Program"][row_index][:-7]

                if filename_from_df in ["var_q8_10", "var_q8_11", "var_q8_12", "var_q8_13"]:
                    pass

                else:
                    # get total testing time
                    curr_tot_time = read_factoring.loc[row_index, "  T_tot[s]"]

                    # get index of median value of reduction time
                    index_of_median = df_median_runtimes.loc[df_median_runtimes['program'] == " " + filename_from_df].index[0]

                    # get median value of reduction time
                    median_value = df_median_runtimes["median_run_time"][index_of_median]

                    # add median reduction time to testing time
                    read_factoring.loc[row_index, "  T_tot[s]"] = curr_tot_time + median_value

        df_all = pd.concat( [df_all, read_factoring] )


    # Step 1: Add an empty "category" column (optional, as assignment in Step 2 will create this column if it doesn't exist)
    df_all['category'] = ''

    # Step 2: Extract the category from the "Program" column and assign it to the "category" column
    df_all['category'] = df_all['Program'].str.split('_').str[0]
    if current_method == "greedy":
        df_all["Approach"] = "Greedy"
    else:
        df_all["Approach"] = "Random"


    df_all_combined = pd.concat( [df_all_combined, df_all] )

filename_to_csv_all_combined = os.path.join( parent_directory, "results_run_all.txt" )
df_all_combined.to_csv( filename_to_csv_all_combined )
