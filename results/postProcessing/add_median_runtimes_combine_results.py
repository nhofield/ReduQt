import numpy as np
import pandas as pd
import os

#aamp_q6_6 and aamp_q7_5 skipped

# ----------
# | Inputs |
# ---------
#Number of runs of experiment 2
number_of_runs = 30



current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)

methods_list = ["greedy", "random"]

category_list = ["var", "gs", "qwalk", "aamp"]

#category_list = ["aamp", "gs", "qwalk", "var"]

all_median_runtimes = []

def add_qubits(dataframe):
    dataframe["Qubits"] = 0
    for row_index in dataframe.index:
        program_filename = dataframe.loc[row_index, "Program"]
        num_qubits = program_filename.split("_")[1][1:]
        dataframe.loc[row_index, "Qubits"] = int(num_qubits)
    return dataframe

for methods_index in range( len(methods_list) ):

    current_method = methods_list[methods_index]

    # Define the name of the subdirectory
    results_experiment2_method_dir_name = f"results_experiment2_{current_method}"
    results_experiment1_method_dir_name = f"results_experiment1_{current_method}"


    # Get the full path to the subdirectory
    results_experiment2_method_dir = os.path.join(parent_directory, results_experiment2_method_dir_name)
    results_experiment1_method_dir = os.path.join(parent_directory, results_experiment1_method_dir_name)

    filename_read_runtimes = os.path.join( results_experiment1_method_dir, f"{current_method}_runtimes.txt" )

    # get dataframe for experiment 1 method_runtimes.txt
    df_method_runtimes = pd.read_csv( filename_read_runtimes )

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

    # define empty dataframe for storing all result data for all runs
    df_all_results = pd.DataFrame()


    for run_index in range( number_of_runs ):

        filename_read_runs = os.path.join( results_experiment2_method_dir, f"results_run_{run_index}.txt" )
        # get results for a given run from a result file for experiment 2 and store in dataframe
        df_method_runtimes = pd.read_csv( filename_read_runs )

        # iterate over each row in the dataframe
        for row_index in range( len(df_method_runtimes) ):

            # if a row is for the factored version
            if df_method_runtimes[" Type"][row_index] == False:

                # filename of the given program, removing mutant and _A extension
                filename_from_df = df_method_runtimes["Program"][row_index][:-7]

                if filename_from_df in ["var_q8_10", "var_q8_11", "var_q8_12", "var_q8_13"]:
                    pass

                else:
                    # get total testing time
                    curr_tot_time = df_method_runtimes.loc[row_index, "  T_tot[s]"]

                    # get index of median value of reduction time
                    index_of_median = df_median_runtimes.loc[df_median_runtimes['program'] == " " + filename_from_df].index[0]

                    # get median value of reduction time
                    median_value = df_median_runtimes["median_run_time"][index_of_median]

                    # add median reduction time to testing time
                    df_method_runtimes.loc[row_index, "  T_tot[s]"] = curr_tot_time + median_value

        # update result dataframe
        df_all_results = pd.concat( [df_all_results, df_method_runtimes] )


    # add category column
    df_all_results["category"] = "category"
    df_all_results["Program Name"] = "prog name missing"

    df_all_results = df_all_results.reset_index()

    # add category var, aamp, gs or qwalk to each program
    for prog_index in df_all_results.index:

        filename_prog = df_all_results.loc[ prog_index, "Program" ]
        filename_prog_list = filename_prog.split("_")
        filename_category = filename_prog_list[0]
        df_all_results.loc[ prog_index, "category" ] = filename_category
        if r"_A" in filename_prog:
            df_all_results.loc[ prog_index, "Program Name" ] = filename_prog[:-7]
        else:
            df_all_results.loc[ prog_index, "Program Name" ] = filename_prog[:-5]


    # -----------------------------#
    # |    Part 2 Category Wise  | #
    # ---------------------------| #

    column_names = ["program", "basis", "reduction_rate"]
    filename_results_bases_median = os.path.join( results_experiment1_method_dir, f"{current_method}_bases_median.txt" )
    df_method_runtimes_median = pd.read_csv( filename_results_bases_median, names=column_names, delimiter = ", ", engine = "python" )


    for category_index in range( len(category_list) ):

        current_category = category_list[ category_index ]
        df_results_category = df_all_results[ df_all_results["category"] == current_category ]
        df_results_category["Program Name Index"] = "prog name index missing"

        unique_progs = df_results_category["Program Name"].unique()
        program_category_dictionary = {}

        for prog_filename_index in range( len(unique_progs) ):
            program_category_dictionary[ unique_progs[prog_filename_index] ] = prog_filename_index

        for prog_index in df_results_category.index:
            fname = df_results_category["Program Name"][prog_index]
            df_results_category.loc[prog_index, "Program Name Index"] = f"{program_category_dictionary[fname]}"

        df_results_category_reduction = df_results_category[ df_results_category[" Type"] == True ]
        df_results_category_reduction["reduction factor"] = "empty"
        df_results_category_reduction["Performance Type"] = "empty"
        df_results_category_reduction["reduction rate"] = "empty"

        for prog_index in df_results_category_reduction.index:
            time_default = df_results_category["  T_tot[s]"][prog_index]
            time_A = df_results_category["  T_tot[s]"][prog_index + 1]

            curr_filename_prog = df_results_category.loc[prog_index, "Program Name"]
            if curr_filename_prog in ["var_q8_10", "var_q8_11", "var_q8_12", "var_q8_13"]:
                pass
            else:
                index_of_reduction_rate = df_method_runtimes_median[df_method_runtimes_median["program"]== curr_filename_prog + r"_R"].index[0]
                df_results_category_reduction.loc[prog_index, "reduction rate"] = 1 - df_method_runtimes_median.loc[index_of_reduction_rate, "reduction_rate"]

            if time_default < time_A:
                df_results_category_reduction.loc[prog_index, "Performance Type"] = f"Slowdown"
                df_results_category_reduction.loc[prog_index, "reduction factor"] = - time_A / time_default
            else:
                df_results_category_reduction.loc[prog_index, "Performance Type"] = f"Speedup"
                df_results_category_reduction.loc[prog_index, "reduction factor"] = time_default / time_A

        df_results_category_reduction = df_results_category_reduction.iloc[:, 1:]  # This keeps all rows but starts from the 3rd column (0-indexed)
        df_results_category_reduction = df_results_category_reduction.reset_index(drop=True)
        df_results_category_reduction = add_qubits(df_results_category_reduction)
        df_results_category_reduction.to_csv(f"df_reduction_run_all_{current_category}_{current_method}.txt", index=False)

print(all_median_runtimes)
