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

b_add = "\\textbf{"
b_end = "}"

round_to = 1

current_directory = os.getcwd()
grandparent_directory = os.path.dirname(os.path.dirname(current_directory))
dir_to_results = os.path.join(grandparent_directory, "results", "postProcessing")
dir_to_results_experiment1 = os.path.join(grandparent_directory, "results")


filename_factoring = os.path.join( dir_to_results_experiment1, r"results_experiment1_factoring", "factoring_runtimes.txt" )
filename_random    = os.path.join( dir_to_results_experiment1, r"results_experiment1_random", "random_runtimes.txt" )


repetitions = 100
program_categories = ["aamp", "qwalk", "var", "gs"]
alpha = 0.05

df_factoring = pd.read_csv( filename_factoring, delimiter = ", ", engine='python')
df_random    = pd.read_csv( filename_random, delimiter = ", ", engine='python')

df_factoring['run time'] = df_factoring['run time'].apply(lambda x: 1000*x )
df_random['run time']    = df_random['run time'].apply(lambda x: 1000*x )

df_pass_counts = pd.DataFrame( index = ["all", "aamp", "qwalk", "var", "gs"], columns = ["\\%(NN)", "\\%(S)", "\\%(M)", "\\%(L)", "\\#(NN) Counts", "\\#(S) Counts", "\\#(M) Counts", "\\#(L) Counts", "Total Counts", "Total Not Pass or (N)"] )
df_pass_counts.index.name = "category"

df_pass_counts_r = pd.DataFrame( index = ["all", "aamp", "qwalk", "var", "gs"], columns = ["\\%(NN)", "\\%(S)", "\\%(M)", "\\%(L)", "\\#(NN) Counts", "\\#(S) Counts", "\\#(M) Counts", "\\#(L) Counts", "Total Counts"] )
df_pass_counts_r.index.name = "category"


df_table_total = pd.DataFrame()

number_of_fail_or_N_all = 0
number_total_counts_all = 0

for program_category_index in range( len(program_categories) ):

    curr_program_category = program_categories[ program_category_index ]

    filename_latex_table = os.path.join( "tables", "runtime", "Full", f"latex_table_runtime_{curr_program_category}.txt" )

    with open( filename_latex_table , "w") as outfile:

        filename_df_all_file = os.path.join( dir_to_results, f"df_reduction_run_all_{curr_program_category}_factoring.txt" )
        df_rq2_data_factored = pd.read_csv( filename_df_all_file )

        df_curr_factoring = df_factoring[ df_factoring['program'].str.startswith(program_categories[program_category_index]) ]
        df_curr_random    = df_random[ df_random['program'].str.startswith(program_categories[program_category_index]) ]

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

        df_table = pd.DataFrame(index = ["\\textbf{ID}", "\\textbf{Category}", "\\textbf{\\#Qubits}", "\\textbf{Depth}", "\\textbf{\\#T}", "\\textbf{f(\\#Q)}", "\\textbf{Factoring [ms]}", "\\textbf{Random [ms]}", "\\textbf{p-value}", "$\\mathbf{\hat{A}}_{12}$", "\\textbf{Magnitude}"], columns = range(len(unique_progs)))

        # to count the number of passes favoring greedy
        number_of_passes = 0
        number_of_L_s = 0
        number_of_M_s = 0
        number_of_S_s = 0

        # to count the number of passes favoring random
        number_of_passes_r = 0
        number_of_S_s_r = 0
        number_of_M_s_r = 0
        number_of_L_s_r = 0

        # count number of N or not passed
        number_of_not_passed_or_Ns = 0

        for prog_index in range( len(unique_progs) ):

            number_total_counts_all += 1

            prog_curr_distribution_factoring = df_curr_factoring[ df_curr_factoring["program"] == unique_progs[prog_index] ]
            prog_curr_distribution_random    = df_curr_random[ df_curr_random["program"] == unique_progs[prog_index] ]


            # fetch number of qubits
            prog_filename = prog_curr_distribution_factoring.at[prog_curr_distribution_factoring["program"].index[0] ,"program"][:-2]
            number_of_qubits = prog_filename.split("_")[1][1:]

            # fetch average runtime rate
            runtime_rate_average_factoring = prog_curr_distribution_factoring["run time"].mean()
            runtime_rate_average_random    = prog_curr_distribution_random["run time"].mean()

            # fetch stds
            runtime_rate_std_factoring = prog_curr_distribution_factoring["run time"].std()
            runtime_rate_std_random    = prog_curr_distribution_random["run time"].std()

            # fetch depth
            matching_row = df_rq2_data_factored.loc[df_rq2_data_factored['Program Name'] == prog_filename].iloc[0]
            depth = matching_row[" depth"]


            # fetch #T
            number_of_transformations = int( prog_curr_distribution_factoring["number_of_transformations"].max() )


            runtime_distribution_factoring = np.array(prog_curr_distribution_factoring["run time"])
            runtime_distribution_random    = np.array(prog_curr_distribution_random["run time"])


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

            if VA_category == "(N)" or p_value > alpha:
                number_of_not_passed_or_Ns += 1
                number_of_fail_or_N_all += 1


            # see if test passed (1) or failed (0)
            if p_value <= alpha and VA_category != "(N)":

                if VA_A > 0.5:
                    # greedy is better
                    number_of_passes += 1

                    if VA_category == "(S)":
                        number_of_S_s += 1
                    else:
                        pass

                    if VA_category == "(M)":
                        number_of_M_s += 1
                    else:
                        pass

                    if VA_category == "(L)":
                        number_of_L_s += 1
                    else:
                        pass

                else:
                    pass

                if VA_A < 0.5:
                    # random is better
                    number_of_passes_r += 1

                    if VA_category == "(S)":
                        number_of_S_s_r += 1
                    else:
                        pass

                    if VA_category == "(M)":
                        number_of_M_s_r += 1
                    else:
                        pass

                    if VA_category == "(L)":
                        number_of_L_s_r += 1
                    else:
                        pass

                else:
                    pass

                # assign values to table dataframe
                df_table.at["\\textbf{ID}", prog_index] = b_add + f"{prog_index}" + b_end

                # assign values to table dataframe
                df_table.at["\\textbf{\\#Qubits}", prog_index] = b_add + f"{number_of_qubits}" + b_end

                # assign depth
                df_table.at["\\textbf{Depth}", prog_index] = b_add + f"{depth}" + b_end

                # assign program category
                df_table.at["\\textbf{Category}", prog_index] = b_add + f"{curr_program_category}" + b_end

                # assign #T and f(#Q)
                df_table.at["\\textbf{\\#T}", prog_index] = b_add + f"{number_of_transformations}" + b_end
                df_table.at["\\textbf{f(\\#Q)}", prog_index] = b_add + f"{f_complexity( int( number_of_qubits) )}" + b_end


                # assign average and std
                df_table.at["\\textbf{Factoring [ms]}", prog_index] = b_add + f"{runtime_rate_average_factoring:.1f} $\\pm$ {runtime_rate_std_factoring:.1f}" + b_end
                df_table.at["\\textbf{Random [ms]}", prog_index] = b_add + f"{runtime_rate_average_random:.1f} $\\pm$ {runtime_rate_std_random:.1f}" + b_end

                # assign test result
                df_table.at["\\textbf{Magnitude}", prog_index] = b_add + f"{VA_category}" + b_end

                # assign Vargha_Delaney_A effect size
                df_table.at["$\\mathbf{\hat{A}}_{12}$", prog_index] = b_add + f"{np.round(VA_A, 3)}" + b_end

                if p_value <= 0.001:
                    rounded_pval = "{:.1e}".format(p_value)
                    df_table.at["\\textbf{p-value}", prog_index] = b_add + f"{rounded_pval}" + b_end

                else:
                    df_table.at["\\textbf{p-value}", prog_index] = b_add + f"{np.round(p_value, 5)}" + b_end


            else:

                # assign values to table dataframe
                df_table.at["\\textbf{ID}", prog_index] = f"{prog_index}"

                # assign values to table dataframe
                df_table.at["\\textbf{\\#Qubits}", prog_index] = f"{number_of_qubits}"

                # assign depth
                df_table.at["\\textbf{Depth}", prog_index] = f"{depth}"

                # assign program category
                df_table.at["\\textbf{Category}", prog_index] = f"{curr_program_category}"

                # assign #T and f(#Q)
                df_table.at["\\textbf{\\#T}", prog_index] = f"{number_of_transformations}"
                df_table.at["\\textbf{f(\\#Q)}", prog_index] = f"{f_complexity( int( number_of_qubits) )}"


                # assign average and std
                df_table.at["\\textbf{Factoring [ms]}", prog_index] = f"{runtime_rate_average_factoring:.1f} $\\pm$ {runtime_rate_std_factoring:.1f}"
                df_table.at["\\textbf{Random [ms]}", prog_index] = f"{runtime_rate_average_random:.1f} $\\pm$ {runtime_rate_std_random:.1f}"



                # assign test result
                df_table.at["\\textbf{Magnitude}", prog_index] = f"{VA_category}"

                # assign Vargha_Delaney_A effect size
                df_table.at["$\\mathbf{\hat{A}}_{12}$", prog_index] = f"{np.round(VA_A, 3)}"



                if p_value <= 0.001:
                    rounded_pval = "{:.1e}".format(p_value)
                    df_table.at["\\textbf{p-value}", prog_index] = f"{rounded_pval}"

                else:
                    df_table.at["\\textbf{p-value}", prog_index] = f"{np.round(p_value, 5)}"



        df_table_transposed = df_table.T

        df_pass_counts.loc[curr_program_category, "Total Not Pass or (N)"] = round( number_of_not_passed_or_Ns/len(df_table_transposed)*100, round_to)

        df_pass_counts.loc[curr_program_category, "\\%(NN)"] = round( number_of_passes/len(df_table_transposed)*100, round_to)
        df_pass_counts.loc[curr_program_category, "\\%(S)"] = round( number_of_S_s/len(df_table_transposed)*100, round_to)
        df_pass_counts.loc[curr_program_category, "\\%(M)"] = round( number_of_M_s/len(df_table_transposed)*100, round_to)
        df_pass_counts.loc[curr_program_category, "\\%(L)"] = round( number_of_L_s/len(df_table_transposed)*100, round_to)
        df_pass_counts.loc[curr_program_category, "\\#(NN) Counts"] = number_of_passes
        df_pass_counts.loc[curr_program_category, "\\#(S) Counts"] = number_of_S_s
        df_pass_counts.loc[curr_program_category, "\\#(M) Counts"] = number_of_M_s
        df_pass_counts.loc[curr_program_category, "\\#(L) Counts"] = number_of_L_s
        df_pass_counts.loc[curr_program_category, "Total Counts"] = len(df_table_transposed)

        df_pass_counts_r.loc[curr_program_category, "\\%(NN)"] = round( number_of_passes_r/len(df_table_transposed)*100, round_to)
        df_pass_counts_r.loc[curr_program_category, "\\%(S)"]  = round( number_of_S_s_r/len(df_table_transposed)*100, round_to)
        df_pass_counts_r.loc[curr_program_category, "\\%(M)"]  = round( number_of_M_s_r/len(df_table_transposed)*100, round_to)
        df_pass_counts_r.loc[curr_program_category, "\\%(L)"]  = round( number_of_L_s_r/len(df_table_transposed)*100, round_to)
        df_pass_counts_r.loc[curr_program_category, "\\#(NN) Counts"] = number_of_passes_r
        df_pass_counts_r.loc[curr_program_category, "\\#(S) Counts"]  = number_of_S_s_r
        df_pass_counts_r.loc[curr_program_category, "\\#(M) Counts"]  = number_of_M_s_r
        df_pass_counts_r.loc[curr_program_category, "\\#(L) Counts"]  = number_of_L_s_r
        df_pass_counts_r.loc[curr_program_category, "Total Counts"]   = len(df_table_transposed)


        df_latex = df_table_transposed.to_latex(index=False, escape=False)

        outfile.write(df_latex + "\n")


df_pass_counts.loc["all", "Total Not Pass or (N)"] = round( number_of_fail_or_N_all/number_total_counts_all*100, round_to)

# favoring greedy
all_tot = df_pass_counts.loc["aamp", "Total Counts"]   + df_pass_counts.loc["qwalk", "Total Counts"]   + df_pass_counts.loc["var", "Total Counts"] + df_pass_counts.loc["gs", "Total Counts"]
all_NN  = df_pass_counts.loc["aamp", "\\#(NN) Counts"] + df_pass_counts.loc["qwalk", "\\#(NN) Counts"] + df_pass_counts.loc["var", "\\#(NN) Counts"] + df_pass_counts.loc["gs", "\\#(NN) Counts"]
all_S   = df_pass_counts.loc["aamp", "\\#(S) Counts"]  + df_pass_counts.loc["qwalk", "\\#(S) Counts"]  + df_pass_counts.loc["var", "\\#(S) Counts"] + df_pass_counts.loc["gs", "\\#(S) Counts"]
all_M   = df_pass_counts.loc["aamp", "\\#(M) Counts"]  + df_pass_counts.loc["qwalk", "\\#(M) Counts"]  + df_pass_counts.loc["var", "\\#(M) Counts"] + df_pass_counts.loc["gs", "\\#(M) Counts"]
all_L   = df_pass_counts.loc["aamp", "\\#(L) Counts"]  + df_pass_counts.loc["qwalk", "\\#(L) Counts"]  + df_pass_counts.loc["var", "\\#(L) Counts"] + df_pass_counts.loc["gs", "\\#(L) Counts"]

df_pass_counts.loc["all", "\\%(NN)"] = round( all_NN/all_tot*100, round_to )
df_pass_counts.loc["all", "\\%(S)"] = round( all_S/all_tot*100, round_to )
df_pass_counts.loc["all", "\\%(M)"] = round( all_M/all_tot*100, round_to )
df_pass_counts.loc["all", "\\%(L)"] = round( all_L/all_tot*100, round_to )


csv_filename = os.path.join( "tables", "runtime", "Summary", "runtime_pass_counts.csv" )
df_pass_counts.to_csv( csv_filename )

# favoring random
all_tot_r = df_pass_counts_r.loc["aamp", "Total Counts"]   + df_pass_counts_r.loc["qwalk", "Total Counts"]   + df_pass_counts_r.loc["var", "Total Counts"] + df_pass_counts_r.loc["gs", "Total Counts"]
all_NN_r  = df_pass_counts_r.loc["aamp", "\\#(NN) Counts"] + df_pass_counts_r.loc["qwalk", "\\#(NN) Counts"] + df_pass_counts_r.loc["var", "\\#(NN) Counts"] + df_pass_counts_r.loc["gs", "\\#(NN) Counts"]
all_S_r   = df_pass_counts_r.loc["aamp", "\\#(S) Counts"]  + df_pass_counts_r.loc["qwalk", "\\#(S) Counts"]  + df_pass_counts_r.loc["var", "\\#(S) Counts"] + df_pass_counts_r.loc["gs", "\\#(S) Counts"]
all_M_r   = df_pass_counts_r.loc["aamp", "\\#(M) Counts"]  + df_pass_counts_r.loc["qwalk", "\\#(M) Counts"]  + df_pass_counts_r.loc["var", "\\#(M) Counts"] + df_pass_counts_r.loc["gs", "\\#(M) Counts"]
all_L_r   = df_pass_counts_r.loc["aamp", "\\#(L) Counts"]  + df_pass_counts_r.loc["qwalk", "\\#(L) Counts"]  + df_pass_counts_r.loc["var", "\\#(L) Counts"] + df_pass_counts_r.loc["gs", "\\#(L) Counts"]

df_pass_counts_r.loc["all", "\\%(NN)"] = round( all_NN_r/all_tot_r*100, round_to )
df_pass_counts_r.loc["all", "\\%(S)"] = round( all_S_r/all_tot_r*100, round_to )
df_pass_counts_r.loc["all", "\\%(M)"] = round( all_M_r/all_tot_r*100, round_to )
df_pass_counts_r.loc["all", "\\%(L)"] = round( all_L_r/all_tot_r*100, round_to )


filename_csv_r = os.path.join( "tables", "runtime", "Summary", "runtime_pass_counts_r.csv" )
df_pass_counts_r.to_csv( filename_csv_r )
