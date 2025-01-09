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


def perform_mann_whitney_and_get_effect_size(df1, df2, pval_tag, prog_index):

    # Perform the Kruskal-Wallis H test
    statistic, p_value = stats.mannwhitneyu( df1, df2, alternative = "greater" )

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
        df_table.at["\\textbf{Magnitude}" + f" {pval_tag}", prog_index] = b_add + f"{VA_category}" + b_end

        # assign Vargha_Delaney_A effect size
        df_table.at["$\\mathbf{\hat{A}}_{12}$" + f" {pval_tag}", prog_index] = b_add + f"{np.round(VA_A, 3)}" + b_end

        if p_value <= 0.001:
            rounded_pval = "{:.1e}".format(p_value)
            df_table.at["\\textbf{p-value}" + f" {pval_tag}", prog_index] = b_add + f"{rounded_pval}" + b_end

        else:
            df_table.at["\\textbf{p-value}" + f" {pval_tag}", prog_index] = b_add + f"{np.round(p_value, 5)}" + b_end

        pass_or_fail_variable = "fail"

        return pass_or_fail_variable, VA_A, VA_category

    else:

        # assign test result
        df_table.at["\\textbf{Magnitude}" + f" {pval_tag}", prog_index] = f"{VA_category}"

        # assign Vargha_Delaney_A effect size
        df_table.at["$\\mathbf{\hat{A}}_{12}$" + f" {pval_tag}", prog_index] = f"{np.round(VA_A, 3)}"

        if p_value <= 0.001:
            rounded_pval = "{:.1e}".format(p_value)
            df_table.at["\\textbf{p-value}" + f" {pval_tag}", prog_index] = f"{rounded_pval}"

        else:
            df_table.at["\\textbf{p-value}" + f" {pval_tag}", prog_index] = f"{np.round(p_value, 5)}"

        return pass_or_fail_variable, VA_A, VA_category


def perform_kruskal_wallis(df1, df2, df3, prog_index, pass_or_fail):
    # Perform the Kruskal-Wallis H test
    statistic, p_value = stats.kruskal( df1, df2, df3 )

    if pass_or_fail == "fail":

        if p_value <= 0.001:
            rounded_pval = "{:.1e}".format(p_value)
            df_table.at["\\textbf{p-value} DFR", prog_index] = b_add + f"{rounded_pval}" + b_end

        else:
            df_table.at["\\textbf{p-value} DFR", prog_index] = b_add + f"{np.round(p_value, 5)}" + b_end

    else:

        if p_value <= 0.001:
            rounded_pval = "{:.1e}".format(p_value)
            df_table.at["\\textbf{p-value} DFR", prog_index] = f"{rounded_pval}"

        else:
            df_table.at["\\textbf{p-value} DFR", prog_index] = f"{np.round(p_value, 5)}"


b_add = "\\textbf{"
b_end = "}"

b_add = ""
b_end = ""

df_pass_counts = pd.DataFrame( index = ["all", "aamp", "qwalk", "var", "gs"], columns = ["\\%passed (NN) DF", "\\%equal DF", "\\%passed (NN) DR", "\\%equal DR", "\\%passed (NN) FR", "\\%equal FR", "\\%passed (NN) Counts DF", "\\%passed (NN) Counts DR", "\\%passed (NN) Counts FR", "Total Counts"] )
df_pass_counts.index.name = "category"

df_pass_counts_r = pd.DataFrame( index = ["all", "aamp", "qwalk", "var", "gs"], columns = ["\\%passed (NN) DF", "\\%passed (NN) DR", "\\%passed (NN) FR", "\\%passed (NN) Counts DF", "\\%passed (NN) Counts DR", "\\%passed (NN) Counts FR", "Total Counts"] )
df_pass_counts_r.index.name = "category"


alpha = 0.05
round_to = 1

current_directory = os.getcwd()
grandparent_directory = os.path.dirname(os.path.dirname(current_directory))

full_or_summary = "Full"

program_categories = ["aamp", "qwalk", "var", "gs"]

filename_all_data = os.path.join( grandparent_directory, "results", r"results_run_all.txt" )

df_all_data = pd.read_csv( filename_all_data )

df_default   = df_all_data[ (df_all_data["Approach"] == "Greedy") & (df_all_data[" Type"] == True) ]
df_factoring = df_all_data[ (df_all_data["Approach"] == "Greedy") & (df_all_data[" Type"] == False) ]
df_random    = df_all_data[ (df_all_data["Approach"] == "Random") & (df_all_data[" Type"] == False) ]

number_equal_DF_all = 0
number_equal_DR_all = 0
number_equal_FR_all = 0


for program_category_index in range( len(program_categories) ):

    curr_program_category = program_categories[ program_category_index ]

    filename_latex_table = os.path.join( "tables", full_or_summary, f"latex_table_runtime_{curr_program_category}.txt" )

    with open( filename_latex_table , "w") as outfile:

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

        df_table = pd.DataFrame(index = ["\\textbf{ID}", "\\textbf{Category}", "\\textbf{\\#Qubits}", "\\textbf{Depth}", "\\textbf{Default [s]}", "\\textbf{Factoring [s]}", "\\textbf{Random [s]}", "\\textbf{p-value} DFR", "\\textbf{p-value} DF", "$\\mathbf{\hat{A}}_{12}$ DF", "\\textbf{Magnitude} DF", "\\textbf{p-value} DR", "$\\mathbf{\hat{A}}_{12}$ DR", "\\textbf{Magnitude} DR", "\\textbf{p-value} FR", "$\\mathbf{\hat{A}}_{12}$ FR", "\\textbf{Magnitude} FR"], columns = range(len(unique_progs)))

        number_fail_DF = 0
        number_fail_DR = 0
        number_fail_FR = 0

        # reversed, so FD, RD and RF
        number_fail_DF_r = 0
        number_fail_DR_r = 0
        number_fail_FR_r = 0

        number_equal_DF = 0
        number_equal_DR = 0
        number_equal_FR = 0


        for prog_index in range( len(unique_progs) ):

            current_program = unique_progs[prog_index]
            prog_curr_distribution_default   = df_curr_default[ df_curr_default["Program var"] == current_program ]
            prog_curr_distribution_factoring = df_curr_factoring[ df_curr_factoring["Program var"] == current_program ]
            prog_curr_distribution_random    = df_curr_random[ df_curr_random["Program var"] == current_program ]

            # fetch number of qubits
            number_of_qubits = prog_curr_distribution_default.loc[ prog_curr_distribution_default["Qubits"].index[0], "Qubits"]

            # fetch average runtime rate
            runtime_rate_average_default   = prog_curr_distribution_default["  T_tot[s]"].mean()
            runtime_rate_average_factoring = prog_curr_distribution_factoring["  T_tot[s]"].mean()
            runtime_rate_average_random    = prog_curr_distribution_random["  T_tot[s]"].mean()

            # fetch stds
            runtime_rate_std_default   = prog_curr_distribution_default["  T_tot[s]"].std()
            runtime_rate_std_factoring = prog_curr_distribution_factoring["  T_tot[s]"].std()
            runtime_rate_std_random    = prog_curr_distribution_random["  T_tot[s]"].std()


            # fetch depth
            depth = prog_curr_distribution_default.loc[ prog_curr_distribution_default[" depth"].index[0], " depth"]

            runtime_distribution_default   = np.array(prog_curr_distribution_default["  T_tot[s]"])
            runtime_distribution_factoring = np.array(prog_curr_distribution_factoring["  T_tot[s]"])
            runtime_distribution_random    = np.array(prog_curr_distribution_random["  T_tot[s]"])



            # between Default-Factoring
            DF_pass_or_fail, A_DF, A_DF_cat = perform_mann_whitney_and_get_effect_size(runtime_distribution_default, runtime_distribution_factoring, "DF", prog_index)

            if DF_pass_or_fail == "pass" or A_DF_cat == "(N)":
                number_equal_DF += 1
                number_equal_DF_all += 1

            else:
                pass

            if DF_pass_or_fail == "fail":

                if A_DF > 0.5:
                    number_fail_DF += 1
                else:
                    pass
                if A_DF < 0.5:
                    number_fail_DF_r += 1
                else:
                    pass


            # between Default-Random
            DR_pass_or_fail, A_DR, A_DR_cat = perform_mann_whitney_and_get_effect_size(runtime_distribution_default, runtime_distribution_random, "DR", prog_index)

            if DR_pass_or_fail == "pass" or A_DR_cat == "(N)":
                number_equal_DR += 1
                number_equal_DR_all += 1
            else:
                pass

            if DR_pass_or_fail == "fail":

                if A_DR > 0.5:
                    number_fail_DR += 1
                else:
                    pass
                if A_DR < 0.5:
                    number_fail_DR_r += 1
                else:
                    pass
            # between Factoring-Random
            FR_pass_or_fail, A_FR, A_FR_cat = perform_mann_whitney_and_get_effect_size(runtime_distribution_factoring, runtime_distribution_random, "FR", prog_index)

            if FR_pass_or_fail == "pass" or A_FR_cat == "(N)":
                number_equal_FR += 1
                number_equal_FR_all += 1
            else:
                pass


            if FR_pass_or_fail == "fail":

                if A_FR > 0.5:
                    number_fail_FR += 1
                else:
                    pass
                if A_FR < 0.5:
                    number_fail_FR_r += 1
                else:
                    pass

            if DF_pass_or_fail == "fail" or DR_pass_or_fail == "fail" or FR_pass_or_fail == "fail":

                perform_kruskal_wallis(runtime_distribution_default, runtime_distribution_factoring, runtime_distribution_random, prog_index, "fail")

                # assign values to table dataframe
                df_table.at["\\textbf{ID}", prog_index] = b_add + f"{prog_index}" + b_end

                # assign values to table dataframe
                df_table.at["\\textbf{\\#Qubits}", prog_index] = b_add + f"{number_of_qubits}" + b_end

                # assign depth
                df_table.at["\\textbf{Depth}", prog_index] = b_add + f"{depth}" + b_end

                # assign program category
                df_table.at["\\textbf{Category}", prog_index] = b_add + f"{curr_program_category}" + b_end


                # assign average and std
                df_table.at["\\textbf{Default [s]}", prog_index] = b_add + f"{runtime_rate_average_default:.1f} $\\pm$ {runtime_rate_std_default:.1f}" + b_end
                df_table.at["\\textbf{Factoring [s]}", prog_index] = b_add + f"{runtime_rate_average_factoring:.1f} $\\pm$ {runtime_rate_std_factoring:.1f}" + b_end
                df_table.at["\\textbf{Random [s]}", prog_index] = b_add + f"{runtime_rate_average_random:.1f} $\\pm$ {runtime_rate_std_random:.1f}" + b_end
            else:

                perform_kruskal_wallis(runtime_distribution_default, runtime_distribution_factoring, runtime_distribution_random, prog_index, "pass")

                # assign values to table dataframe
                df_table.at["\\textbf{ID}", prog_index] = f"{prog_index}"

                # assign values to table dataframe
                df_table.at["\\textbf{\\#Qubits}", prog_index] = f"{number_of_qubits}"

                # assign depth
                df_table.at["\\textbf{Depth}", prog_index] = f"{depth}"

                # assign program category
                df_table.at["\\textbf{Category}", prog_index] = f"{curr_program_category}"


                # assign average and std
                df_table.at["\\textbf{Default [s]}", prog_index] = f"{runtime_rate_average_default:.1f} $\\pm$ {runtime_rate_std_default:.1f}"
                df_table.at["\\textbf{Factoring [s]}", prog_index] = f"{runtime_rate_average_factoring:.1f} $\\pm$ {runtime_rate_std_factoring:.1f}"
                df_table.at["\\textbf{Random [s]}", prog_index] = f"{runtime_rate_average_random:.1f} $\\pm$ {runtime_rate_std_random:.1f}"

        # right to left
        df_pass_counts.loc[ curr_program_category, "\\%passed (NN) DF" ] = round( number_fail_DF/len(unique_progs)*100, round_to )
        df_pass_counts.loc[ curr_program_category, "\\%passed (NN) DR" ] = round( number_fail_DR/len(unique_progs)*100, round_to )
        df_pass_counts.loc[ curr_program_category, "\\%passed (NN) FR" ] = round( number_fail_FR/len(unique_progs)*100, round_to )

        df_pass_counts.loc[ curr_program_category, "\\%passed (NN) Counts DF" ] = number_fail_DF
        df_pass_counts.loc[ curr_program_category, "\\%passed (NN) Counts DR" ] = number_fail_DR
        df_pass_counts.loc[ curr_program_category, "\\%passed (NN) Counts FR" ] = number_fail_FR

        df_pass_counts.loc[ curr_program_category, "Total Counts" ] = len(unique_progs)

        # equal
        df_pass_counts.loc[ curr_program_category, "\\%equal DF" ] = round( number_equal_DF/len(unique_progs)*100, round_to )
        df_pass_counts.loc[ curr_program_category, "\\%equal DR" ] = round( number_equal_DR/len(unique_progs)*100, round_to )
        df_pass_counts.loc[ curr_program_category, "\\%equal FR" ] = round( number_equal_FR/len(unique_progs)*100, round_to )


        # left to right
        df_pass_counts_r.loc[ curr_program_category, "\\%passed (NN) DF" ] = round( number_fail_DF_r/len(unique_progs)*100, round_to )
        df_pass_counts_r.loc[ curr_program_category, "\\%passed (NN) DR" ] = round( number_fail_DR_r/len(unique_progs)*100, round_to )
        df_pass_counts_r.loc[ curr_program_category, "\\%passed (NN) FR" ] = round( number_fail_FR_r/len(unique_progs)*100, round_to )

        df_pass_counts_r.loc[ curr_program_category, "\\%passed (NN) Counts DF" ] = number_fail_DF_r
        df_pass_counts_r.loc[ curr_program_category, "\\%passed (NN) Counts DR" ] = number_fail_DR_r
        df_pass_counts_r.loc[ curr_program_category, "\\%passed (NN) Counts FR" ] = number_fail_FR_r

        df_pass_counts_r.loc[ curr_program_category, "Total Counts" ] = len(unique_progs)


        df_table_transposed = df_table.T

        df_latex = df_table_transposed.to_latex(index=False, escape=False)

        outfile.write(df_latex + "\n")

# right to left
all_tot = df_pass_counts.loc["aamp", "Total Counts"]   + df_pass_counts.loc["qwalk", "Total Counts"]   + df_pass_counts.loc["var", "Total Counts"] + df_pass_counts.loc["gs", "Total Counts"]
df_pass_counts.loc["all", "Total Counts"] = all_tot

all_DF  = df_pass_counts.loc["aamp", "\\%passed (NN) Counts DF"] + df_pass_counts.loc["qwalk", "\\%passed (NN) Counts DF"] + df_pass_counts.loc["var", "\\%passed (NN) Counts DF"] + df_pass_counts.loc["gs", "\\%passed (NN) Counts DF"]
all_DR  = df_pass_counts.loc["aamp", "\\%passed (NN) Counts DR"] + df_pass_counts.loc["qwalk", "\\%passed (NN) Counts DR"] + df_pass_counts.loc["var", "\\%passed (NN) Counts DR"] + df_pass_counts.loc["gs", "\\%passed (NN) Counts DR"]
all_FR  = df_pass_counts.loc["aamp", "\\%passed (NN) Counts FR"] + df_pass_counts.loc["qwalk", "\\%passed (NN) Counts FR"] + df_pass_counts.loc["var", "\\%passed (NN) Counts FR"] + df_pass_counts.loc["gs", "\\%passed (NN) Counts FR"]

df_pass_counts.loc["all", "\\%passed (NN) Counts DF"] = all_DF
df_pass_counts.loc["all", "\\%passed (NN) Counts DR"] = all_DR
df_pass_counts.loc["all", "\\%passed (NN) Counts FR"] = all_FR

df_pass_counts.loc["all", "\\%passed (NN) DF"] = round( all_DF/all_tot*100, round_to )
df_pass_counts.loc["all", "\\%equal DF"] = round( number_equal_DF_all/all_tot*100, round_to )

df_pass_counts.loc["all", "\\%passed (NN) DR"] = round( all_DR/all_tot*100, round_to )
df_pass_counts.loc["all", "\\%equal DR"] = round( number_equal_DR_all/all_tot*100, round_to )

df_pass_counts.loc["all", "\\%passed (NN) FR"] = round( all_FR/all_tot*100, round_to )
df_pass_counts.loc["all", "\\%equal FR"] = round( number_equal_FR_all/all_tot*100, round_to )



csv_filename = os.path.join( "tables", "Summary", "runtime_pass_counts.csv" )
df_pass_counts.to_csv( csv_filename )


# left to right
all_tot_r = df_pass_counts_r.loc["aamp", "Total Counts"]   + df_pass_counts_r.loc["qwalk", "Total Counts"]   + df_pass_counts_r.loc["var", "Total Counts"] + df_pass_counts_r.loc["gs", "Total Counts"]
df_pass_counts_r.loc["all", "Total Counts"] = all_tot_r

all_DF_r  = df_pass_counts_r.loc["aamp", "\\%passed (NN) Counts DF"] + df_pass_counts_r.loc["qwalk", "\\%passed (NN) Counts DF"] + df_pass_counts_r.loc["var", "\\%passed (NN) Counts DF"] + df_pass_counts_r.loc["gs", "\\%passed (NN) Counts DF"]
all_DR_r  = df_pass_counts_r.loc["aamp", "\\%passed (NN) Counts DR"] + df_pass_counts_r.loc["qwalk", "\\%passed (NN) Counts DR"] + df_pass_counts_r.loc["var", "\\%passed (NN) Counts DR"] + df_pass_counts_r.loc["gs", "\\%passed (NN) Counts DR"]
all_FR_r  = df_pass_counts_r.loc["aamp", "\\%passed (NN) Counts FR"] + df_pass_counts_r.loc["qwalk", "\\%passed (NN) Counts FR"] + df_pass_counts_r.loc["var", "\\%passed (NN) Counts FR"] + df_pass_counts_r.loc["gs", "\\%passed (NN) Counts FR"]

df_pass_counts_r.loc["all", "\\%passed (NN) Counts DF"] = all_DF_r
df_pass_counts_r.loc["all", "\\%passed (NN) Counts DR"] = all_DR_r
df_pass_counts_r.loc["all", "\\%passed (NN) Counts FR"] = all_FR_r

df_pass_counts_r.loc["all", "\\%passed (NN) DF"] = round( all_DF_r/all_tot_r*100, round_to )
df_pass_counts_r.loc["all", "\\%passed (NN) DR"] = round( all_DR_r/all_tot_r*100, round_to )
df_pass_counts_r.loc["all", "\\%passed (NN) FR"] = round( all_FR_r/all_tot_r*100, round_to )



csv_filename_r = os.path.join( "tables", "Summary", "runtime_pass_counts_r.csv" )
df_pass_counts_r.to_csv( csv_filename_r )
