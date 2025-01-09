import pandas as pd
import numpy as np

category = "gs"

#only_for_qubits_above = 15


def compute_above_below(approach):

    df_all = pd.read_csv("df_reduction_run_all_" + f"{category}" + "_" + f"{approach}" + ".txt")
    # Assuming df_all is your DataFrame and 'reduction factor' is a column in it

    #df_all = df_all[df_all["Qubits"] > only_for_qubits_above]

    performance_counts = df_all["Performance Type"].value_counts()

    # Print the counts
    print(approach)
    speedups  = performance_counts[0]
    slowdowns = performance_counts[1]
    tot = speedups + slowdowns
    print(performance_counts)
    print("speedups =", speedups/tot*100)
    print("slowdowns =", slowdowns/tot*100)



compute_above_below("factoring")
#compute_above_below("random")
