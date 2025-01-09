import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr, linregress
import matplotlib.colors  # Import this to correctly use to_rgb
import os




def give_rounded(val):

    if val <= 0.001:
        return "{:.1e}".format(val)

    else:
        return f"{np.round(val, 5)}"

#program_category_list = ["var"]
program_category_list = ["aamp", "qwalk", "var", "gs"]

current_method1 = "greedy"
current_method2 = "random"

# fontsize for figures
fsize = 16

# Define the directory
figures_directory = Path("figures")

# Check if the directory exists, if not, create it
figures_directory.mkdir(exist_ok=True)

table_df = pd.DataFrame(index = ["PCC", "p-value", "Slope"], columns = ["aamp F", "aamp R", "qwalk F", "qwalk R", "var F", "var R", "gs F", "gs R" ] )

df_speedups_combined = pd.DataFrame()
df_slowdowns_combined = pd.DataFrame()

df_combined_all = pd.DataFrame()

current_directory = os.getcwd()
grandparent_directory = os.path.dirname(os.path.dirname(current_directory))
dir_to_results = os.path.join(grandparent_directory, "results", "postProcessing")


for program_category_index in range( len(program_category_list) ):

    program_category = program_category_list[program_category_index]

    filename_reduction_all1 = os.path.join( dir_to_results, f"df_reduction_run_all_{program_category}_{current_method1}.txt" )
    filename_reduction_all2 = os.path.join( dir_to_results, f"df_reduction_run_all_{program_category}_{current_method2}.txt" )

    df_factored = pd.read_csv( filename_reduction_all1 )
    df_random   = pd.read_csv( filename_reduction_all2 )




    df_factored['reduction rate'] = pd.to_numeric(df_factored['reduction rate'])
    df_random['reduction rate'] = pd.to_numeric(df_random['reduction rate'])

    df_factored['reduction factor'] = pd.to_numeric(df_factored['reduction factor'])
    df_random['reduction factor'] = pd.to_numeric(df_random['reduction factor'])

    df_factored['Qubits'] = pd.to_numeric(df_factored['Qubits'])
    df_random['Qubits'] = pd.to_numeric(df_random['Qubits'])

    log_scale = "no"


    if log_scale == "yes":

        for val in df_factored.index:

            if df_factored.loc[val, "reduction factor"] < 0:
                ...
                df_factored.loc[val, "reduction factor"] = - np.log2( - df_factored.loc[val, "reduction factor"] )

            elif df_factored.loc[val, "reduction factor"] > 0:
                ...
                df_factored.loc[val, "reduction factor"] = np.log2( df_factored.loc[val, "reduction factor"] )

            else:
                pass

        for val in df_random.index:

            if df_random.loc[val, "reduction factor"] < 0:
                ...
                df_random.loc[val, "reduction factor"] = - np.log2( - df_random.loc[val, "reduction factor"] )

            elif df_random.loc[val, "reduction factor"] > 0:
                ...
                df_random.loc[val, "reduction factor"] = np.log2( df_random.loc[val, "reduction factor"] )

            else:
                pass




    plt.figure()

    #sns.barplot( x="Program Name Index", y="reduction factor", hue="Performance Type",  data=df_factored )
    #sns.barplot( x="Program Name Index", y="reduction factor", hue="Performance Type",  data=df_random )

    # Plot the first lineplot
    #sns.lineplot(x="Program Name Index", y="reduction factor", hue="Performance Type", data=df_factored)

    # Plot the second lineplot
    #sns.lineplot(x="Program Name Index", y="reduction factor", hue="Performance Type", data=df_random)

    df_factored['Approach'] = 'Greedy'
    df_random['Approach']   = 'Random'


    df_combined = pd.concat([df_factored, df_random])

    df_combined = df_combined.reset_index(drop=True)

    df_combined_all = pd.concat( [df_combined_all, df_combined] )

    # To plot barplots

    speedups_df = df_combined[df_combined["Performance Type"] == "Speedup"]

    slowdowns_df = df_combined[df_combined["Performance Type"] == "Slowdown"]

    df_speedups_combined = pd.concat( [df_speedups_combined, speedups_df] )
    df_slowdowns_combined = pd.concat( [df_slowdowns_combined, slowdowns_df] )




plt.figure( figsize=(8, 6) )

#x_val = " depth"
#x_val = "reduction rate"
#x_val = "Program Name Index"
x_val = "category"


# Updated palettes for grouping speedups with orange and slowdowns with blue
palette_speedups = {
    'Greedy': '#fdae61',  # Darker Orange for Factoring in speedups
    'Random': '#fee08b'      # Lighter Orange for Random in speedups
}

palette_slowdowns = {
    'Greedy': '#2c7bb6',  # Darker Blue for Factoring in slowdowns
    'Random': '#abd9e9'      # Lighter Blue for Random in slowdowns
}

sns.boxplot(
    x=x_val,
    y="reduction factor",
    hue="Approach",        # Differentiates line color by Approach
    data=df_combined_all,        # Your DataFrame
    palette=palette_slowdowns,         # Apply your custom palette
    saturation = 0.8,
    )


# for filename
plot_type = "boxplot"

if x_val == " depth":
    x_label = "Depth"

if x_val == "Program Name Index":
    x_label = "Program Number"

if x_val == "reduction rate":
    x_label = r"$log_2$ Reduction Rate"
if x_val == "Qubits":
    x_label = "#Qubits"
if x_val == "category":
    x_label = "Program Category"

#plt.title(f"Program Category = {program_category}", fontsize=fsize, fontweight='bold')

plt.grid(True)
plt.xlabel(x_label, fontsize=fsize, fontweight='bold')

#-----------------------------
# make log y ticks
if log_scale == "yes":
    ax = plt.gca()

    # Define the range of exponents you want to label
    # This should cover the range of log2-transformed values you have
    exponents = np.arange(np.floor(ax.get_ylim()[0]), np.ceil(ax.get_ylim()[1]) + 1)

    # Generate the tick labels based on these exponents
    # For positive exponents, it will be '2^x'
    # For negative exponents, it will be '-2^(-x)'
    # For zero, it will just be '1'
    tick_labels = [f"$-2^{{{-int(exp)}}}$" if exp < 0 else ("$1$" if exp == 0 else f"$2^{{{int(exp)}}}$") for exp in exponents]

    # Set the new y-tick labels
    ax.set_yticks(exponents)
    ax.set_yticklabels(tick_labels)

    # Redraw the plot to show the new labels
    plt.draw()
#-----------------------------
# make log y ticks


if log_scale == "yes":
    plt.ylabel(r"Factor Improvement over Default ($log_2$)", fontsize=fsize, fontweight='bold')
else:
    plt.ylabel(r"Factor Improvement over Default", fontsize=fsize, fontweight='bold')


plt.tick_params(axis='both', which='major', labelsize=fsize-3)  # Adjust the font size for both axes
plt.tick_params(axis='both', which='minor', labelsize=fsize-3)  # Optional: if you have minor ticks
#plt.yscale("symlog", base = 2, linthresh = 1)


plt.legend(fontsize=fsize-3)
# Set background color and grid
#plt.gcf().set_facecolor('#f5f5f5')  # Light gray background for the figure
#plt.gca().set_facecolor('#f5f5f5')  # Light gray background for the axes
#plt.grid(color='white', linestyle='-', linewidth=0.7)  # White grid for contrast

# After creating your plot, save it to the 'figures' directory
fig_filename = f"plot_rq2_{plot_type}_all"  # or .pdf, .svg, etc., depending on your preferred format
fig_filepath = figures_directory / fig_filename  # Constructs the full path

# Save the figure
plt.savefig(fig_filepath)

plt.show()

print("-------------------------")
print(f"Category =", f"{program_category}")
print("-------------------")
print(f"factored method")
speedup_column = df_factored["Performance Type"]

number_of_speedups = len(df_factored[speedup_column=="Speedup"])
number_of_slowdowns = len(df_factored[speedup_column=="Slowdown"])

print("speedups=",number_of_speedups)
print("slowdowns=",number_of_slowdowns)
print("percent speedups=", 100*( number_of_speedups / (number_of_slowdowns + number_of_speedups) ))
print(number_of_speedups+number_of_slowdowns)
print("-------------------")

print("-------------------")
print(f"random method")
speedup_column = df_random["Performance Type"]

number_of_speedups = len(df_random[speedup_column=="Speedup"])
number_of_slowdowns = len(df_random[speedup_column=="Slowdown"])

print("speedups=",number_of_speedups)
print("slowdowns=",number_of_slowdowns)
print("percent speedups=", 100*( number_of_speedups / (number_of_slowdowns + number_of_speedups) ))
print(number_of_speedups+number_of_slowdowns)
print("-------------------")
