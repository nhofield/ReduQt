import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr, linregress
import matplotlib.colors  # Import this to correctly use to_rgb
import os

pd.set_option('display.max_rows', None)  # This will display all rows
pd.set_option('display.max_columns', None)  # This will display all columns
pd.set_option('display.width', None)  # This will try to fit the display width to avoid wrapping
pd.set_option('display.max_colwidth', None)  # This will display full content of each column



def give_rounded(val):

    if val <= 0.001:
        return "{:.1e}".format(val)

    else:
        return f"{np.round(val, 5)}"

#program_category_list = ["var"]
program_category_list = ["aamp", "qwalk", "var", "gs"]
program_category_list_caption = ["Grov", "Qwalk", "Var", "Gs"]

current_method1 = "greedy"
current_method2 = "random"

# fontsize for figures
fsize = 16

# Define the directory
figures_directory = Path("figures")

# Check if the directory exists, if not, create it
figures_directory.mkdir(exist_ok=True)

table_df = pd.DataFrame(index = ["PCC", "p-value", "Slope"], columns = ["aamp F", "aamp R", "qwalk F", "qwalk R", "var F", "var R", "gs F", "gs R" ] )

current_directory = os.getcwd()
grandparent_directory = os.path.dirname(os.path.dirname(current_directory))
dir_to_postprocessing = os.path.join(grandparent_directory, "results", "postProcessing")


for program_category_index in range( len(program_category_list) ):

    program_category = program_category_list[program_category_index]
    program_category_corr = program_category_list_caption[program_category_index]

    filename_reduction_all1 = os.path.join( dir_to_postprocessing, f"df_reduction_run_all_{program_category}_{current_method1}.txt" )
    filename_reduction_all2 = os.path.join( dir_to_postprocessing, f"df_reduction_run_all_{program_category}_{current_method2}.txt" )

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



    plt.figure( figsize=(8, 6) )

    #x_val = " depth"
    #x_val = "reduction rate"
    #x_val = "Program Name Index"
    x_val = "Qubits"


    # To plot barplots
    speedups_df = df_combined[df_combined["Performance Type"] == "Speedup"]


    # Updated palettes for grouping speedups with orange and slowdowns with blue
    palette_speedups = {
        'Greedy': '#fdae61',  # Darker Orange for Factoring in speedups
        'Random': '#fee08b'      # Lighter Orange for Random in speedups
    }

    palette_slowdowns = {
        'Greedy': '#2c7bb6',  # Darker Blue for Factoring in slowdowns
        'Random': '#abd9e9'      # Lighter Blue for Random in slowdowns
    }


    #print( speedups_df[['Qubits', 'reduction factor']] )
    print("Unique-------------------")

    print(speedups_df["Qubits"].unique())
    print("Unique-------------------")

    sns.boxplot(
        x=x_val,
        y="reduction factor",
        hue="Approach",        # Differentiates line color by Approach
        data=df_combined,        # Your DataFrame
        palette=palette_slowdowns,         # Apply your custom palette
        saturation = 0.8,

        )



    slowdowns_df = df_combined[df_combined["Performance Type"] == "Slowdown"]

    """
    sns.boxplot(
        x=x_val,
        y="reduction factor",
        hue="Approach",        # Differentiates line color by Approach
        data=slowdowns_df,        # Your DataFrame
        palette=palette_slowdowns,         # Apply your custom palette
        saturation = 0.8,
         )
    """

    # for filename
    plot_type = "boxplot"

    if x_val == " depth":
        x_label = "Depth"

    if x_val == "Program Name Index":
        x_label = "Program Number"

    if x_val == "reduction rate":
        x_label = r"Reduction Rate"

    if x_val == "Qubits":
        x_label = "#Qubits"


    plt.title(f"Category = {program_category_corr}", fontsize=fsize, fontweight='bold')

    plt.grid(True)
    plt.xlabel(x_label, fontsize=fsize, fontweight='bold')
    if log_scale == "yes":
        plt.ylabel(r"Runtime Improvement over Default ($log_2$)", fontsize=fsize, fontweight='bold')
    else:
        plt.ylabel(r"Runtime Improvement over Default", fontsize=fsize, fontweight='bold')

    plt.tick_params(axis='both', which='major', labelsize=fsize-3)  # Adjust the font size for both axes
    plt.tick_params(axis='both', which='minor', labelsize=fsize-3)  # Optional: if you have minor ticks

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


    plt.legend(fontsize=fsize-3)
    # Set background color and grid
    #plt.gcf().set_facecolor('#f5f5f5')  # Light gray background for the figure
    #plt.gca().set_facecolor('#f5f5f5')  # Light gray background for the axes
    #plt.grid(color='white', linestyle='-', linewidth=0.7)  # White grid for contrast
    #plt.yscale("symlog", base = 2, linthresh = 1)
    # After creating your plot, save it to the 'figures' directory
    fig_filename = f"plot_rq2_{plot_type}_{program_category}.pdf"  # or .pdf, .svg, etc., depending on your preferred format
    fig_filepath = figures_directory / fig_filename  # Constructs the full path

    # Save the figure
    plt.savefig(fig_filepath, format='pdf')

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

    print("-----Correlation Tests-------")


    # create table df

    corr_x_val = x_val
    #corr_x_val = " depth"

    df_factored[speedup_column=="Speedup"]


    speedups_factored = df_factored[speedup_column=="Speedup"].sort_values(by=corr_x_val, ascending=True)
    speedups_factored['reduction factor'].replace([np.inf, -np.inf], -3.855122, inplace=True)

    slowdowns_factored = df_factored[speedup_column=="Slowdown"].sort_values(by=corr_x_val, ascending=True)

    speedups_random = df_random[speedup_column=="Speedup"].sort_values(by=corr_x_val, ascending=True)

    slowdowns_random = df_random[speedup_column=="Slowdown"].sort_values(by=corr_x_val, ascending=True)

    # correlation between reduction rate and depth
    pearson_correlation_depth, p_value_depth = spearmanr( speedups_factored[corr_x_val], speedups_factored['reduction factor'])
    slope, intercept, r_value, p_valLR, std_err = linregress( speedups_factored[corr_x_val], speedups_factored['reduction factor'])
    table_df.at["PCC", f"{program_category} F"] = give_rounded(pearson_correlation_depth)

    table_df.at["p-value", f"{program_category} F"] = give_rounded(p_value_depth)
    table_df.at["Slope", f"{program_category} F"] = f"{give_rounded(slope)} $\pm$ {give_rounded(std_err)}"

    print("Median Performance Factor Speedups Factored =", speedups_factored["reduction factor"].median())
    print("Median Depth Speedups Factored =", speedups_factored[corr_x_val].median())
    print("Pearson correlation coefficient Speedups Factored:", pearson_correlation_depth, p_value_depth)
    print(f"Slope = {slope} pm {std_err}")

    pearson_correlation_depth, p_value_depth = spearmanr( slowdowns_factored[corr_x_val], slowdowns_factored['reduction factor'])
    slope, intercept, r_value, p_valLR, std_err = linregress( slowdowns_factored[corr_x_val], slowdowns_factored['reduction factor'])


    print("Median Performance Factor Slowdowns Factored =", slowdowns_factored["reduction factor"].median())
    print("Median Depth Slowdowns Factored =", slowdowns_factored[corr_x_val].median())

    print("Pearson correlation coefficient Slowdowns Factored:", pearson_correlation_depth, p_value_depth)
    print(f"Slope = {slope} pm {std_err}")

    pearson_correlation_depth, p_value_depth = spearmanr(speedups_random[corr_x_val], speedups_random['reduction factor'])
    slope, intercept, r_value, p_valLR, std_err = linregress(speedups_random[corr_x_val], speedups_random['reduction factor'])
    table_df.at["PCC", f"{program_category} R"] = give_rounded(pearson_correlation_depth)
    table_df.at["p-value", f"{program_category} R"] = give_rounded(p_value_depth)
    table_df.at["Slope", f"{program_category} R"] = f"{give_rounded(slope)} $\pm$ {give_rounded(std_err)}"

    print("Median Performance Factor Speedups Random =", speedups_random["reduction factor"].median())
    print("Median Depth Speedups Random =", speedups_random[corr_x_val].median())
    print("Pearson correlation coefficient Speedups Random:", pearson_correlation_depth, p_value_depth)
    print(f"Slope = {slope} pm {std_err}")

    pearson_correlation_depth, p_value_depth = spearmanr(slowdowns_random[corr_x_val], slowdowns_random['reduction factor'])
    slope, intercept, r_value, p_valLR, std_err = linregress(slowdowns_random[corr_x_val], slowdowns_random['reduction factor'])

    print("Median Performance Factor Slowdowns Random =", slowdowns_random["reduction factor"].median())
    print("Median Depth Slowdowns Random =", slowdowns_random[corr_x_val].median())
    print("Pearson correlation coefficient Slowdonws Random:", pearson_correlation_depth, p_value_depth)
    print(f"Slope = {slope} pm {std_err}")

filename_outfile = os.path.join( "tables", "correlation_table.txt" )

outfile = open( filename_outfile, "w" )
df_latex = table_df.to_latex(index=True, escape=False)

#df_latex = df_latex.replace('\\$', '$')

#df_latex = df_latex.replace('\\textbackslash pm', '\\pm')
#df_latex = df_latex.replace('\\textbackslash pm', '\\pm')

outfile.write(df_latex + "\n")
outfile.close()
