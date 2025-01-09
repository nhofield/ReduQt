# We reuse the configuration file handling of Quito
# for program specifications
# https://github.com/Simula-COMPLEX/quito

import sys
import os
import inspect
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])

if sys.platform.startswith('win32'):
    ROOT='\\'
elif sys.platform.startswith('linux'):
    ROOT='/'
elif sys.platform.startswith('darwin'):
    ROOT='/'



# ----------
# | Inputs |
# ---------
# command line argument to specify id within r2 repetitions (for parallelization)
input_arg_cl = int( sys.argv[1] )

# specify approach
approach = "Greedy"
# Inputs end ------------


from qiskit import (
    #IBMQ,
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    execute,
    Aer,
)
import numpy as np
from scipy import stats
import os

import configparser
import os.path
import importlib
import time
import warnings

import logging


# standard variables
K=200
M=20
BUDGET=0
C_LEVEL=0.01
S_TEST = 'one-sample Wilcoxon signed rank test'
T1=0
T2=0
START = 0

def check_unique(l):
    return len(l) == len(set(l))

def end_running():
    exit()

def dec2bin(n,bit):
    a = 1
    list = []
    while a > 0:
        a, b = divmod(n, 2)
        list.append(str(b))
        n = a
    s = ""
    for i in range(len(list) - 1, -1, -1):
        s += str(list[i])
    s = s.zfill(bit)
    return s

def input_group(valid_input):
    index = [] #unique input index
    index_flag = valid_input[0]
    index.append(0)
    for i in range(1,len(valid_input)):
        if valid_input[i] != index_flag:
            index.append(i)
            index_flag = valid_input[i]
    return index

def check_full_ps(valid_input, p):
    index = input_group(valid_input)
    p_sum = 0
    for i in range(len(index)):
        start = index[i]
        if i == len(index) - 1:
            end = len(valid_input)
        else:
            end = index[i+1]
        for j in range(start,end):
            p_sum += p[j]
        if (1 - p_sum) > 1E-10:
            print("Error: This is not a complete program specification.")
            end_running()
        else:
            p_sum = 0


def get_unique(l):
    unique = []
    for i in l:
        if i not in unique:
            unique.append(i)
    return unique

def get_all(bit):
    all = []
    for i in range(pow(2,bit)):
        i_bin = dec2bin(i, bit)
        all.append(i_bin)
    return all

def execute_quantum_program(inputID, outputID, num_qubit, i, module_name):
    """
    Takes a quantum program python function and creates
    a quantum register to run it with a given input.
    """
    q = QuantumRegister(num_qubit)
    c = ClassicalRegister(len(outputID))
    qc = QuantumCircuit(q, c)

    for j in range(len(inputID)):
        # print(i)
        if i[len(inputID) - 1 - j] == '1':
            # print(int(inputID[j]))
            qc.x(int(inputID[j]))


    module = importlib.import_module(module_name)
    run_method = getattr(module,"run")
    run_method(qc)

    result = execute(qc, Aer.get_backend('qasm_simulator'), shots=1).result().get_counts(qc)

    return result



def check_same(l,value):
    for i in range(len(l)):
        if l[i] != value:
            return False
    return True

def check_WOO(i, o , valid_inputs, valid_outputs):
    """
    Fails woo if it returns False
    """
    flag = False

    for k in range(len(valid_inputs)):
        if valid_inputs[k] == i and valid_outputs[k] == o:
            flag = True
    if flag == False:
        print('fail for woo')
        return False
    return True

def get_programs_list():
    """
    Function for generating a list of names of the study subjects
    """
    programs_list = []

    # var programs
    num_programs_var = 12
    lower_thresh_var = 2
    upper_thresh_var = 8
    id_var_range = 4
    ids_var = [1, 2, 2, 1, 3, 1, 1]

    for qubit_index in range(lower_thresh_var, upper_thresh_var + 1):
        for program_index in range( ids_var[qubit_index-lower_thresh_var] ):
            for range_index in range( id_var_range ):

                program_name = r"var_" + f"q{qubit_index}_{program_index}{range_index}.init"
                program_name_A = r"var_" + f"q{qubit_index}_{program_index}{range_index}_A.init"

                programs_list.append(program_name)
                programs_list.append(program_name_A)

    # aamp programs
    lower_thresh_aamp = 6
    upper_thresh_aamp = 9
    ids_number = 12

    for qubit_index in range(lower_thresh_aamp, upper_thresh_aamp + 1):
        for program_index in range( ids_number ):

            program_name = r"aamp_" + f"q{qubit_index}_{program_index}.init"
            program_name_A = r"aamp_" + f"q{qubit_index}_{program_index}_A.init"

            programs_list.append(program_name)
            programs_list.append(program_name_A)

    # qwalk programs
    lower_thresh_qwalk = 3
    upper_thresh_qwalk = 5
    ids_number = 13

    for qubit_index in range(lower_thresh_qwalk, upper_thresh_qwalk + 1):
        for program_index in range( ids_number ):

            program_name = r"qwalk_" + f"q{qubit_index}_{program_index}.init"
            program_name_A = r"qwalk_" + f"q{qubit_index}_{program_index}_A.init"

            programs_list.append(program_name)
            programs_list.append(program_name_A)

    # gs programs
    lower_thresh_gs = 3
    upper_thresh_gs = 16

    for qubit_index in range(lower_thresh_gs, upper_thresh_gs + 1):

        program_name = r"gs_" + f"q{qubit_index}.init"
        program_name_A = r"gs_" + f"q{qubit_index}_A.init"

        programs_list.append(program_name)
        programs_list.append(program_name_A)


    programs_list_mutants = []

    for program_index in range(0, len(programs_list), 2 ):
        file_default = programs_list[program_index][:-5]
        file_A       = programs_list[program_index + 1][:-5]
        num_qubits_file = int( file_default.split("_")[1][1:] )

        for qubit_index in range( 3 ):
            for generator_index in range(2):
                ext = f"_M{generator_index + 1}_{qubit_index}"
                mutant_default = file_default + ext + ".init"
                mutant_A       = file_A + ext + ".init"

                if mutant_default[:9] == "aamp_q6_6":
                    pass
                elif mutant_default[:9] == "aamp_q7_5":
                    pass
                else:
                    programs_list_mutants.append( mutant_default )
                    programs_list_mutants.append( mutant_A )

        for angle_index in range(9):
            ext = f"_M{3}_{angle_index}"
            mutant_default = file_default + ext + ".init"
            mutant_A       = file_A + ext + ".init"
            if mutant_default[:9] == "aamp_q6_6":
                pass
            elif mutant_default[:9] == "aamp_q7_5":
                pass
            else:
                programs_list_mutants.append( mutant_default )
                programs_list_mutants.append( mutant_A )

    return programs_list_mutants




def chisquare_scipy( counts, p ):
    """
    Performs the chisquare test between the theoretical distribution p
    and the sample distribution counts
    """
    pvalue = []

    for i in range(M):

        if np.isnan(counts[:, i]).any() == True:
            pvalue.append(-1)

        else:

            for i in range( len(counts[0]) ):

                pvalue.append( stats.chisquare( counts[:,i], np.array( p )*N_E ).pvalue )

    return pvalue


def pdo_test_assessment(inputs, outputs, pvalue):
    """
    The pdo oracle assesses the pvalue resulting from the
    chisquare test according to the confidence level C_LEVEL
    """

    for i in range(len(pvalue)):
        if pvalue[i] == -1:
            raise ValueError("pval = -1, not enough inputs for test.")

        elif pvalue[i] < C_LEVEL:

            # fail for pdo
            return "pdo"

        elif pvalue[i] >= C_LEVEL:
            # no fail for pdo
            return "pass"

def sample_quantum_program(inputID, valid_input, valid_output, num_qubit, outputID, p, module_name, slurm_job_id):
    """
    Samples the quantum program within experiment2:
    -----------
    | Inputs: |
    -----------
    inputID, quantum program input
    outputID, length of classic register for measurements
    valid_input, inputs from program specification
    valid_output, outputs from program specification
    num_qubit, qubit count for the given program
    p, theoretical distribution
    module_name, quantum program name
    slurm_job_id, for parallelization
    ------------
    | Outputs: |
    ------------
    module_name, same as described in the input
    outcome, test outcome pass or fail
    num_circuit_runs, number of circuit executions
    T_qc, test runtime
    """

    unique_inputs = get_unique(valid_input)

    M = 1
    counts = np.zeros((len(valid_input),M))
    fre = np.zeros((len(valid_input),M))
    count_cases = 0
    woo_flag = 1

    num_circuit_runs = 0

    # runtime with everything?
    T_qc = 0

    input_value = unique_inputs[0]

    for execution_id in range(N_E):

        count_cases += 1
        start = time.time()
        result = execute_quantum_program(inputID, outputID, num_qubit, input_value, module_name)
        num_circuit_runs += 1
        end = time.time()
        T_qc += end-start
        o = list(result)[0]

        if check_WOO(input_value, o, valid_input, valid_output) == False:

            # woo fails
            outcome = "woo"

            return module_name, outcome, num_circuit_runs, T_qc

        for mark in range( len( valid_input ) ):
            if valid_input[mark] == input_value and valid_output[mark] == o:
                counts[mark][0] += 1

    #getting frequencies
    sum = np.zeros(M)

    input_index = input_group(valid_input)

    for i in range(len(input_index)):
        start = input_index[i]
        if i == len(input_index) - 1:
            end = len(valid_input)
        else:
            end = input_index[i+1]
        # print("start=" + str(start))
        # print("end=" + str(end))
        for j in range(start, end):
            sum += counts[j]
        for j in range(start, end):
            fre[j] = counts[j]/sum
        # print(sum)
        sum = np.zeros(M)

    pvalue = chisquare_scipy(counts, p)

    # Now, either pdo fails or the test passes
    # get test assessment for pdo oracle
    outcome = pdo_test_assessment(valid_input, valid_output, pvalue)

    return module_name, outcome, num_circuit_runs, T_qc


def check_configuration_file(config):
    if config.has_section('program') == False:
        print("Error: Quito cannot find section 'program' in this configuration file.")
        end_running()
    else:
        if config.has_option('program', 'root') == False:
            print("Error: Quito cannot find the root of the program.")
            end_running()
        if config.has_option('program', 'num_qubit') == False:
            print("Error: Quito cannot find the number of qubits of the program.")
            end_running()
        if config.has_option('program', 'inputID') == False:
            print("Error: Quito cannot find the input IDs of the program.")
            end_running()
        if config.has_option('program', 'outputID') == False:
            print("Error: Quito cannot find the output IDs of the program.")
            end_running()
    if config.has_section('program_specification_category') == False:
        print("Error: Quito cannot find section 'program_specification_category' in this configuration file.")
        end_running()
    else:
        if config.has_option('program_specification_category', 'ps_category') == False:
            print("Error: Quito cannot find the category of the program specification.")
            end_running()
    if config.has_section('quito_configuration') == False:
        print("Error: Quito cannot find section 'quito_configuration' in this configuration file.")
        end_running()
    else:
        if config.has_option('quito_configuration', 'coverage_criterion') == False:
            print("Error: Quito cannot find the coverage criterion you choose.")
            end_running()

    ps_category = config.get('program_specification_category', 'ps_category')
    if ps_category == 'full' or ps_category == 'partial':
        if config.has_section('program_specification') == False:
            print("Error: Quito cannot find the program specification.")
            end_running()
    return ps_category

def check_inputID_outputID(num_qubit, inputID, outputID):
    if check_unique(inputID) == False:
        print("Wrong input IDs")
        end_running()
    if check_unique(outputID) == False:
        print("Wrong output IDs")
        end_running()
    inputID.sort()
    outputID.sort()

    if int(inputID[-1]) > num_qubit - 1:
        print("Wrong input IDs")
        end_running()
    if int(inputID[-1]) > num_qubit - 1:
        print("Wrong output IDs")
        end_running()

    return inputID, outputID

def check_bin(bin_str, n):
    if len(bin_str) != n:
        print("Error: The format of the program specification is wrong.")
        end_running()
   # print("check bin: "+str(bin_str))
    for i in range(len(bin_str)):
        if bin_str[i] != '0' and bin_str[i] != '1':
            print("Error: The format of the program specification is wrong.")
            end_running()


def execute_experiment2(root_con, slurm_job_id):

    if os.path.isfile(root_con) == True:
        config = configparser.ConfigParser(allow_no_value=True)
        config.read(root_con, encoding='utf-8')
    else:
        print("Error: Cannot find the configuration file.")
        end_running()

    ps_category = check_configuration_file(config)
    if ps_category != 'no' and ps_category != 'partial' and ps_category != 'full':
        print("Error: The format of program specification category is wrong.")
        end_running()

    #get quantum program
    root = root_con.split(".")[0] + ".py"

    if os.path.isfile(root) != True:
        print("Error: Quito cannot find the quantum program file.")
        end_running()

    root_list = root.split(ROOT)
    program_file = root_list[len(root_list)-1]
    program_folder = root_list[:len(root_list)-1]
    program_folder = ROOT.join(str(i) for i in program_folder)
    sys.path.append(program_folder)
    # print(program_file.split('.')[0])
    module_name = program_file.split('.')[0]
    print("module_name", module_name)

    #get inputID, outputID and numner of qubits
    inputID_o = config.get('program','inputID').split(',')
    outputID_o = config.get('program','outputID').split(',')
    num_qubit = int(config.get('program','num_qubit'))
    inputID, outputID = check_inputID_outputID(num_qubit, inputID_o, outputID_o)

    #ps_category = config.get('program_specification_category','ps_category')
    coverage_criterion = config.get('quito_configuration', 'coverage_criterion')
    print(coverage_criterion)
    if coverage_criterion != 'IC' and coverage_criterion != 'OC' and coverage_criterion != 'IOC':
        print("Error: The format of coverage criterion is not right.")
        end_running()

    if config.get('quito_configuration', 'K') != None:
        # sample size N_E
        global N_E
        N_E = int(config.get('quito_configuration', 'K'))
    if config.get('quito_configuration', 'M') != None:
        global  M
        M = int(config.get('quito_configuration', 'M'))
    if config.get('quito_configuration', 'confidence_level') != None:
        global C_LEVEL
        C_LEVEL = float(config.get('quito_configuration', 'confidence_level'))
    if config.get('quito_configuration', 'statistical_test') != None:
        global  S_TEST
        S_TEST = config.get('quito_configuration', 'statistical_test')

    global BUDGET

    # get reduqt_parameters
    global mutation
    mutation = config.get("reduqt_parameters", "Mutation")
    global original
    original = config.get("reduqt_parameters", "original")
    global depth
    depth = config.get("reduqt_parameters", "depth")



    if ps_category == 'no':
        if coverage_criterion == 'IC':
            input_coverage_no(inputID, outputID, num_qubit, module_name)
        else:
            BUDGET = pow(2, len(inputID)) * 10  # default
            if config.get('quito_configuration', 'BUDGET') != None:
                BUDGET = int(config.get('quito_configuration', 'BUDGET'))

            # check budget
            if BUDGET < pow(2, len(inputID)):
                print("Error: Budget is smaller than the number of inputs.")
                end_running()
            if coverage_criterion == 'OC':
                output_coverage_no(inputID, outputID, num_qubit, module_name)
            elif coverage_criterion == 'IOC':
                input_output_coverage_no(inputID, outputID, num_qubit, module_name)


    else:
        #get PS
        valid_input = []
        valid_output = []
        p = []
        ps = config.items('program_specification')

        if check_unique(ps) == False:
            print("Program specifications not unique")
            end_running()

        #sort PS according to input and output
        ps.sort(key=lambda x:x[0])

        for i in range(len(ps)):
            valid_input_item = ps[i][0][:len(inputID)]
            valid_output_item = ps[i][0][len(inputID)+1:]
            check_bin(valid_input_item,len(inputID))
            check_bin(valid_output_item,len(outputID))
            valid_input.append(valid_input_item)
            valid_output.append(valid_output_item)
            p.append(float(ps[i][1]))



        if ps_category == 'full':
            check_full_ps(valid_input, p)
        # print(outputID)

        if ps_category == 'full':
            if coverage_criterion == 'IC':
                module_name, outcome, num_circuit_runs, T_qc= sample_quantum_program(inputID, valid_input, valid_output, num_qubit, outputID, p, module_name, slurm_job_id)
            elif coverage_criterion == 'OC':
                module_name, outcome, num_circuit_runs, T_qc= output_coverage(inputID, valid_input, valid_output, num_qubit, outputID, p, module_name)
            elif coverage_criterion == 'IOC':
                input_output_coverage(inputID, valid_input, valid_output, num_qubit, outputID, p, module_name)


        return module_name, outcome, num_circuit_runs, T_qc, mutation, original, depth



if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    # path handling
    current_file_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(current_file_directory)

    # result folder name
    result_folder_name = r"results_experiment2"

    # Check if the result folder exists
    if not os.path.exists(result_folder_name):
        # Create the folder
        os.makedirs(result_folder_name)
        print(f"Folder '{result_folder_name}' created.")

    # where to output results
    resultFolder = os.path.join(current_file_directory, result_folder_name)


    if approach == "Greedy":
        mutant_dir_approach = "Mutants"
    elif approach == "Random":
        mutant_dir_approach = "Mutants_random"
    else:
        pass

    # where to find mutants and config folders
    mutant_directory = os.path.join(current_file_directory, mutant_dir_approach, f"Mutants_repetition_{input_arg_cl}")

    # fetch programs
    programs_list = get_programs_list()

    num_experiments = len(programs_list)

    experiment_results_file = open(resultFolder + f'/results_run_{input_arg_cl}.txt','w')

    experiment_results_file.write("Program, Mutation, Type,  T_tot[s], T_qc[s], Outcome, N_E, M, Nc, depth\n")

    experiment_results_file.close()


    for i in range(num_experiments):

        filename_list = programs_list[i].split("_")

        if filename_list[0] == "gs":
            sub_dir_prog = f"{filename_list[0]}_{filename_list[1]}"

        else:
            sub_dir_prog = f"{filename_list[0]}_{filename_list[1]}_{filename_list[2]}"


        # path to current program and its configuration file
        root_config_file = os.path.join(mutant_directory, sub_dir_prog,  programs_list[i])
        print("root_config_file=", root_config_file)

        print("current program =", programs_list[i])

        START = time.time()
        module_name, outcome, num_circuit_runs, T_qc, mutation, original, depth = execute_experiment2(root_config_file, input_arg_cl)
        end = time.time()
        T_tot  = end - START

        experiment_results_file = open(resultFolder + f'/results_run_{input_arg_cl}.txt','a')
        experiment_results_file.write("{0},{1},{2},{3:.2f},{4:.2f},{5},{6},{7},{8},{9}\n" .format(module_name, mutation, original, T_tot, T_qc, outcome, N_E, M, num_circuit_runs, depth))
        experiment_results_file.close()

        print('\n')
        print("The total run time is "+"{:.2f}".format(T1)+"s.")
        print('The execution time of the quantum program is ' + "{:.2f}".format(T2) + "s.")
