import qiskit as qk
import numpy as np
import math
import inspect
import time
import sys
import os
import pandas as pd
from pathlib import Path
from qiskit.quantum_info import Statevector

# ----------
# | Inputs |
# ---------

# number of repetitions of experiment 1
r1 = 100

# which approach to use
approach = "greedy"


# ----------------
#| Path Handling |
#----------------
# Calculate the path to the 'study_subjects' directory
root_dir = Path(__file__).resolve().parent

target_dir = root_dir / 'study_subjects'

# Add 'study_subjects' to the Python path
sys.path.append(str(target_dir))

# some path handling
working_directory = os.path.dirname(os.path.abspath(__file__)) + r"/"
results_experiment1_directory = os.path.join(working_directory, r"results_experiment1/")
study_subjects_directory = os.path.join(working_directory, r"study_subjects/")

# Go up two levels from the working directory
up_two_levels = os.path.dirname(os.path.dirname(working_directory))

# Navigate to the Mutants directory in Experiment2
mutants_directory = os.path.join(up_two_levels, "Experiment2", "Mutants") + r"/"

filenames = []

coverage_criterion = "IC"

# import var programs
num_programs_var = 11
lower_thresh_var = 2
upper_thresh_var = 8
ids_var = [1, 2, 2, 1, 3, 1, 1]

for qubit_index in range(lower_thresh_var, upper_thresh_var + 1):
    for program_index in range( ids_var[qubit_index-lower_thresh_var] ):
        for extra_id in range(4):

            prog_filename = r"import " + "var_" + f"q{qubit_index}_{program_index}{extra_id}_R"
            exec(prog_filename)

            program_name = prog_filename[7:]

            program = eval(program_name)

            filenames.append(program_name)

# import aamp (grov) programs
lower_thresh_aamp = 6
upper_thresh_aamp = 9
ids_number = 12

for qubit_index in range(lower_thresh_aamp, upper_thresh_aamp + 1):
    for program_index in range( ids_number ):

        prog_filename = r"import " + "aamp_" + f"q{qubit_index}_{program_index}_R"
        exec(prog_filename)

        program_name = prog_filename[7:]

        program = eval(program_name)

        filenames.append(program_name)

# import qwalk programs
lower_thresh_qwalk = 3
upper_thresh_qwalk = 5
ids_number = 13

for qubit_index in range(lower_thresh_qwalk, upper_thresh_qwalk + 1):
    for program_index in range( ids_number ):

        prog_filename = r"import " + "qwalk_" + f"q{qubit_index}_{program_index}_R"
        exec(prog_filename)

        program_name = prog_filename[7:]

        program = eval(program_name)

        filenames.append(program_name)

# import gs programs
lower_thresh_gs = 3
upper_thresh_gs = 16

for qubit_index in range(lower_thresh_gs, upper_thresh_gs + 1):

    prog_filename = r"import " + "gs_" + f"q{qubit_index}_R"
    exec(prog_filename)

    program_name = prog_filename[7:]

    program = eval(program_name)

    filenames.append(program_name)


print("----------")
print("Programs")
print("----------")
print(filenames)
print("----------")

print("Total Number of Programs =", len(filenames))


def test_circuit(circuit, number_of_qubits):
    """
    Utility function for printing the state vector
    and the corresponding states before measurement
    in decimal representation for comparison with
    https://oreilly-qc.github.io/# API

    --------------
    |  Inputs:  |
    --------------
    circuit: a run function in the quito format
    example for 3 qubits:
    def run(qc):
        qc.h(2)
        qc.x(1)
    number_of_qubits: a integer with number of qubits
    --------------
    |  Returns:  |
    --------------
    state_vector: complex vector from the Aer state_vector simulator
    states: a list of non-zero states in the state_vector in decimal representation
    """

    qc = qk.QuantumCircuit(number_of_qubits)
    circuit(qc)

    print(qc.draw())


    backend = qk.Aer.get_backend('statevector_simulator')
    job = qk.execute(qc, backend)
    result = job.result()
    state_vector = np.array( result.get_statevector(qc) )

    states = []
    for state_index in range(len(state_vector)):
        if state_vector[state_index] != 0:
            states.append(state_index)



    return state_vector, states

def decimal_to_binary(value, number_of_qubits):
    """
    -----------------
    | Takes inputs: |
    -----------------
    value - a decimal number
    number_of_qubits - the number of qubits
    -------------
    |  Returns: |
    -------------
    binary representation of value with leading zeros
    according to the number of qubits
    """
    output_string = np.binary_repr(value)

    # add leading zeros
    while len(output_string) < number_of_qubits:
        output_string = "0" + output_string

    return output_string

def rename_all_indices(dataframe):
    """
    Renames indices of dataframe
    """
    change_dict = {}

    for index in range( len(dataframe) ):

        change_dict[ dataframe.index[index] ] = index

    dataframe = dataframe.rename( change_dict )

    return dataframe

def fetch_and_write_median_basis_file(num_runs, approach):
    """
    To write approach_bases_median
    """
    if approach == "greedy":
        filename = r"results_experiment1/greedy_bases.txt"
        filename_out = r"results_experiment1/greedy_bases_median.txt"

    if approach == "random":
        filename = r"results_experiment1/random_bases.txt"
        filename_out = r"results_experiment1/random_bases_median.txt"

    df = pd.read_csv( filename, delimiter = ", ", engine="python")

    with open( filename_out, "w") as outfile:

        for program_index in range( len(filenames) ):
            prog_curr = filenames[program_index]
            df_prog   = df[ df["program"] == prog_curr ]
            median_index = int( round( num_runs/2 ) )
            df_index_sorted = df_prog['reduction'].argsort()
            sorted_df = df_prog.iloc[df_index_sorted]
            sorted_df = sorted_df.reset_index()

            target_index = int(round(num_runs / 2))  # Calculate the middle index


            row_with_median = sorted_df.iloc[target_index]

            outfile.write(f'{row_with_median["program"]}, {row_with_median["basis"]}, {row_with_median["reduction"]}\n')





def convert_state_vector_to_program_spec_format(state_vector, input_string, ):
    """
    -----------------
    | Takes inputs: |
    -----------------
    state_vector
    -------------
    |  Returns: |
    -------------
    string of a table with rows of the format
    input,output=probability
    """

    state_vector_length = len(state_vector)
    number_of_qubits    = int( np.log2(state_vector_length) )
    program_spec_string = ""
    prob = abs(state_vector)**2


    for state_index in range( state_vector_length ):

        output_string = decimal_to_binary(state_index, number_of_qubits)

        if prob[state_index] > 1e-15:

            program_spec_string += "".join( input_string[::-1].split(",") ) + ',' + output_string + '=' + str( prob[state_index] ) + '\n'

        else:

            pass

    return program_spec_string


def count_terms(state_vector):
    """
    Counts and returns the integer number of non-zero
    terms in the state vector superposition
    """
    # get probability array
    prob = abs( state_vector )**2
    # count number of terms greater than zero (machine precision)
    return np.sum( ( prob > 1e-14 )*1 )


class QuantumProgram:
    def __init__(self, num_qubits, circuit, filename, input_space="all_inputs"):
        """
        input: "0,0,1,0,x,y,...,0"
        basis: "i,i,h,h,i,h,...,i"
        circuit: quantum circuit without measurements
        """

        self.num_qubits = num_qubits
        self.circuit = circuit
        self.filename = filename
        self.basis = "i,"*self.num_qubits
        self.basis = self.basis[:-1]
        self.identity_basis = self.basis
        self.input = "0,"*self.num_qubits
        self.input = self.input[:-1]

        self.state_vector = 1
        self.depth = 1

        if input_space == "zero":
            self.generate_input_space()
            self.change_of_input_and_basis(self.input, self.basis)

        elif input_space == "all_inputs":
            self.input_space = self.generate_input_space_all()
            self.change_of_input_and_basis(self.input, self.basis)
        else:
            self.input_space = input_space
            self.input = self.input_space[0]
            self.change_of_input_and_basis(self.input, self.basis)

    def read_state_vector_for_ancilla(self, filename):
        with open( r"study_subjects/" + filename + ".init", "r") as infile:
            file_list = infile.read().split()
            state_vector_index = file_list.index("[state_vector]")
            program_specification_index = file_list.index("[program_specification]")

            state_vector_string = ""

            for sv_index in range(state_vector_index + 1, program_specification_index):

                state_vector_string += file_list[sv_index]

            state_vector_string = state_vector_string[1:-1]
            state_vector_string = state_vector_string.split(",")
            state_vector_list = [complex(state) for state in state_vector_string]
            state_vector = np.array( state_vector_list )

        return state_vector




    def write_ancilla_circuit(self, ancilla_program_filename, state_vector):

        with open( r"study_subjects/" + ancilla_program_filename + r".py", "w") as outfile:
            outfile.write("input_space = 'zero'\n")
            outfile.write("\n")
            outfile.write("def run(qc):\n")
            outfile.write(f"    qc.initialize({list(state_vector)})\n")

    def generate_input_space(self):
        """
        Generates a list of inputs for the quantum program
        For example, for a three-qubit program it will return
        ["0,0,0", "0,0,1", "0,1,0", "0,1,1", \
        "1,0,0", "1,0,1", "1,1,0", "1,1,1"]
        """
        self.input_space = []

        self.input_space.append( self.input )


    def generate_input_space_all(self):
        """
        Generates a list of all inputs for the quantum program
        For example, for a three-qubit program it will return
        ["0,0,0", "0,0,1", "0,1,0", "0,1,1", \
        "1,0,0", "1,0,1", "1,1,0", "1,1,1"]
        """
        input_space_all = []

        for value in range(2**self.num_qubits):

            # counter string temporary string for iteration
            counter_string = np.binary_repr(value)
            # input string
            input = ""

            # add leading zeros
            while len(counter_string) < self.num_qubits:
                counter_string = "0" + counter_string

            # add commas in input string
            for character in counter_string:
                input = input + character + ","

            # append input without last comma element
            input_space_all.append( input[:-1] )

        return input_space_all

    def change_of_input(self, new_input):
        """
        New input should be a string of binary values
        according to 'x,x,...,x' of length = number of qubits
        where x can hold the values 0 or 1
        """

        # update input
        self.input = new_input

    def change_of_input_and_basis(self, new_input, new_basis):
        """
        Inputs: new input - 'x,x,x,...,x'
        new_basis - 'x,x,x,...,x' where x = either i (identity) or h (Hadamard)
        Returns: nothing
        Modifies self.input, self.basis
        and updates self.state_vector according to new_input and new_basis
        """
        # update input
        self.input = new_input
        # update basis
        self.basis = new_basis

        quantum_reg = qk.QuantumRegister(self.num_qubits)
        classic_reg = qk.ClassicalRegister(self.num_qubits)
        quantum_circuit = qk.QuantumCircuit(quantum_reg, classic_reg)


        # list form of input
        new_input_list = new_input.split(",")
        # list form of basis
        new_basis_list = new_basis.split(",")

        input_default = "0,"*self.num_qubits
        input_default = input_default[:-1]

        if input != input_default:
            for input_index in range( self.num_qubits ):
                if new_input_list[input_index] != "0":
                    eval( f"quantum_circuit.x({input_index})" )

        self.circuit(quantum_circuit)

        identity_basis = "i,"*self.num_qubits

        if new_basis != identity_basis[:-1]:
            for gate_index in range( self.num_qubits ):
                if new_basis_list[gate_index] != "i":
                    gate = new_basis_list[gate_index]
                    eval( f"quantum_circuit.{gate}({gate_index})" )

        # run statevector simulator
        backend = qk.Aer.get_backend('statevector_simulator')

        job = qk.execute(quantum_circuit, backend)
        result = job.result()

        self.state_vector = np.array( result.get_statevector(quantum_circuit) )
        self.depth = quantum_circuit.depth()

    def __str__(self):
        program_representation = f"""Attributes of quantum program:
        Filename = {self.filename}.py
        Qubit Count = {self.num_qubits}
        State Vector = {self.state_vector}
        Depth = {self.depth}
        Input = {self.input}
        Measurement Basis = {self.basis}
        Input Space = {self.input_space}
        ---------------
        |   Circuit   |
        ---------------
        {inspect.getsource(self.circuit)}"""
        return program_representation


    def write_ps(self, K_factor = 100, M = 10, BUDGET = 6000, confidence_level = 0.05,  Mutation = 0, original = True, optimized_basis = None):
        """
        Writes a program specification
        K = 100
        M = 10
        BUDGET = 6000
        confidence_level = 0.05
        """
        # store current input for reverting later
        original_input = self.input

        if Mutation != 0:
            if original == True:
                file_extension = f"M{Mutation}"
            if original == False:
                file_extension = f"M{Mutation}" + r"_A"
        else:
            if original == True:
                file_extension = ""
            if original == False:
                file_extension = r"_A"

        if optimized_basis == None:
            optimized_basis = "identity"

        number_of_non_zero_terms = count_terms(self.state_vector)

        K = K_factor * number_of_non_zero_terms * len(self.input_space)

        with open( r"study_subjects/" + self.filename[:-2] + file_extension + r".init", "w") as current_init_file:
            current_init_file.write("[program]\n")
            current_init_file.write("root=" + mutants_directory + self.filename + file_extension + ".py\n")
            current_init_file.write("num_qubit=" + f"{self.num_qubits}\n")
            current_init_file.write("inputID=" + ",".join( [f"{qubit_number}" for qubit_number in range(self.num_qubits)] ) + "\n" )
            current_init_file.write("outputID=" + ",".join( [f"{qubit_number}" for qubit_number in range(self.num_qubits)] ) + "\n" )
            current_init_file.write("\n")
            current_init_file.write("[program_specification_category]\n")
            # program program_specification category fixed to full
            current_init_file.write("ps_category=full\n")
            current_init_file.write("\n")
            current_init_file.write("[quito_configuration]\n")
            current_init_file.write(f"coverage_criterion={coverage_criterion}\n")
            current_init_file.write(f"K={K}\n")
            current_init_file.write(f"M={M}\n")
            current_init_file.write(f"BUDGET={BUDGET}\n")
            current_init_file.write(f"confidence_level={confidence_level}\n")
            current_init_file.write("statistical_test=chi-square test\n")
            current_init_file.write("\n")
            current_init_file.write("[reduqt_parameters]\n")
            # mutation = 0 <-> default program specification
            current_init_file.write(f"Basis={optimized_basis}\n")
            current_init_file.write(f"Mutation={Mutation}\n")
            current_init_file.write(f"original={original}\n")
            current_init_file.write(f"depth={self.depth}\n")
            current_init_file.write("\n")
            current_init_file.write("[state_vector]\n")
            help_sv = [state for state in self.state_vector]
            current_init_file.write(f"{help_sv}\n")
            current_init_file.write("\n")
            current_init_file.write("[program_specification]\n")

            current_init_file.write( convert_state_vector_to_program_spec_format(self.state_vector, self.input) )
#            print(self.input_space)
            for input in self.input_space[1:]:
                self.change_of_input_and_basis(input, self.basis)

                current_init_file.write( convert_state_vector_to_program_spec_format(self.state_vector, input) )

            # revert to original input
            self.change_of_input(original_input)

    def write_circuit_program(self, basis = None, Mutation = 0, original = True):

        list_of_numbers_0_num_qubits_minus_1 = [number for number in range(self.num_qubits)]


        if Mutation != 0:
            if original == True:
                file_extension = f"M{Mutation}"
            if original == False:
                file_extension = f"M{Mutation}" + r"_A"
        else:
            if original == True:
                file_extension = ""
            if original == False:
                file_extension = r"_A"

        if basis == None:
            basis = self.identity_basis


        basis_list = basis.split(",")
        with open( r"study_subjects/" + self.filename[:-2] + file_extension + r".py", "w") as circuit_file:
            """
            Writes a circuit file in a quito format
            adds optimized measurement basis and measurements
            to the default circuit
            """

            circuit_file.write(f"{inspect.getsource(self.circuit)}\n")
            circuit_file.write(f"    # measurement basis {basis}\n")

            # add measurement basis
            for gate_index in range( len(basis_list) ):
                if basis_list[gate_index] != "i":
                    circuit_file.write(f"    qc.{basis_list[gate_index]}({gate_index})\n")

            # add measurement operator
            circuit_file.write(f"    qc.measure({list_of_numbers_0_num_qubits_minus_1}, {list_of_numbers_0_num_qubits_minus_1})")



def correct_init_files():

    programs_list = []

    # var programs
    num_programs_var = 12
    lower_thresh_var = 2
    upper_thresh_var = 8
    num_vars = 4
    ids_var = [1, 2, 2, 1, 3, 1, 2]

    for qubit_index in range(lower_thresh_var, upper_thresh_var + 1):
        for program_index in range( ids_var[qubit_index-lower_thresh_var] ):
            for var_index in range( num_vars ):

                program_name = r"var_" + f"q{qubit_index}_{program_index}{var_index}.init"
                program_name_A = r"var_" + f"q{qubit_index}_{program_index}{var_index}_A.init"

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




    for program_index in range( len(programs_list) ):

        with open( r"study_subjects/" + programs_list[program_index], "r" ) as infile:
            read_lines_file = infile.readlines()

        # count number of output states
        num_pairs = len( read_lines_file[read_lines_file.index("[program_specification]\n") + 1:] )

        print(programs_list[program_index])

        new_K = 10*num_pairs
        new_conf_level = 0.01
#        read_lines_file[1] = r"root=" + root_variable_init + programs_list[program_index][:-4] + "py" + "\n"

        read_lines_file[11] = f"K={new_K}\n"
        read_lines_file[12] = f"M=1\n"
        read_lines_file[14] = f"confidence_level={new_conf_level}\n"

        with open( r"study_subjects/" + programs_list[program_index], "w" ) as outfile:
            for line in read_lines_file:
                outfile.write(line)



def convert_to_basis(num_qubits, value):
    """
    Changes decimal value to basis string
    For instance for 3 qubits, 0 to i,i,i
    1 to i,i,h etc.
    """
    value_binary_string = np.binary_repr(value)

    # add leading zeros
    while len(value_binary_string) < num_qubits:
        value_binary_string = "0" + value_binary_string

    value_binary_list = [x for x in value_binary_string]

    for digit_index in range( num_qubits ):
        if value_binary_list[digit_index] == "0":
            value_binary_list[digit_index] = "i"
        else:
            value_binary_list[digit_index] = "h"

    return ",".join( value_binary_list )


def random_search(quantum_program, greedy_bases_df, run_index):

    # initial number of state_vector components, to be returned
    num_terms_init = count_terms(quantum_program.state_vector)

    filtered_df = greedy_bases_df[ greedy_bases_df['run'] == run_index ]
    filtered_df = filtered_df[ filtered_df['program'] == quantum_program.filename ]
    number_of_transformations = filtered_df['number_of_transformations'].iloc[0]

    # randomly select a set of integers between 0 and num_qubits^2-1 without replacement
    # the set should be the same size as the number of objective function executions performed by
    # the greedy approach for the same run and same program
    make_random_selection = np.random.choice( range(0, quantum_program.num_qubits**2 - 1), size=number_of_transformations, replace=False)

    # array for storing counts for each Hadamard gate choice
    num_terms_arr = np.zeros( number_of_transformations )

    for selection_index in range(number_of_transformations):

        hadamard_basis = convert_to_basis(quantum_program.num_qubits, make_random_selection[selection_index])

        quantum_program.change_of_input_and_basis( quantum_program.input, hadamard_basis )

        num_terms_arr[selection_index] = count_terms(quantum_program.state_vector)

    num_minima_terms = (num_terms_arr == num_terms_arr[np.argmin(num_terms_arr)] )*1

    # randomly pick a minima, can be one or several
    min_index = np.random.choice( np.where( num_minima_terms == 1 )[0] )

    min_value = num_terms_arr[min_index]

    optimized_basis = convert_to_basis( quantum_program.num_qubits, make_random_selection[min_index] )

    if min_value >= num_terms_init:

        # if no reductions found, keep initial basis
        min_value = num_terms_init

        optimized_basis = quantum_program.identity_basis

    return optimized_basis, num_terms_init, min_value, number_of_transformations


def reduqt(quantum_program):
    # find optimized Hadamard-basis

    basis_indices = list( range( quantum_program.num_qubits ) )

    hadamard_basis_list = quantum_program.identity_basis.split(",")

    # initial number of state_vector components for algo
    num_terms_init = count_terms(quantum_program.state_vector)

    # initial number of state_vector components for return
    num_terms_init_return = num_terms_init

    # array for storing counts for each Hadamard gate choice
    num_terms_arr = np.zeros( quantum_program.num_qubits )

    # count number of Hadamard gates in basis
    number_of_hadamard_gates = np.sum( np.array(hadamard_basis_list) == "h" )

    # just a temporary helping counter for programming
    counter = 0

    # number of transformations / number of executions of objective function
    number_of_transformations = 0

    while number_of_hadamard_gates < quantum_program.num_qubits:

        for basis_index in basis_indices:

            hadamard_basis_list[basis_index] = "h"

            hadamard_basis = ",".join(hadamard_basis_list)

            quantum_program.change_of_input_and_basis( quantum_program.input, hadamard_basis )

            number_of_transformations += 1

            num_terms_arr[basis_index] = count_terms(quantum_program.state_vector)

            hadamard_basis_list[basis_index] = "i"


        # compute the number of equal minima

        num_minima_terms = (num_terms_arr == num_terms_arr[np.argmin(num_terms_arr)] )*1

        # randomly pick a minima, can be one or several
        min_index = np.random.choice( np.where( num_minima_terms == 1 )[0] )

        min_value = num_terms_arr[min_index]

        if min_value >= num_terms_init:
            # no further reductions have been found, break loop

            min_value = num_terms_init

            break

        # update basis
        hadamard_basis_list[ min_index ] = "h"

        counter += 1

        basis_indices.remove( min_index )

        num_terms_init = min_value

    hadamard_basis = ",".join(hadamard_basis_list)

    optimized_basis = hadamard_basis

    return optimized_basis, num_terms_init_return, min_value, number_of_transformations



def main(num_runs, approach, greedy_bases_df=None):

    if approach == "greedy":
        filename_greedy_runtimes = r"results_experiment1/greedy_runtimes.txt"
        filename_greedy_bases = r"results_experiment1/greedy_bases.txt"

    if approach == "random":
        filename_greedy_runtimes = r"results_experiment1/random_runtimes.txt"
        filename_greedy_bases = r"results_experiment1/random_bases.txt"

    # file for storing runtimes
    times_file = open( filename_greedy_runtimes, "w" )
    times_file.write("run, program, qubits, run time, number_of_transformations\n")

    times_file.close()

    basis_file = open( filename_greedy_bases, "w" )
    basis_file.write("run, program, basis, Nps_init, Nps_final, reduction\n")

    basis_file.close()

    for program_index in range( len(filenames) ):

        print( "program = ", filenames[program_index] )

        optimal_bases_list = []
        reduction_rate_array = np.zeros(num_runs)

        for run_index in range(num_runs):
        # main loop applying reduction to all programs

            basis_file = open( filename_greedy_bases, "a" )
            times_file = open( filename_greedy_runtimes, "a" )

            time_start_current_program = time.time()

            # set current_program variable
            current_program = eval( filenames[program_index] )

            num_qubits = int( filenames[program_index].split("_")[1][1:] )

            # create quantum program instance
            quantum_program1 = QuantumProgram( num_qubits, current_program.run, filenames[program_index], current_program.input_space )

            # write default program specification
            quantum_program1.write_ps()

            # define ancilla program filename
            ancilla_program_filename = filenames[program_index][:-2] + r"_anc"

            print(ancilla_program_filename)

            ancilla_sv = quantum_program1.read_state_vector_for_ancilla(filenames[program_index][:-2])

            # create ancilla circuit file
            quantum_program1.write_ancilla_circuit(ancilla_program_filename, ancilla_sv)

            # test 1
            assert np.allclose(quantum_program1.state_vector, ancilla_sv), "Mismatch"


            # import ancilla run file
            import_ancilla = f"import {ancilla_program_filename}"

            exec( import_ancilla )

            ancilla_program = eval(ancilla_program_filename)

            # create ancilla program instance
            ancilla_quantum_program = QuantumProgram( num_qubits, ancilla_program.run, filenames[program_index], "zero" )

            # test 2
            assert np.allclose(quantum_program1.state_vector, ancilla_quantum_program.state_vector), "Mismatch"

            if approach == "greedy":
                # locate optimized_basis using greedy approach
                optimized_basis, init_terms, min_terms, number_of_transformations = reduqt(ancilla_quantum_program)

            if approach == "random":
                # locate optimized_basis using random search approach
                optimized_basis, init_terms, min_terms, number_of_transformations = random_search(ancilla_quantum_program, greedy_bases_df, run_index)


            reduction_percent = min_terms / init_terms

            optimal_bases_list.append(optimized_basis)
            reduction_rate_array[ run_index ] = reduction_percent

            basis_file.write(f"{run_index}, {filenames[program_index]}, {optimized_basis}, {init_terms}, {min_terms}, {reduction_percent}\n")


            if run_index == num_runs - 1:

                optimized_basis = optimal_bases_list[ np.argsort(reduction_rate_array)[int( round(num_runs/2) )] ]

                # change basis
                quantum_program1.change_of_input_and_basis( quantum_program1.input, optimized_basis)

                # write reduced ps
                quantum_program1.write_ps(original = False, optimized_basis = optimized_basis)

                # write reduced circuit file with original program and optimized basis
                quantum_program1.write_circuit_program(basis = optimized_basis, original = False)

                time_stop_current_program = time.time()
                run_time_current = time_stop_current_program - time_start_current_program

                # write run time to file for current program
                times_file.write(f"{run_index}, {filenames[program_index]}, {num_qubits}, {run_time_current}, {number_of_transformations}\n")

                # close files
                times_file.close()

                basis_file.close()

            else:

                # change basis

                ancilla_quantum_program.change_of_input_and_basis( ancilla_quantum_program.input, optimized_basis)

                # write reduced ps
                ancilla_quantum_program.write_ps(original = False, optimized_basis = optimized_basis)

                # write reduced circuit file with original program and optimized basis
                quantum_program1.write_circuit_program(basis = optimized_basis, original = False)

                time_stop_current_program = time.time()
                run_time_current = time_stop_current_program - time_start_current_program

                # write run time to file for current program
                times_file.write(f"{run_index}, {filenames[program_index]}, {num_qubits}, {run_time_current}, {number_of_transformations}\n")

                # close files
                times_file.close()

                basis_file.close()



if approach == "greedy":
    print("Running greedy")
    start_time = time.time()
    main(r1, approach)
    stop_time = time.time()
    print("Total Runtime greedy = ", stop_time - start_time)

    # fetch and write file with median bases from basis results
    fetch_and_write_median_basis_file(r1, approach)

# need to have run approach = "greedy" first to have greedy_bases_median.txt file
if approach == "random":
    greedy_bases_df = pd.read_csv( "results_experiment1/greedy_runtimes.txt", delimiter = ", ", engine="python")
    print("Running Random")
    start_time = time.time()
    main(r1, approach, greedy_bases_df)
    stop_time = time.time()
    print("Total Runtime random = ", stop_time - start_time)
    # fetch and write file with median bases from basis results
    fetch_and_write_median_basis_file(r1, approach)

# Add sample sizes and correct paths to init files
correct_init_files()
