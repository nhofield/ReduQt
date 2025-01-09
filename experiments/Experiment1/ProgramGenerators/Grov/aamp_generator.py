import qiskit as qk
import numpy as np
import inspect
from itertools import combinations

# Amplitude amplitfication
# Author: Noah Hegerland Oldfield




def print_state_vector(state_vector):

    for state_index in range( len(state_vector) ):
#        prob = np.around( abs( state_vector[state_index] )**2 , 4)
        prob = abs( state_vector[state_index] )**2

        print("|" + f"{state_index}" + ">", "amplitude = ", state_vector[state_index], "probability = ", prob, "%")


def oracle(states_to_amplify, qc, qubit_indices, all_indices, num_qubits):
    """
    This oracle flips the phase of all states in
    'states_to_flip'
    oracle is equal to the inverse of oracle
    states_to_amplify is list of decimal integer values
    [0,2,3,5,...]
    """

    for state_index in range( len(states_to_amplify) ):

        state_string_binary = np.binary_repr( states_to_amplify[state_index] )

        list_of_bits_to_flip = []

        # add leading zeros
        while len(state_string_binary) < ( num_qubits - 1 ):
            state_string_binary = "0" + state_string_binary

        # reverse the string
        state_string_binary = state_string_binary[::-1]


        for bit_index in range( num_qubits - 1 ):
            if state_string_binary[bit_index] != "1":
                list_of_bits_to_flip.append(bit_index)


        # transforms the state to be phase flipped to the |1111...> state
        if len(list_of_bits_to_flip) == 0:
            # performs the phase flip to the state involving the ancilla bit
            qc.mcp(np.pi, qubit_indices, num_qubits - 1)

        else:
            qc.x(list_of_bits_to_flip)
            # performs the phase flip to the state involving the ancilla bit
            qc.mcp(np.pi, qubit_indices, num_qubits - 1)
            # reverses the |1111...> state transformation
            qc.x(list_of_bits_to_flip)

def groverOperator(qc, qubit_indices, all_indices, num_qubits):
    """
    Applies the Grover operator to the given quantum circuit qc
    """
    qc.barrier(all_indices)
    qc.h(qubit_indices)
    qc.x(qubit_indices)
    qc.mcp(np.pi, qubit_indices, num_qubits - 1)
    qc.x(qubit_indices)
    qc.h(qubit_indices)
    qc.barrier(all_indices)

def add_measurements(qc):
    qc.measure(all_indices, all_indices)

def oracle_quito(states_to_phase_flip, qc, qubit_indices, all_indices, num_qubits):
    """
    This oracle flips the phase of all states in
    'states_to_flip'
    oracle is equal to the inverse of oracle
    states_to_phase_flip is list of decimal integer values
    [0,2,3,5,...]
    """

    for state_index in range( len(states_to_amplify) ):

        state_string_binary = np.binary_repr( states_to_amplify[state_index] )

        list_of_bits_to_flip = []

        # add leading zeros
        while len(state_string_binary) < ( num_qubits - 1 ):
            state_string_binary = "0" + state_string_binary

        # reverse the string
        state_string_binary = state_string_binary[::-1]


        for bit_index in range( num_qubits - 1 ):
            if state_string_binary[bit_index] != "1":
                list_of_bits_to_flip.append(bit_index)

        # transforms the state to be phase flipped to the |1111...> state
        qc.x(list_of_bits_to_flip)
        # performs the phase flip to the state involving the ancilla bit
        qc.mcp(np.pi, qubit_indices, num_qubits - 1)
        # reverses the |1111...> state transformation
        qc.x(list_of_bits_to_flip)

def main_quito(num_qubits, states_to_amplify, num_iterations, qc):
    """
    Runs the program once
    returns state_vector and circuit object
    """

    all_indices = [x for x in range(num_qubits)]
    qubit_indices = all_indices[:-1]

    # ancilla qubit set to 1 throughout whole algorithm
    qc.x( all_indices[-1] )
    qc.h(qubit_indices)

    qc.barrier(all_indices)
    for iteration in range( num_iterations ):
        oracle_quito(states_to_amplify, qc, qubit_indices, all_indices, num_qubits)
        groverOperator(qc, qubit_indices, all_indices, num_qubits)

    # set ancilla to 0
    qc.x( all_indices[-1] )


def main(num_qubits, states_to_amplify, num_iterations):
    """
    Runs the program once
    returns state_vector and circuit object
    """

    all_indices = [x for x in range(num_qubits)]
    qubit_indices = all_indices[:-1]

    qc = qk.QuantumCircuit(num_qubits, num_qubits)

    # ancilla qubit set to 1 throughout whole algorithm
    qc.x( all_indices[-1] )
    qc.h(qubit_indices)

    qc.barrier(all_indices)
    for iteration in range( num_iterations ):
        oracle(states_to_amplify, qc, qubit_indices, all_indices, num_qubits)
        groverOperator(qc, qubit_indices, all_indices, num_qubits)

    # set ancilla to 0
    qc.x( all_indices[-1] )





    backend = qk.Aer.get_backend("statevector_simulator")
    job = qk.execute(qc, backend)
    state_vector = job.result().get_statevector(decimals=3)

    return np.array(state_vector), qc




def minimize_error(state_vector_tolerance, num_qubits, states_to_amplify):

    state_vector_constraint = 1
    number_of_iterations = 1

    while state_vector_constraint >= state_vector_tolerance:



        outputstate, qc = main(num_qubits, states_to_amplify, number_of_iterations)

        print_state_vector(outputstate)

        for state_index in range( len(outputstate) ):
            if outputstate[state_index] not in outputstate[states_to_amplify]:
                state_vector_constraint = abs( outputstate[state_index] )**2
                break

        print("State Vector Constraint = ", state_vector_constraint)
        number_of_iterations += 1

        print("Number of iterations = ", number_of_iterations)

    return number_of_iterations, outputstate, qc


def write_circuit_program(num_qubits, states_to_amplify, num_iterations, id):

    filename = r"aamp_" + f"q{num_qubits}" + f"_{id}" + r".py"

    main_program_function_string   = inspect.getsourcelines(main_quito)
    oracle_program_function_string = inspect.getsourcelines(oracle_quito)
    groverOperator_program_function_string = inspect.getsourcelines(groverOperator)



    with open(filename, "w") as outfile:
        outfile.write("def run(qc):\n")
        outfile.write("    import numpy as np\n")
        outfile.write("\n")

        for line in oracle_program_function_string[0]:
            outfile.write("    " + line)

        outfile.write("\n")

        for line in groverOperator_program_function_string[0]:
            outfile.write("    " + line)

        outfile.write("\n")

        for line in main_program_function_string[0]:
            outfile.write("    " + line)

        outfile.write("\n")

        outfile.write("\n")
        outfile.write(f"    num_qubits = {num_qubits}")
        outfile.write("\n")
        outfile.write(f"    states_to_amplify = {states_to_amplify}")
        outfile.write("\n")
        outfile.write(f"    num_iterations = {num_iterations}")
        outfile.write("\n")

        outfile.write(f"    main_quito( num_qubits, states_to_amplify, num_iterations, qc )")


def test_program(num_qubits, states_to_amplify, num_iterations):

    qc = qk.QuantumCircuit(num_qubits, num_qubits)
    import aa_q30_1
    aa_q30_1.run(qc)
    backend = qk.Aer.get_backend("statevector_simulator")
    job = qk.execute(qc, backend)
    statevector = job.result().get_statevector()
#    print_state_vector(statevector)


#-----------
# Main Loop!
#-----------

state_vector_tolerance = 1E-4

num_qubits_lower_bound = 7
num_qubits_upper_bound = 15
number_of_programs = 2*(8 - 2)

for qubit_index in range( num_qubits_lower_bound, num_qubits_upper_bound + 1 ):
    print("qubits = ", qubit_index)
    state_vector_range = np.arange(qubit_index**2)

    num_qubits = qubit_index - 1

    state_vector_length = 2**num_qubits
    state_vector_length_half_minus_2 = 2**(num_qubits - 1) - 2

    increment = ( state_vector_length_half_minus_2 - 2 )/number_of_programs

    print(state_vector_length)
    print(state_vector_length_half_minus_2)
    print(increment)

    increment = int(increment)

    id = 0

    for choice_size_index in range(2, state_vector_length_half_minus_2, increment):
        random_state_choice = list( np.random.randint(0, state_vector_length_half_minus_2, choice_size_index) )
        print(random_state_choice)
        number_of_iterations, outputstate, qc = minimize_error(state_vector_tolerance, qubit_index, random_state_choice)

        write_circuit_program(qubit_index, random_state_choice, number_of_iterations, id)
        id += 1


#print(qc.draw())
#print("optimal_number_of_iterations = ", optimal_number_of_iterations)
#print_state_vector(state_vector)

#write_circuit_program(num_qubits, states_to_amplify, 1, 1)
