import qiskit as qk
import inspect
import numpy as np

def print_state_vector(state_vector):

    for state_index in range( len(state_vector) ):
#        prob = np.around( abs( state_vector[state_index] )**2 , 4)
        prob = abs( state_vector[state_index] )**2

        print("|" + f"{state_index}" + ">", "amplitude = ", state_vector[state_index], "probability = ", prob, "%")

def main_quito(num_qubits, depth, qc):
    """Returns a quantum circuit implementing the Quantum Walk algorithm.
    Keyword arguments:
    num_qubits -- number of qubits of the returned quantum circuit
    depth -- number of quantum steps
    coin_state_preparation -- optional quantum circuit for state preparation
    ancillary_mode -- defining the decomposition scheme
    """

    n = num_qubits - 1  # because one qubit is needed for the coin

    n_anc = 0

    coin = n
    node = list( range(n) )

    ancillary_mode = "noancilla"

    for _ in range(depth):
        # Hadamard coin operator
        qc.h(coin)

        # controlled increment
        for i in range(0, n - 1):
            qc.mcx( [ coin ] + node[i + 1 :], node[i], mode=ancillary_mode)
        qc.cx(coin, node[n - 1])

        # controlled decrement
        qc.x(coin)
        qc.x(node[1:])
        for i in range(0, n - 1):
            qc.mcx( [ coin ] + node[i + 1 :], node[i], mode=ancillary_mode)
        qc.cx( coin, node[n - 1])
        qc.x(node[1:])
        qc.x(coin)


def main( num_qubits, depth ):
    """Returns a quantum circuit implementing the Quantum Walk algorithm.
    Keyword arguments:
    num_qubits -- number of qubits of the returned quantum circuit
    depth -- number of quantum steps
    coin_state_preparation -- optional quantum circuit for state preparation
    ancillary_mode -- defining the decomposition scheme
    """

    n = num_qubits - 1  # because one qubit is needed for the coin
    coin = qk.QuantumRegister(1, "coin")
    node = qk.QuantumRegister(n, "node")

    n_anc = 0
    qc = qk.QuantumCircuit(node, coin, name="qwalk")

    ancillary_mode = "noancilla"

    for _ in range(depth):
        # Hadamard coin operator
        qc.h(coin)

        # controlled increment
        for i in range(0, n - 1):
            qc.mcx(coin[:] + node[i + 1 :], node[i], mode=ancillary_mode)
        qc.cx(coin, node[n - 1])

        # controlled decrement
        qc.x(coin)
        qc.x(node[1:])
        for i in range(0, n - 1):
            qc.mcx(coin[:] + node[i + 1 :], node[i], mode=ancillary_mode)
        qc.cx(coin, node[n - 1])
        qc.x(node[1:])
        qc.x(coin)

    return qc

def write_circuit_program(num_qubits, depth, id):

    filename = r"qwalk_" + f"q{num_qubits}" + f"_{id}" + r".py"

    main_program_function_string = inspect.getsourcelines(main_quito)
    qubit_list = [x for x in range(num_qubits)]

    with open(filename, "w") as outfile:
        outfile.write('input_space = "zero"\n')
        outfile.write('\n')
        outfile.write("def run(qc):\n")
        outfile.write("\n")

        for line in main_program_function_string[0]:
            outfile.write("    " + line)

        outfile.write("\n")

        outfile.write("\n")
        outfile.write(f"    num_qubits = {num_qubits}")
        outfile.write("\n")
        outfile.write(f"    depth = {depth}")
        outfile.write("\n")

        outfile.write(f"    main_quito( num_qubits, depth, qc )\n")
        outfile.write(f"    qc.measure({qubit_list}, {qubit_list})")

    filename = r"qwalk_" + f"q{num_qubits}" + f"_{id}_R" + r".py"

    main_program_function_string = inspect.getsourcelines(main_quito)

    with open(filename, "w") as outfile:
        outfile.write('input_space = "zero"\n')
        outfile.write('\n')
        outfile.write("def run(qc):\n")
        outfile.write("\n")

        for line in main_program_function_string[0]:
            outfile.write("    " + line)

        outfile.write("\n")

        outfile.write("\n")
        outfile.write(f"    num_qubits = {num_qubits}")
        outfile.write("\n")
        outfile.write(f"    depth = {depth}")
        outfile.write("\n")
        outfile.write(f"    main_quito( num_qubits, depth, qc )")

def test_statevector(probabilities):
    # returns True or False
    unique, counts = np.unique(probabilities, return_counts=True)

    if unique[0] == 0:

        counts = counts[1:]

        if len(counts) == 1 and counts[0] <= 2:
            return False

#        print(unique)
#        print(counts)
        number_of_multiple_entries = len(np.where(counts >= 2)[0])
#        print("Number of multiple entries = ", number_of_multiple_entries)
#        print("Ratio = ", number_of_multiple_entries/len(counts))
        if number_of_multiple_entries/len(counts) >= 0.5:
            return True
        else:
            return False

    else:

        if len(counts) == 1 and counts[0] <= 2:
            return False

        number_of_multiple_entries = len(np.where(counts >= 2)[0])
        if number_of_multiple_entries/len(counts) >= 0.5:
            return True
        else:
            return False

def test_program(num_qubits, id):

    filename = r"qwalk_" + f"q{num_qubits}" + f"_{id}_R"
    import_filename_string = r"import " + filename
    exec(import_filename_string)
    program = eval(filename)
    qc = qk.QuantumCircuit(num_qubits, num_qubits)
    program.run(qc)
#    print("Depth = ", qc.depth())
    backend = qk.Aer.get_backend("statevector_simulator")
    job = qk.execute(qc, backend)
    statevector = job.result().get_statevector(decimals=4)
    probabilities = abs(np.array(statevector))**2
    print_state_vector(statevector)

    return test_statevector(probabilities)
#-----------
# Main Loop!
#-----------


def write_test_print_program(num_qubits, depth, id):
    #write program
    write_circuit_program(num_qubits, depth, id)
    result = test_program(num_qubits, id)

    return result

def program_selector():

    num_depths = 30

    initial_depth = 2

    qubit_lower_threshold = 4
    qubit_upper_threshold = 6
    accepted = np.zeros((qubit_upper_threshold-qubit_lower_threshold, num_depths-initial_depth))
    for qbit_index in range(qubit_lower_threshold, qubit_upper_threshold):
        for d in range(initial_depth, num_depths):
            val = write_test_print_program(qbit_index, d, 100 + d)
            accepted[qbit_index-qubit_lower_threshold][d-initial_depth] = val

    print( accepted )


program_selector()

exit()

num_qubits_lower_bound = 3
num_qubits_upper_bound = 10
depth_increment = 1
num_depths = 30
initial_depth = 15

for qubit_index in range( num_qubits_lower_bound, num_qubits_upper_bound ):

    print("qubits = ", qubit_index)
    id = 0
    for depth_index in range( initial_depth, num_depths + 1 , depth_increment ):

        write_circuit_program(qubit_index, depth_index, id)

        id += 1
