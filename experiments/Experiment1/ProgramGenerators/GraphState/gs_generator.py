import qiskit as qk
import numpy as np
import inspect
from itertools import combinations

# Graph States
# Author: Noah Hegerland Oldfield


def print_state_vector(state_vector):

    for state_index in range( len(state_vector) ):
#        prob = np.around( abs( state_vector[state_index] )**2 , 4)
        prob = abs( state_vector[state_index] )**2

        print("|" + f"{state_index}" + ">", "amplitude = ", state_vector[state_index], "probability = ", prob, "%")



def main_quito(num_qubits, qc):
    """
    Runs the program once
    returns state_vector and circuit object
    """

    qubit_range = [x for x in range( num_qubits )]

    qubit_range_shift = qubit_range[1:] + [ qubit_range[0] ]

    qubit_range_last_element = qubit_range[-1]

    qubit_range_shift_last_element = qubit_range_shift[-1]

    qubit_range[-1] = qubit_range_shift_last_element

    qubit_range_shift[-1] = qubit_range_last_element

    qc.h( qubit_range )

    for qubit_index in range( num_qubits ):
        qc.cz( qubit_range[ qubit_index ], qubit_range_shift[ qubit_index ] )


def main(num_qubits):
    """
    Runs the program once
    returns state_vector and circuit object
    """

    qc = qk.QuantumCircuit(num_qubits, num_qubits)

    qubit_range = [x for x in range( num_qubits )]

    qubit_range_shift = qubit_range[1:] + [ qubit_range[0] ]

    qubit_range_last_element = qubit_range[-1]

    qubit_range_shift_last_element = qubit_range_shift[-1]

    qubit_range[-1] = qubit_range_shift_last_element

    qubit_range_shift[-1] = qubit_range_last_element

    qc.h( qubit_range )

    for qubit_index in range( num_qubits ):
        qc.cz( qubit_range[ qubit_index ], qubit_range_shift[ qubit_index ] )

    backend = qk.Aer.get_backend("statevector_simulator")
    job = qk.execute(qc, backend)
    state_vector = job.result().get_statevector(decimals=3)

    return np.array(state_vector), qc



def write_circuit_program(num_qubits):

    qubit_list = [x for x in range(num_qubits)]

    filename = r"gs_" + f"q{num_qubits}" + r".py"

    main_program_function_string   = inspect.getsourcelines(main_quito)

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

        outfile.write(f"    main_quito( num_qubits, qc )\n")
        outfile.write(f"    qc.measure({qubit_list}, {qubit_list})")

    filename = r"gs_" + f"q{num_qubits}_R" + r".py"

    main_program_function_string   = inspect.getsourcelines(main_quito)

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

        outfile.write(f"    main_quito( num_qubits, qc )")


def test_program(num_qubits):

    qc = qk.QuantumCircuit(num_qubits, num_qubits)
    import gs_q30_27
    gs_q30_27.run(qc)
    backend = qk.Aer.get_backend("statevector_simulator")
    job = qk.execute(qc, backend)
    statevector = job.result().get_statevector()
#    print_state_vector(statevector)


#-----------
# Main Loop!
#-----------

num_qubits_lower_bound = 3
num_qubits_upper_bound = 20

for qubit_index in range( num_qubits_lower_bound, num_qubits_upper_bound + 1 ):

    print("qubits = ", qubit_index)

    write_circuit_program(qubit_index)
