input_space = "zero"

def run(qc):
    import numpy as np

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


    num_qubits = 9
    states_to_amplify = [52, 48, 99, 73, 50, 116, 110, 50, 73, 42, 79, 96, 120, 9, 54, 121, 20, 115, 106, 59, 0, 75, 114, 49, 31, 120, 19, 51, 96, 40, 31, 45, 113, 49, 89, 35, 21, 38, 85, 94, 47, 76, 9, 30, 1, 36, 76, 49, 52, 104, 61, 66, 10, 18, 99, 61, 110, 109, 124, 44, 124, 119, 48, 109, 23, 6, 118, 4, 46, 46, 99, 68]
    num_iterations = 2
    main_quito( num_qubits, states_to_amplify, num_iterations, qc )
    qc.measure([0,1,2,3,4,5,6,7,8], [0,1,2,3,4,5,6,7,8])
