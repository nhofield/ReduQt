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
    states_to_amplify = [20, 40, 79, 93, 67, 81, 82, 71, 52, 31, 46, 79, 98, 63, 48, 102, 51, 21, 83, 108, 55, 29, 92, 121, 60, 62, 92, 73, 20, 110, 34, 72, 117, 29, 68, 99, 79, 42, 77, 35, 78, 70, 49, 17, 70, 108, 10, 98, 17, 46, 117, 103, 90, 34, 23, 99, 31, 83, 39, 57, 29, 30, 49, 6, 73, 9, 14, 49, 119, 17, 101, 123, 58, 117, 38, 39, 23, 65, 115, 4, 7, 105]
    num_iterations = 2
    main_quito( num_qubits, states_to_amplify, num_iterations, qc )
