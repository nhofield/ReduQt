def run(qc):

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


    num_qubits = 11
    main_quito( num_qubits, qc )

    # measurement basis i,h,i,h,i,h,i,i,h,i,i
    qc.h(1)
    qc.h(3)
    qc.h(5)
    qc.h(8)
    qc.measure([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])