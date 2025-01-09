def run(qc):

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


    num_qubits = 3
    depth = 14
    main_quito( num_qubits, depth, qc )

    # measurement basis h,i,i
    qc.h(0)
    qc.measure([0, 1, 2], [0, 1, 2])