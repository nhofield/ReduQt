def run(qc):
    from qiskit.circuit.library import MCXGate
    qc.x(0)
    qc.h(2)
    qc.z(2)
    qc.x(4)
    qc.h(5)
    qc.z(5)
    qc.append( MCXGate(4), [0, 1, 2, 4, 3] )
    qc.append( MCXGate(3), [0, 1, 4, 2] )
    qc.append( MCXGate(2), [0, 4, 1] )
    qc.append( MCXGate(1), [4, 0] )

    qc.append( MCXGate(4), [1, 2, 4, 5, 3] )
    qc.append( MCXGate(3), [1, 4, 5, 2] )
    qc.append( MCXGate(2), [4, 5, 1] )
    qc.append( MCXGate(4), [1, 2, 4, 5, 3] )
    qc.append( MCXGate(3), [1, 4, 5, 2] )
    qc.append( MCXGate(2), [4, 5, 1] )
    qc.append( MCXGate(2), [2, 5, 3] )
    qc.append( MCXGate(1), [5, 2] )

    # measurement basis i,i,h,i,i,i
    qc.h(2)
    qc.measure([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5])