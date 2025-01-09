# Ex 12-3: Shor step-by-step
# Could be factorized by Bell state basis
# program circuit

number_of_qubits = 8
input_space = ["0,0,0,0,1,0,0,1"]

def run(qc):
    import math
    qc.x(0)
    qc.x(2)
    qc.h([4, 5, 6, 7])
    qc.cswap(4, 2, 3)
    qc.cswap(4, 1, 2)
    qc.cswap(4, 0, 1)
    qc.cswap(5, 1, 3)
    qc.cswap(5, 0, 2)
    qc.cswap(5, 0, 1)
    qc.h(7)
    qc.cp(math.radians(-90  ), 6, 7)
    qc.cp(math.radians(-45  ), 5, 7)
    qc.cp(math.radians(-22.5), 4, 7)
    qc.h(6)
    qc.cp(math.radians(-90  ), 5, 6)
    qc.cp(math.radians(-45  ), 4, 6)
    qc.h(5)
    qc.cp(math.radians(-90  ), 4, 5)
    qc.h(4)
    qc.swap(4, 7)
    qc.swap(5, 6)
    qc.measure([0,1,2,3,4,5,6,7], [0,1,2,3,4,5,6,7])
