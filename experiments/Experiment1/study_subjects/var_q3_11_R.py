# Example 4-1 Basic teleportation of state |+>

# program circuit
number_of_qubits = 3
input_space = ["1,0,0"]

def run(qc):
    qc.h(0)
    qc.h(1)
    qc.cx(1,2)
    qc.cx(0,1)
    qc.h(0)
