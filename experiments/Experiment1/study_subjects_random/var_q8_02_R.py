# Ex 11-3: Drawing into small tiles

# program circuit
number_of_qubits = 8
input_space = ["0,1,0,0,0,0,0,0"]

def run(qc):
    import math
    qc.h([0,1,2,3])
    qc.cx(2, 4)
    qc.cx(3, 5)
    qc.x(6)
    qc.cx(6, 7)
    qc.x([6, 7])
    qc.mcp( math.radians(180), [4, 5, 6, 7], 0)
    qc.mcp( math.radians(180), [4, 5, 6, 7], 1)
    qc.mcp( math.radians(180), [4, 5, 6, 7], 2)
    qc.mcp( math.radians(180), [4, 5, 6, 7], 3)
    qc.x([6, 7])
    qc.cx(6, 7)
    qc.x(6)
    qc.cx(2, 4)
    qc.cx(3, 5)
    qc.cx(0, 4)
    qc.cx(1, 5)
    qc.x(7)
    qc.x(4)
    qc.x(6)
    qc.mcp( math.radians(180), [4, 5, 6, 7], 0)
    qc.mcp( math.radians(180), [4, 5, 6, 7], 1)
    qc.mcp( math.radians(180), [4, 5, 6, 7], 2)
    qc.mcp( math.radians(180), [4, 5, 6, 7], 3)
    qc.x(4)
    qc.x(6)
    qc.x(7)
    qc.cx(0, 4)
    qc.cx(1, 5)
