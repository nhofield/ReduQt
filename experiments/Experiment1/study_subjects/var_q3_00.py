## Programming Quantum Computers
##   by Eric Johnston, Nic Harrigan and Mercedes Gimeno-Segovia
##   O'Reilly Media
##
## More samples like this can be found at http://oreilly-qc.github.io
if __name__ == "__main__":

    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer, IBMQ, BasicAer
    import math
    ## Uncomment the next line to see diagrams when running in a notebook
    #%matplotlib inline

    ## Example 3-3: Phase Kickback
    # Set up the program
    reg1 = QuantumRegister(2, name='reg1')
    reg2 = QuantumRegister(1, name='reg2')
    qc = QuantumCircuit(reg1, reg2)

    qc.h(reg1)         # put a into reg1 superposition of 0,1,2,3
    qc.cu1(math.pi/4, reg1[0], reg2)
    qc.cu1(math.pi/2, reg1[1], reg2)

    backend = BasicAer.get_backend('statevector_simulator')
    job = execute(qc, backend)
    result = job.result()

    outputstate = result.get_statevector(qc, decimals=3)
    print(outputstate)
    qc.draw()        # draw the circuit

## Example 3-3: Phase Kickback
# program circuit
number_of_qubits = 3
input_space = "zero"

def run(qc):
    import math
    qc.h(0)
    qc.h(1)
    qc.cp(math.pi, 0, 2)
    qc.cp(math.pi, 1, 2)
    qc.measure([0,1,2], [0,1,2])
