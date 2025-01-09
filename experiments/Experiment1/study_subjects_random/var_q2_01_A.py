def run(qc):
    import math
    theta = math.pi/2
    qc.h(0)
    qc.h(1)
    qc.rz(theta/2, 1)
    qc.cx(0, 1)
    qc.rz(-theta/2, 1)
    qc.cx(0, 1)
    qc.rz(-theta/2, 0)

    # measurement basis i,h
    qc.h(1)
    qc.measure([0, 1], [0, 1])