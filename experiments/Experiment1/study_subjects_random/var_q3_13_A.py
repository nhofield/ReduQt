def run(qc):
    qc.h(0)
    qc.h(1)
    qc.cx(1,2)
    qc.cx(0,1)
    qc.h(0)

    # measurement basis i,h,h
    qc.h(1)
    qc.h(2)
    qc.measure([0, 1, 2], [0, 1, 2])