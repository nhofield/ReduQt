def run(qc):
    import math
    qc.h(0)
    qc.h(1)
    qc.cp(math.pi, 0, 2)
    qc.cp(math.pi, 1, 2)

    # measurement basis h,h,i
    qc.h(0)
    qc.h(1)
    qc.measure([0, 1, 2], [0, 1, 2])