def run(qc):
    import math
    qc.h([0,1,2])
    qc.x(1)
    qc.x([0,1])
    qc.mcx([0,1], 3)
    qc.x([0,1])
    qc.x(3)
    qc.mcp( math.radians(180), [2,3], 0)
    qc.mcp( math.radians(180), [2,3], 1)
    qc.x([0,1])
    qc.x(3)
    qc.mcx([0,1], 3)
    qc.x([0,1])
    qc.x(1)

    # measurement basis h,h,i,i
    qc.h(0)
    qc.h(1)
    qc.measure([0, 1, 2, 3], [0, 1, 2, 3])