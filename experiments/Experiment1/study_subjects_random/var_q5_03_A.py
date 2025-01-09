def run(qc):
    import numpy as np
    qc.h( [0, 1] )
    qc.x(3)
    qc.h(4)
    qc.x(1)
    qc.cx(1, 2)
    qc.x(0)
    qc.cx(0, 1)
    qc.mcx( [0, 1], 2)
    qc.x(4)
    qc.mcp( np.pi, [0, 1, 2], 3 )
    qc.mcp( np.pi, [0, 1, 2], 4 )
    qc.x(4)
    qc.mcx( [0, 1], 2 )
    qc.cx(0, 1)
    qc.x(0)
    qc.cx(1, 2)
    qc.x(1)

    # measurement basis i,i,i,i,h
    qc.h(4)
    qc.measure([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])