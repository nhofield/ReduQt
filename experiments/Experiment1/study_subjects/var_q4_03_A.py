def run(qc):
    import math
    all = [0,1,2,3]
    number_of_iterations = 4

    qc.h(all)

    for iteration in range(number_of_iterations):


        #------- Mirror
        qc.x(all)
        qc.mcp( math.radians(180), all[:-1], all[-1] )
        qc.x(all)
        #-------

        #-------
        qc.x(all[1:])
        qc.mcp( math.radians(180), all[:-1], all[-1] )
        qc.x(all[1:])
        #-------

        #-------
        qc.x([0,2,3])
        qc.mcp( math.radians(180), all[:-1], all[-1] )
        qc.x([0,2,3])
        #-------

        #-------
        qc.x([2,3])
        qc.mcp( math.radians(180), all[:-1], all[-1] )
        qc.x([2,3])
        #-------

        #Grover iteration
        qc.h(all)
        qc.x(all)
        qc.mcp( math.radians(180), all[:-1], all[-1] )
        qc.x(all)
        qc.h(all)

    # measurement basis h,h,h,h
    qc.h(0)
    qc.h(1)
    qc.h(2)
    qc.h(3)
    qc.measure([0, 1, 2, 3], [0, 1, 2, 3])