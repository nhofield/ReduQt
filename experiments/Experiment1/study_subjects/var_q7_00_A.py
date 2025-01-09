def run(qc):
    import math
    qc.h([0,1,2])

    # clause 1
    qc.x([0,1])
    qc.mcx([0,1], 3)
    qc.x([0,1])
    qc.x(3)

    # clause 2
    qc.x(0)
    qc.x(0)
    qc.x(2)
    qc.mcx([0,2], 4)
    qc.x(0)
    qc.x(2)
    qc.x(4)
    qc.x(0)

    # clause 3
    qc.x([1,2])
    qc.x([1,2])
    qc.mcx([1,2], 5)
    qc.x([1,2])
    qc.x(5)
    qc.x([1,2])

    # clause 4
    qc.x(0)
    qc.x(2)
    qc.mcx([0,2], 6)
    qc.x(0)
    qc.x(2)
    qc.x(6)

    # flip phase
    qc.mcp(math.radians(180), [1,3,4,5,6], 0)
    qc.mcp(math.radians(180), [1,3,4,5,6], 2)

    # inverse clause 4
    qc.x(6)
    qc.x(0)
    qc.x(2)
    qc.mcx([0,2], 6)
    qc.x(0)
    qc.x(2)

    # inverse clause 3
    qc.x([1,2])
    qc.x([1,2])
    qc.x(5)
    qc.mcx([1,2], 5)
    qc.x([1,2])
    qc.x([1,2])

    # inverse clause 2
    qc.x(0)
    qc.x(0)
    qc.x(2)
    qc.x(4)
    qc.mcx([0,2], 4)
    qc.x(0)
    qc.x(2)
    qc.x(0)

    # inverse clause 1
    qc.x([0,1])
    qc.x(3)
    qc.mcx([0,1], 3)
    qc.x([0,1])

    # Grover mirror
    qc.h([0,1,2])
    qc.x([0,1,2])
    qc.mcp(math.radians(180), [0,1,2], 3)
    qc.mcp(math.radians(180), [0,1,2], 4)
    qc.mcp(math.radians(180), [0,1,2], 5)
    qc.mcp(math.radians(180), [0,1,2], 6)
    qc.x([0,1,2])
    qc.h([0,1,2])

    # measurement basis h,h,h,i,i,i,i
    qc.h(0)
    qc.h(1)
    qc.h(2)
    qc.measure([0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6])