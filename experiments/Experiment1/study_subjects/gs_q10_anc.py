input_space = 'zero'

def run(qc):
    qc.initialize([(0.06250000000000001+0j), 0j, (0.06250000000000001+0j), 0j, (0.06250000000000001+0j), 0j, (-0.06250000000000001+0j), 0j, (0.06250000000000001+0j), 0j, (0.06250000000000001+0j), 0j, (-0.06250000000000001+0j), 0j, (0.06250000000000001+0j), 0j, (0.06250000000000001+0j), 0j, (0.06250000000000001+0j), 0j, (0.06250000000000001+0j), 0j, (-0.06250000000000001+0j), 0j, (-0.06250000000000001+0j), 0j, (-0.06250000000000001+0j), 0j, (0.06250000000000001+0j), 0j, (-0.0625+0j), 0j, (0.06250000000000001+0j), 0j, (0.06250000000000001+0j), 0j, (0.06250000000000001+0j), 0j, (-0.06250000000000001+0j), 0j, (0.06250000000000001+0j), 0j, (0.06250000000000001+0j), 0j, (-0.06250000000000001+0j), 0j, (0.0625+0j), 0j, (-0.06250000000000001+0j), 0j, (-0.06250000000000001+0j), 0j, (-0.06250000000000001+0j), 0j, (0.0625+0j), 0j, (0.06250000000000001+0j), 0j, (0.0625+0j), 0j, (-0.0625+0j), 0j, (0.062499999999999986+0j), 0j, (0.06250000000000001+0j), 0j, (0.06250000000000001+0j), 0j, (0.06250000000000001+0j), 0j, (-0.06250000000000001+0j), 0j, (0.06250000000000001+0j), 0j, (0.06250000000000001+0j), 0j, (-0.06250000000000001+0j), 0j, (0.0625+0j), 0j, (0.06250000000000001+0j), 0j, (0.06250000000000001+0j), 0j, (0.06250000000000001+0j), 0j, (-0.0625+0j), 0j, (-0.06250000000000001+0j), 0j, (-0.0625+0j), 0j, (0.0625+0j), 0j, (-0.062499999999999986+0j), 0j, (-0.06250000000000001+0j), 0j, (-0.06250000000000001+0j), 0j, (-0.06250000000000001+0j), 0j, (0.0625+0j), 0j, (-0.06250000000000001+0j), 0j, (-0.0625+0j), 0j, (0.0625+0j), 0j, (-0.062499999999999986+0j), 0j, (0.06250000000000001+0j), 0j, (0.0625+0j), 0j, (0.0625+0j), 0j, (-0.062499999999999986+0j), 0j, (-0.0625+0j), 0j, (-0.062499999999999986+0j), 0j, (0.062499999999999986+0j), 0j, (-0.06249999999999998+0j), 0j, (0.06250000000000001+0j), 0j, (0.06250000000000001+0j), 0j, (0.06250000000000001+0j), 0j, (-0.06250000000000001+0j), 0j, (0.06250000000000001+0j), 0j, (0.06250000000000001+0j), 0j, (-0.06250000000000001+0j), 0j, (0.0625+0j), 0j, (0.06250000000000001+0j), 0j, (0.06250000000000001+0j), 0j, (0.06250000000000001+0j), 0j, (-0.0625+0j), 0j, (-0.06250000000000001+0j), 0j, (-0.0625+0j), 0j, (0.0625+0j), 0j, (-0.062499999999999986+0j), 0j, (0.06250000000000001+0j), 0j, (0.06250000000000001+0j), 0j, (0.06250000000000001+0j), 0j, (-0.0625+0j), 0j, (0.06250000000000001+0j), 0j, (0.0625+0j), 0j, (-0.0625+0j), 0j, (0.062499999999999986+0j), 0j, (-0.06250000000000001+0j), 0j, (-0.0625+0j), 0j, (-0.0625+0j), 0j, (0.062499999999999986+0j), 0j, (0.0625+0j), 0j, (0.062499999999999986+0j), 0j, (-0.062499999999999986+0j), 0j, (0.06249999999999998+0j), 0j, (-0.06250000000000001+0j), 0j, (-0.06250000000000001+0j), 0j, (-0.06250000000000001+0j), 0j, (0.0625+0j), 0j, (-0.06250000000000001+0j), 0j, (-0.0625+0j), 0j, (0.0625+0j), 0j, (-0.062499999999999986+0j), 0j, (-0.06250000000000001+0j), 0j, (-0.0625+0j), 0j, (-0.0625+0j), 0j, (0.062499999999999986+0j), 0j, (0.0625+0j), 0j, (0.062499999999999986+0j), 0j, (-0.062499999999999986+0j), 0j, (0.06249999999999998+0j), 0j, (0.0625+0j), 0j, (0.0625+0j), 0j, (0.0625+0j), 0j, (-0.062499999999999986+0j), 0j, (0.0625+0j), 0j, (0.062499999999999986+0j), 0j, (-0.062499999999999986+0j), 0j, (0.06249999999999998+0j), 0j, (-0.0625+0j), 0j, (-0.062499999999999986+0j), 0j, (-0.062499999999999986+0j), 0j, (0.06249999999999998+0j), 0j, (0.062499999999999986+0j), 0j, (0.06249999999999998+0j), 0j, (-0.06249999999999998+0j), 0j, (0.06249999999999997+0j), 0j, (0.06250000000000001+0j), 0j, (0.06250000000000001+0j), 0j, (0.06250000000000001+0j), 0j, (-0.06250000000000001+0j), 0j, (0.06250000000000001+0j), 0j, (0.06250000000000001+0j), 0j, (-0.06250000000000001+0j), 0j, (0.0625+0j), 0j, (0.06250000000000001+0j), 0j, (0.06250000000000001+0j), 0j, (0.06250000000000001+0j), 0j, (-0.0625+0j), 0j, (-0.06250000000000001+0j), 0j, (-0.0625+0j), 0j, (0.0625+0j), 0j, (-0.062499999999999986+0j), 0j, (0.06250000000000001+0j), 0j, (0.06250000000000001+0j), 0j, (0.06250000000000001+0j), 0j, (-0.0625+0j), 0j, (0.06250000000000001+0j), 0j, (0.0625+0j), 0j, (-0.0625+0j), 0j, (0.062499999999999986+0j), 0j, (-0.06250000000000001+0j), 0j, (-0.0625+0j), 0j, (-0.0625+0j), 0j, (0.062499999999999986+0j), 0j, (0.0625+0j), 0j, (0.062499999999999986+0j), 0j, (-0.062499999999999986+0j), 0j, (0.06249999999999998+0j), 0j, (0.06250000000000001+0j), 0j, (0.06250000000000001+0j), 0j, (0.06250000000000001+0j), 0j, (-0.0625+0j), 0j, (0.06250000000000001+0j), 0j, (0.0625+0j), 0j, (-0.0625+0j), 0j, (0.062499999999999986+0j), 0j, (0.06250000000000001+0j), 0j, (0.0625+0j), 0j, (0.0625+0j), 0j, (-0.062499999999999986+0j), 0j, (-0.0625+0j), 0j, (-0.062499999999999986+0j), 0j, (0.062499999999999986+0j), 0j, (-0.06249999999999998+0j), 0j, (-0.0625+0j), 0j, (-0.0625+0j), 0j, (-0.0625+0j), 0j, (0.062499999999999986+0j), 0j, (-0.0625+0j), 0j, (-0.062499999999999986+0j), 0j, (0.062499999999999986+0j), 0j, (-0.06249999999999998+0j), 0j, (0.0625+0j), 0j, (0.062499999999999986+0j), 0j, (0.062499999999999986+0j), 0j, (-0.06249999999999998+0j), 0j, (-0.062499999999999986+0j), 0j, (-0.06249999999999998+0j), 0j, (0.06249999999999998+0j), 0j, (-0.06249999999999997+0j), 0j, (-0.0625+0j), (-0+0j), (-0.0625+0j), (-0+0j), (-0.0625+0j), (-0+0j), (0.0625-0j), (-0+0j), (-0.0625+0j), (-0+0j), (-0.0625+0j), (-0+0j), (0.0625-0j), (-0+0j), (-0.062499999999999986+0j), (-0+0j), (-0.0625+0j), (-0+0j), (-0.0625+0j), (-0+0j), (-0.0625+0j), (-0+0j), (0.062499999999999986-0j), (-0+0j), (0.0625-0j), (-0+0j), (0.062499999999999986-0j), (-0+0j), (-0.062499999999999986+0j), (-0+0j), (0.06249999999999998-0j), (-0+0j), (-0.0625+0j), (-0+0j), (-0.0625+0j), (-0+0j), (-0.0625+0j), (-0+0j), (0.062499999999999986-0j), (-0+0j), (-0.0625+0j), (-0+0j), (-0.062499999999999986+0j), (-0+0j), (0.062499999999999986-0j), (-0+0j), (-0.06249999999999998+0j), (-0+0j), (0.0625-0j), (-0+0j), (0.062499999999999986-0j), (-0+0j), (0.062499999999999986-0j), (-0+0j), (-0.06249999999999998+0j), (-0+0j), (-0.062499999999999986+0j), (-0+0j), (-0.06249999999999998+0j), (-0+0j), (0.06249999999999998-0j), (-0+0j), (-0.06249999999999997+0j), (-0+0j), (0.0625-0j), (-0+0j), (0.0625-0j), (-0+0j), (0.0625-0j), (-0+0j), (-0.062499999999999986+0j), (-0+0j), (0.0625-0j), (-0+0j), (0.062499999999999986-0j), (-0+0j), (-0.062499999999999986+0j), (-0+0j), (0.06249999999999998-0j), (-0+0j), (0.0625-0j), (-0+0j), (0.062499999999999986-0j), (-0+0j), (0.062499999999999986-0j), (-0+0j), (-0.06249999999999998+0j), (-0+0j), (-0.062499999999999986+0j), (-0+0j), (-0.06249999999999998+0j), (-0+0j), (0.06249999999999998-0j), (-0+0j), (-0.06249999999999997+0j), (-0+0j), (-0.062499999999999986+0j), (-0+0j), (-0.062499999999999986+0j), (-0+0j), (-0.062499999999999986+0j), (-0+0j), (0.06249999999999998-0j), (-0+0j), (-0.062499999999999986+0j), (-0+0j), (-0.06249999999999998+0j), (-0+0j), (0.06249999999999998-0j), (-0+0j), (-0.06249999999999997+0j), (-0+0j), (0.062499999999999986-0j), (-0+0j), (0.06249999999999998-0j), (-0+0j), (0.06249999999999998-0j), (-0+0j), (-0.06249999999999997+0j), (-0+0j), (-0.06249999999999998+0j), (-0+0j), (-0.06249999999999997+0j), (-0+0j), (0.06249999999999997-0j), (-0+0j), (-0.06249999999999996+0j), (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), 0j, (-0+0j), (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, (-0+0j), -0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j, -0j, 0j])