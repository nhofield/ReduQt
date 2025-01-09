import os
import numpy as np

programs_list = []

current_file_directory = os.path.dirname(os.path.abspath(__file__))
dir_mutants = os.path.join(current_file_directory, "Mutants/")
dir_mutants_random = os.path.join(current_file_directory, "Mutants_random/")


parent_directory = os.path.dirname(current_file_directory)
dir_study_subjects = os.path.join(parent_directory, "Experiment1", "study_subjects/")
dir_study_subjects_random = os.path.join(parent_directory, "Experiment1", "study_subjects_random/")


# var programs
num_programs_var = 12
lower_thresh_var = 2
upper_thresh_var = 8
num_vars = 4
ids_var = [1, 2, 2, 1, 3, 1, 1]

def give_random_num(index, random_range):
    return np.random.choice( range(random_range) )


for qubit_index in range(lower_thresh_var, upper_thresh_var + 1):
    for program_index in range( ids_var[qubit_index-lower_thresh_var] ):
        for var_index in range( num_vars ):

            program_name = r"var_" + f"q{qubit_index}_{program_index}{var_index}.init"
            program_name_A = r"var_" + f"q{qubit_index}_{program_index}{var_index}_A.init"

            programs_list.append(program_name)
            programs_list.append(program_name_A)

# aamp programs
lower_thresh_aamp = 6
upper_thresh_aamp = 9
ids_number = 12

for qubit_index in range(lower_thresh_aamp, upper_thresh_aamp + 1):
    for program_index in range( ids_number ):

        program_name = r"aamp_" + f"q{qubit_index}_{program_index}.init"
        program_name_A = r"aamp_" + f"q{qubit_index}_{program_index}_A.init"

        programs_list.append(program_name)
        programs_list.append(program_name_A)

# qwalk programs
lower_thresh_qwalk = 3
upper_thresh_qwalk = 5
ids_number = 13

for qubit_index in range(lower_thresh_qwalk, upper_thresh_qwalk + 1):
    for program_index in range( ids_number ):

        program_name = r"qwalk_" + f"q{qubit_index}_{program_index}.init"
        program_name_A = r"qwalk_" + f"q{qubit_index}_{program_index}_A.init"

        programs_list.append(program_name)
        programs_list.append(program_name_A)

# gs programs
lower_thresh_gs = 3
upper_thresh_gs = 16

for qubit_index in range(lower_thresh_gs, upper_thresh_gs + 1):

    program_name = r"gs_" + f"q{qubit_index}.init"
    program_name_A = r"gs_" + f"q{qubit_index}_A.init"

    programs_list.append(program_name)
    programs_list.append(program_name_A)



mutant_operators = ["qc.x", "qc.z", "qc.ry"]
count = 0
number_of_ry = 3
number_of_repetitions = 30

if not os.path.exists(current_file_directory + r"/Mutants"):
    # Create the Mutants directory if it does not exist
    os.mkdir(current_file_directory + r"/Mutants")

# make repetition directories for factoring
for repetition_index in range(number_of_repetitions):
    if not os.path.exists(dir_mutants + f"/Mutants_repetition_{repetition_index}"):
        # Create the Mutants directory if it does not exist
        os.mkdir(dir_mutants + f"/Mutants_repetition_{repetition_index}")


if not os.path.exists(current_file_directory + r"/Mutants_random"):
    # Create the Mutants directory if it does not exist
    os.mkdir(current_file_directory + r"/Mutants_random")

# make repetition directories for random
for repetition_index in range(number_of_repetitions):
    if not os.path.exists(dir_mutants_random + f"/Mutants_repetition_{repetition_index}"):
        # Create the Mutants directory if it does not exist
        os.mkdir(dir_mutants_random + f"/Mutants_repetition_{repetition_index}")

for repetition_index in range(number_of_repetitions):

    for program_index in range(len(programs_list)):
        # Define the directory path
        dir_path = os.path.join(dir_mutants + f"Mutants_repetition_{repetition_index}/", programs_list[program_index][:-5])

        # Define the directory path for random
        dir_path_random = os.path.join(dir_mutants_random + f"Mutants_repetition_{repetition_index}/", programs_list[program_index][:-5])

        # Check if the directory does not exist
        if r"_A" not in programs_list[program_index]:
            if not os.path.exists(dir_path):
                # Create the directory if it does not exist
                os.mkdir(dir_path)

        # Check if the directory for random does not exist
        if r"_A" not in programs_list[program_index]:
            if not os.path.exists(dir_path_random):
                # Create the directory if it does not exist
                os.mkdir(dir_path_random)

for repetition_index in range( number_of_repetitions ):
    print("Repetition dir =", repetition_index)

    for program_index in range(0, len(programs_list), 2 ):

        # save default_init
        with open( dir_study_subjects + programs_list[program_index], "r" ) as infile:
            default_lines_init = infile.readlines()

        # save default_py
        with open( dir_study_subjects + programs_list[program_index][:-5] + ".py", "r" ) as infile:
            default_lines_py = infile.readlines()

        # save A_init
        with open( dir_study_subjects + programs_list[program_index + 1], "r" ) as infile:
            default_lines_init_A = infile.readlines()

        # save A_py
        with open( dir_study_subjects + programs_list[program_index + 1][:-5] + ".py", "r" ) as infile:
            default_lines_py_A = infile.readlines()


        # save A_init random
        with open( dir_study_subjects_random + programs_list[program_index + 1], "r" ) as infile:
            default_lines_init_A_random = infile.readlines()

        # save A_py random
        with open( dir_study_subjects_random + programs_list[program_index + 1][:-5] + ".py", "r" ) as infile:
            default_lines_py_A_random = infile.readlines()



        num_qubits_lines = default_lines_init[2]
        num_qubits = int( num_qubits_lines[num_qubits_lines.index("=") + 1:] )

        #print(programs_list[program_index])
        #print(programs_list[program_index + 1])
        #print(programs_list[program_index][:-5] + ".py")
        #print(programs_list[program_index + 1][:-5] + ".py")




        for num_qubit_index in range( 3 ):

            # mutant programs for x,z mutant generators
            add_root = r"/" + programs_list[program_index][:-5] + r"/"
            for mutant_generator_index in range( 2 ):

                # factoring
                # paths for 1 mutant (1 for default and 1 for _A), both init and .py file paths
                M1_ext = f"_M{mutant_generator_index + 1}_{num_qubit_index}"
                mutant_default_init_M1   = dir_mutants + f"Mutants_repetition_{repetition_index}" + add_root + programs_list[program_index][:-5] + M1_ext + r".init"
                mutant_default_init_A_M1 = dir_mutants + f"Mutants_repetition_{repetition_index}" + add_root + programs_list[program_index + 1][:-5] + M1_ext + r".init"
                mutant_default_py_M1     = dir_mutants + f"Mutants_repetition_{repetition_index}" + add_root + programs_list[program_index][:-5] + M1_ext + r".py"
                mutant_default_py_A_M1   = dir_mutants + f"Mutants_repetition_{repetition_index}" + add_root + programs_list[program_index + 1][:-5] + M1_ext + r".py"
                mutant_progs = [mutant_default_init_M1, mutant_default_init_A_M1, mutant_default_py_M1, mutant_default_py_A_M1]

                # random
                # random paths for 1 mutant (1 for default and 1 for _A), both init and .py file paths
                M1_ext_random = f"_M{mutant_generator_index + 1}_{num_qubit_index}"
                mutant_default_init_M1_random   = dir_mutants_random + f"Mutants_repetition_{repetition_index}" + add_root + programs_list[program_index][:-5] + M1_ext + r".init"
                mutant_default_init_A_M1_random = dir_mutants_random + f"Mutants_repetition_{repetition_index}" + add_root + programs_list[program_index + 1][:-5] + M1_ext + r".init"
                mutant_default_py_M1_random     = dir_mutants_random + f"Mutants_repetition_{repetition_index}" + add_root + programs_list[program_index][:-5] + M1_ext + r".py"
                mutant_default_py_A_M1_random   = dir_mutants_random + f"Mutants_repetition_{repetition_index}" + add_root + programs_list[program_index + 1][:-5] + M1_ext + r".py"
                mutant_progs_random = [mutant_default_init_M1_random, mutant_default_init_A_M1_random, mutant_default_py_M1_random, mutant_default_py_A_M1_random]


                # default_lines_init
                mutant_default_lines_init = default_lines_init
                mutant_default_lines_init[19] = f"Mutation={mutant_generator_index + 1}|{num_qubit_index}\n"
                mutant_default_lines_init[1] = r"root=" + dir_mutants + f"/Mutants_repetition_{repetition_index}/" + programs_list[program_index][:-5] + "/" + programs_list[program_index][:-5] + M1_ext + ".py" + "\n"

                # default_lines_init_A
                mutant_default_lines_init_A = default_lines_init_A
                mutant_default_lines_init_A[19] = f"Mutation={mutant_generator_index + 1}|{num_qubit_index}\n"
                mutant_default_lines_init_A[1] = r"root=" + dir_mutants + f"/Mutants_repetition_{repetition_index}/" + programs_list[program_index][:-5] + "/" + programs_list[program_index + 1][:-5] + M1_ext + ".py" + "\n"

                # default_lines_init_A random
                mutant_default_lines_init_A_random = default_lines_init_A_random
                mutant_default_lines_init_A_random[19] = f"Mutation={mutant_generator_index + 1}|{num_qubit_index}\n"
                mutant_default_lines_init_A_random[1] = r"root=" + dir_mutants + f"/Mutants_repetition_{repetition_index}/" + programs_list[program_index][:-5] + "/" + programs_list[program_index + 1][:-5] + M1_ext + ".py" + "\n"


                rand_assign_number = give_random_num(num_qubit_index, num_qubits)

                # default_lines_py
                mutant_default_lines_py = default_lines_py
                last_line = mutant_default_lines_py[-1]
                mutant_default_lines_py = mutant_default_lines_py[:-1] + \
                                          [f"    {mutant_operators[mutant_generator_index]}({rand_assign_number})\n"] + \
                                          [last_line]
                # -------------
                # | Factoring |
                # ------------
                # default_lines_py_A
                mutant_default_lines_py_A = default_lines_py_A
                for line in range( len(mutant_default_lines_py_A) ):
                    if mutant_default_lines_py_A[line][:17] == "    # measurement":
                        keep_line = line

                last_lines = mutant_default_lines_py_A[keep_line:]

                mutant_default_lines_py_A = mutant_default_lines_py_A[:keep_line] + \
                [f"    {mutant_operators[mutant_generator_index]}({rand_assign_number})\n"] + \
                last_lines


                mutant_files = [mutant_default_lines_init, mutant_default_lines_init_A, \
                                mutant_default_lines_py, mutant_default_lines_py_A]



                for mutant_prog_index in range( len(mutant_progs) ):
                    with open( mutant_progs[mutant_prog_index], "w" ) as outfile:
                        for line_write in mutant_files[mutant_prog_index]:
                            outfile.write( line_write )

                # ---------
                #| random |
                # --------
                # default_lines_py_A_random
                mutant_default_lines_py_A_random = default_lines_py_A_random
                for line in range( len(mutant_default_lines_py_A_random) ):
                    if mutant_default_lines_py_A_random[line][:17] == "    # measurement":
                        keep_line_random = line

                last_lines_random = mutant_default_lines_py_A_random[keep_line_random:]

                mutant_default_lines_py_A_random = mutant_default_lines_py_A_random[:keep_line_random] + \
                [f"    {mutant_operators[mutant_generator_index]}({rand_assign_number})\n"] + \
                last_lines_random


                mutant_files_random = [mutant_default_lines_init, mutant_default_lines_init_A_random, \
                                mutant_default_lines_py, mutant_default_lines_py_A_random]



                for mutant_prog_index in range( len(mutant_progs_random) ):
                    with open( mutant_progs_random[mutant_prog_index], "w" ) as outfile:
                        for line_write in mutant_files_random[mutant_prog_index]:
                            outfile.write( line_write )

        ry_counter = 1

        for position_index in range( 3 ):

            rand_location = give_random_num(position_index, num_qubits)

            for angle_index in range(1, number_of_ry + 1):

                # mutant programs for ry mutant generators
                add_root = r"/" + programs_list[program_index][:-5] + r"/"

                # factoring
                # paths for 1 mutant (1 for default and 1 for _A), both init and .py file paths
                M1_ext = f"_M{2 + 1}_{ry_counter - 1}"
                mutant_default_init_M1   = dir_mutants + f"Mutants_repetition_{repetition_index}" + add_root + programs_list[program_index][:-5] + M1_ext + r".init"
                mutant_default_init_A_M1 = dir_mutants + f"Mutants_repetition_{repetition_index}" + add_root + programs_list[program_index + 1][:-5] + M1_ext + r".init"
                mutant_default_py_M1     = dir_mutants + f"Mutants_repetition_{repetition_index}" + add_root + programs_list[program_index][:-5] + M1_ext + r".py"
                mutant_default_py_A_M1   = dir_mutants + f"Mutants_repetition_{repetition_index}" + add_root + programs_list[program_index + 1][:-5] + M1_ext + r".py"
                mutant_progs = [mutant_default_init_M1, mutant_default_init_A_M1, mutant_default_py_M1, mutant_default_py_A_M1]

                # random
                # random paths for 1 mutant (1 for default and 1 for _A), both init and .py file paths
                M1_ext_random = f"_M{2 + 1}_{ry_counter - 1}"
                mutant_default_init_M1_random   = dir_mutants_random + f"Mutants_repetition_{repetition_index}" + add_root + programs_list[program_index][:-5] + M1_ext + r".init"
                mutant_default_init_A_M1_random = dir_mutants_random + f"Mutants_repetition_{repetition_index}" + add_root + programs_list[program_index + 1][:-5] + M1_ext + r".init"
                mutant_default_py_M1_random     = dir_mutants_random + f"Mutants_repetition_{repetition_index}" + add_root + programs_list[program_index][:-5] + M1_ext + r".py"
                mutant_default_py_A_M1_random   = dir_mutants_random + f"Mutants_repetition_{repetition_index}" + add_root + programs_list[program_index + 1][:-5] + M1_ext + r".py"
                mutant_progs_random = [mutant_default_init_M1_random, mutant_default_init_A_M1_random, mutant_default_py_M1_random, mutant_default_py_A_M1_random]


                # default_lines_init
                mutant_default_lines_init = default_lines_init
                mutant_default_lines_init[19] = f"Mutation={2 + 1}|{ry_counter}\n"
                mutant_default_lines_init[1] = r"root=" + dir_mutants + f"/Mutants_repetition_{repetition_index}/" + programs_list[program_index][:-5] + "/" + programs_list[program_index][:-5] + M1_ext + ".py" + "\n"

                # default_lines_init_A
                mutant_default_lines_init_A = default_lines_init_A
                mutant_default_lines_init_A[19] = f"Mutation={2 + 1}|{ry_counter}\n"
                mutant_default_lines_init_A[1] = r"root=" + dir_mutants + f"/Mutants_repetition_{repetition_index}/" + programs_list[program_index][:-5] + "/" + programs_list[program_index + 1][:-5] + M1_ext + ".py" + "\n"

                # default_lines_init_A random
                mutant_default_lines_init_A_random = default_lines_init_A_random
                mutant_default_lines_init_A_random[19] = f"Mutation={2 + 1}|{ry_counter}\n"
                mutant_default_lines_init_A_random[1] = r"root=" + dir_mutants + f"/Mutants_repetition_{repetition_index}/" + programs_list[program_index][:-5] + "/" + programs_list[program_index + 1][:-5] + M1_ext + ".py" + "\n"



                # default_lines_py
                mutant_default_lines_py = default_lines_py
                last_line = mutant_default_lines_py[-1]
                mutant_default_lines_py = mutant_default_lines_py[:-1] + \
                                          [f"    {mutant_operators[2]}({angle_index/10*2},{int( round(num_qubits/2) )})\n"] + \
                                          [last_line]

                # draw random angle between 0 and 2pi
                random_angle = np.random.uniform( 0, 2*np.pi )

                # ------------
                # | Factoring |
                # ------------
                # default_lines_py_A
                mutant_default_lines_py_A = default_lines_py_A
                for line in range( len(mutant_default_lines_py_A) ):
                    if mutant_default_lines_py_A[line][:17] == "    # measurement":
                        keep_line = line

                last_lines = mutant_default_lines_py_A[keep_line:]


                mutant_default_lines_py_A = mutant_default_lines_py_A[:keep_line] + \
                [f"    {mutant_operators[2]}({random_angle},{rand_location})\n"] + \
                last_lines

                mutant_files = [mutant_default_lines_init, mutant_default_lines_init_A, \
                                mutant_default_lines_py, mutant_default_lines_py_A]


                for mutant_prog_index in range( len(mutant_progs) ):
                    with open( mutant_progs[mutant_prog_index], "w" ) as outfile:
                        for line_write in mutant_files[mutant_prog_index]:
                            outfile.write( line_write )
                # ------------
                # |  Random  |
                # ------------
                # default_lines_py_A_random
                mutant_default_lines_py_A_random = default_lines_py_A_random
                for line in range( len(mutant_default_lines_py_A_random) ):
                    if mutant_default_lines_py_A_random[line][:17] == "    # measurement":
                        keep_line_random = line

                last_lines_random = mutant_default_lines_py_A_random[keep_line_random:]


                mutant_default_lines_py_A_random = mutant_default_lines_py_A_random[:keep_line_random] + \
                [f"    {mutant_operators[2]}({random_angle},{rand_location})\n"] + \
                last_lines_random

                mutant_files_random = [mutant_default_lines_init, mutant_default_lines_init_A_random, \
                                mutant_default_lines_py, mutant_default_lines_py_A_random]


                for mutant_prog_index in range( len(mutant_progs_random) ):
                    with open( mutant_progs_random[mutant_prog_index], "w" ) as outfile:
                        for line_write in mutant_files_random[mutant_prog_index]:
                            outfile.write( line_write )

                ry_counter += 1
