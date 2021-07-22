import numpy as np
from SCython.Utilities import verilog_file_generator as vfg

if __name__ == '__main__':
    # TODO Update the following four parameters
    precisions = 8
    coefficients = np.array([1, -1, 1, -1, 1, 0, 1, -1, -1, 0, 0])  # filter coefficients
    # directory is a string representing where you want to store the verilog file.
    directory = r"C:\Users\Tim\PycharmProjects\stochastic-computing\Experiments\000_Susan_related_July-7-21\data\verilog\scratch_tests"

    # file_ID is a unique string that you can make up so that the verilog files have different names. You'll need to enter this file_ID in the makefile.
    # for instance, file_ID = "FIR1" can be used for the first single stage FIR filter we synthesize
    # and, file_ID = "IFIR1_S1" can be used for the first stage of the first IFIR filter we synthesize
    # and, file_ID = "IFIR1_S2" can be used for the second stage of the first IFIR filter we synthesize
    # Note that file_ID does *not* need to specify parameters like precision.
    file_ID = "FIR1"

    # dont worry about the following ID parameters for now.
    pcc_array_ID = ""
    tree_ID = ""
    output_ID = ""

    # the file name of the verilog file lists relevant parameters:
    # 2^n is the intended SN length. n determines the size of the RNS and mux select input generator.
    # q is the bitwidth of the PCCs (i.e., comparators). Often we just set q = n, but setting q < n lowers area in exchange for accuracy loss.
    file_string = vfg.gen_verilog_cemux_file_string(precision, coefficients, pcc_array_ID=pcc_array_ID, tree_ID=tree_ID,
                                                    output_ID=output_ID)
    file_name = rf"{directory}\cemux_{file_ID}_n{precision}_q{precision}.v"
    with open(file_name, "w") as f:
        f.write(file_string)

