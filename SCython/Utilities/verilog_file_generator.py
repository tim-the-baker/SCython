from SCython.Circuits import mux_adders
from SCython.SNG import PCC
from SCython.Utilities.verilog_module_generator import *

###########################
#   CeMux Related Files   #
###########################
def gen_verilog_cemux_file_string(precision, weights, bipolar=True, pcc_array_ID="", tree_ID="", output_ID=""):
    cemux = mux_adders.CeMux(tree_height=precision, weights=weights, pcc=PCC.Comparator, bipolar=bipolar)
    invert_list = cemux.quant_norm_weights[cemux.quant_norm_weights != 0] < 0  # list for which inputs have neg weights
    pcc_input_size = len(invert_list)

    # begin the file
    file_string = ""
    file_string += "`timescale 1 ns / 1 ns\n\n"
    # generate verilog code for the filter's memory
    file_string += gen_verilog_filter_memory(cemux.quant_norm_weights, pcc_precision=precision,
                                             rns_precision=precision, gated=True)

    # generate verilog code for a counter module that is used as the VDC RNS and for the mux select input
    file_string += gen_verilog_counter_with_rev(precision)

    # generate verilog for a comparator module and for a comparator module that inverts the RNS input.
    file_string += gen_verilog_comparator(precision, bipolar, invert_output=False)
    file_string += gen_verilog_comparator(precision, bipolar, invert_output=True)

    # generate verilog for an array of comparators used to generate CeMux's inputs.
    file_string += gen_verilog_comparator_array(precision, share_r=True, invert_list=invert_list, full_correlation=True,
                                                ID=pcc_array_ID)

    # generate verilog for cemux's hardwired tree
    file_string += gen_verilog_hardwired_mux_tree(cemux.quant_norm_weights, cemux.vhdl_wire_map,
                                                  tree_height=precision, ID=tree_ID)

    # generate verilog for the output counter
    file_string += gen_verilog_output_counter(precision, bipolar, ID=output_ID)

    # generate verilog for the filtercore and top-level CeMux module
    file_string += gen_verilog_cemux_filtercore(precision, pcc_input_size, pcc_array_ID, tree_ID, output_ID)
    file_string += gen_verilog_cemux_toplevel(precision, pcc_input_size)
    return file_string
