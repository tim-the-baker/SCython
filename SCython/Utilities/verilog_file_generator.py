from SCython.Circuits import stream_adders
from SCython.SNG import PCC
from SCython.Utilities.verilog_module_generator import *

#################################
#   CeMux/CeMaj Related Files   #
#################################
def gen_verilog_cemux_file_string(precision, weights, bipolar=True, pcc_array_ID="", tree_ID="", output_ID="", latency_factor=None):
    cemux = stream_adders.CeMux(tree_height=latency_factor, weights=weights, data_pcc=PCC.Comparator(precision), bipolar=bipolar)
    invert_list = cemux.quant_norm_weights[cemux.quant_norm_weights != 0] < 0  # list for which inputs have neg weights
    pcc_input_size = len(invert_list)

    # begin the file
    file_string = ""
    file_string += "`timescale 1 ns / 1 ns\n\n"
    # generate verilog code for the filter's memory
    file_string += gen_verilog_filter_memory(cemux.quant_norm_weights, pcc_precision=precision,
                                             rns_precision=latency_factor, gated=True)

    # generate verilog code for a counter module that is used as the VDC RNS and for the mux select input
    file_string += gen_verilog_counter_with_rev(latency_factor)

    # generate verilog for a comparator module and for a comparator module that inverts the RNS input.
    file_string += gen_verilog_comparator(precision, bipolar, invert_output=False)
    file_string += gen_verilog_comparator(precision, bipolar, invert_output=True)

    # generate verilog for an array of comparators used to generate CeMux's inputs.
    file_string += gen_verilog_comparator_array(precision, share_r=True, invert_list=invert_list, full_correlation=True,
                                                ID=pcc_array_ID)

    # generate verilog for cemux's hardwired tree
    file_string += gen_verilog_hardwired_mux_tree(cemux.quant_norm_weights, cemux.vhdl_wire_map,
                                                  tree_height=latency_factor, ID=tree_ID)

    # generate verilog for the output counter
    file_string += gen_verilog_output_counter(latency_factor, bipolar, ID=output_ID)

    # generate verilog for the filtercore and top-level CeMux module
    file_string += gen_verilog_cemux_filtercore(precision, pcc_input_size, pcc_array_ID, tree_ID, output_ID, latency_factor)
    file_string += gen_verilog_cemux_toplevel(precision, pcc_input_size, latency_factor=latency_factor)
    return file_string


def gen_verilog_cemaj_file_string(precision, weights, bipolar=True, pcc_array_ID="", tree_ID="", output_ID="", latency_factor=None):
    latency_factor = precision if latency_factor is None else latency_factor
    assert latency_factor >= precision, "Latency factor must be >= precision"
    cemux = stream_adders.CeMux(tree_height=latency_factor, weights=weights, data_pcc=PCC.Comparator(precision), bipolar=bipolar)
    invert_list = cemux.quant_norm_weights[cemux.quant_norm_weights != 0] < 0  # list for which inputs have neg weights
    pcc_input_size = len(invert_list)

    # begin the file
    file_string = ""
    file_string += "`timescale 1 ns / 1 ns\n\n"
    # generate verilog code for the filter's memory
    file_string += gen_verilog_filter_memory(cemux.quant_norm_weights, pcc_precision=precision,
                                             rns_precision=latency_factor, gated=True)

    # generate verilog code for a counter module that is used as the VDC RNS and for the mux select input
    file_string += gen_verilog_counter_with_rev(latency_factor)

    # generate verilog for WBG P1, WBG P2 and for WBG P2 that inverts the output bit.
    file_string += gen_verilog_wbg_p1(precision)
    file_string += gen_verilog_wbg_p2(precision, bipolar, invert_output=False)
    file_string += gen_verilog_wbg_p2(precision, bipolar, invert_output=True)

    # generate verilog for an array of comparators used to generate CeMux's inputs.
    file_string += gen_verilog_wbg_array(precision, share_r=True, invert_list=invert_list, full_correlation=True,
                                                ID=pcc_array_ID)

    # generate verilog for cemux's hardwired tree
    file_string += gen_verilog_hardwired_maj_tree(cemux.quant_norm_weights, cemux.vhdl_wire_map,
                                                  tree_height=latency_factor, ID=tree_ID)

    # generate verilog for the output counter
    file_string += gen_verilog_output_counter(latency_factor, bipolar, ID=output_ID)

    # generate verilog for the filtercore and top-level CeMux module
    file_string += gen_verilog_cemaj_filtercore(precision, pcc_input_size, pcc_array_ID, tree_ID, output_ID, latency_factor)
    file_string += gen_verilog_cemux_toplevel(precision, pcc_input_size, latency_factor)
    return file_string
