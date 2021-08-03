# TODO: Account for coefficients who weights are zero. Important for when flattened synthesis is *not* used.
import numpy as np
from SCython.SNG import RNS, PCC
from SCython.Circuits import stream_adders

####################################
#   Random Number Source Modules   #
####################################
def gen_verilog_counter_with_rev(precision):
    """
    This verilog module is a counter that increments by 1 each clock cycle. It has two outputs, the current counter state
    and the current counter state bitwise reversed. This module is useful in CeMux as it implements the mux select input
    generator and the VDC RNS in one.
    :param precision: bit-width of counter.
    :return: verilog code for the module in string form.
    """
    n = precision
    file_string = f"module counter (\n" \
                  f"\tinput  clock, reset,\n" \
                  f"\toutput logic [{n-1}:0] state,\n" \
                  f"\toutput logic [{n-1}:0] rev_state\n);\n" \
                  f"\talways_comb begin\n" \
                  f"\t\tfor (int i = 0; i < {n}; i+=1) begin\n" \
                  f"\t\t\trev_state[i] = state[{n-1}-i];\n" \
                  f"\t\tend\n" \
                  f"\tend\n\n" \
                  f"\talways_ff @(posedge clock) begin\n" \
                  f"\t\tif (reset == 1) state <= 'b0; else\n" \
                  f"\t\t                state <= state + 1;\n" \
                  f"\tend\n" \
                  f"endmodule\n\n\n"
    return file_string


######################################
#   Probability Conversion Modules   #
######################################
def gen_verilog_comparator(precision, bipolar, invert_output):
    """
    Generates verilog code for a comparator module
    :param precision: bit-width of comparator PCC
    :param bipolar: whether this PCC generates a bipolar SN or not. If this PCC is used to generate a bipolar SN, then
    the MSB of the control input is inverted before comparison.
    :param invert_output: whether the output of the comparator should be inverted. This is useful in some MUX circuits.
    :return: verilog code for the module in string form.
    """
    n = precision
    module_name = "compara_inv" if invert_output else "compara"
    file_string = f"module {module_name} (\n" \
                  f"\tinput  logic [{n-1}:0] r,\n" \
                  f"\tinput  logic [{n-1}:0] p,\n" \
                  f"\toutput logic       SN\n);\n"

    if bipolar:  # invert MSB of control input
        p_str = "p2"
        file_string += f"\tlogic [{n - 1}:0] p2;\n"
        file_string += f"\tassign p2[{n-1}] = ~p[{n-1}];\n" \
                       f"\tassign p2[{n-2}:0] = p[{n-2}:0];\n"
    else:
        p_str = "p"

    if invert_output:
        file_string += f"\tassign SN = ~(r < {p_str});\n"
    else:
        file_string += f"\tassign SN = r < {p_str};\n"

    file_string += f"endmodule\n\n\n"
    return file_string


def gen_verilog_comparator_array(precision, share_r, invert_list, full_correlation, ID=""):
    """
    Generates verilog code for an array of comparators.
    :param precision: bit-width of the comparators.
    :param share_r: Boolean value that is True when the comparators share a random number input (must be True for now)
    :param invert_list: list of Boolean values where invert_list[i]=True means that the i-th comparator should invert its
    output while invert_list[i]=False means that the i-th comparator should not invert its output.
    :param full_correlation: whether the SNs should be generate in a manner such that SCC(X1,X2) = +1 for all X1,X2. Only
    works when share_r is True.
    :param ID: identification number or string to append to the end of the module's name. This is used when you need more than
    one logically distinct comparator arrays in a single verilog file.
    :return: verilog code for the module in string form.
    """
    assert share_r, "Only sharing R is implemented at the moment."
    assert not full_correlation or share_r, "When full_correlation=True, share_r must also be True"
    n = precision
    M = len(invert_list)  # M is the number of control inputs to the comparator array and the number of outputs.

    file_string = f"module compara_array{ID} (\n" \
                  f"\tinput  logic [{n-1}:0] in [{M-1}:0],\n" \
                  f"\tinput  logic [{n-1}:0] r,\n" \
                  f"\toutput logic       SNs [{M-1}:0]\n);\n"

    if full_correlation:
        # Handle inverted WBG P1 with inverted R input if corr_adj is used
        file_string += f"\tlogic [{n-1}:0] r_inv;\n" \
                       f"\tassign r_inv = ~r;\n\n"
        r_neg = "r_inv"
        spacing = " "*4
    else:
        r_neg = "r"
        spacing = ""

    for j in range(M):
        if invert_list[j] == 1:
            file_string += f"\tcompara_inv comp{j}(.r({r_neg}), .p(in[{j}]), .SN(SNs[{j}]));\n"
        else:
            file_string += f"\tcompara     comp{j}(.r(r), {spacing}.p(in[{j}]), .SN(SNs[{j}]));\n"

    file_string += f"endmodule\n\n\n"

    return file_string


def gen_verilog_wbg_p1(precision):
    n = precision
    file_string = f"module wbg_p1  (\n" \
                  f"\tinput  logic [{n-1}:0]     r,\n" \
                  f"\toutput logic [{n-1}:0]     out\n);\n"
    inverts = ""
    for idx in range(n-1, -1, -1):  # loop backwards
        file_string += f"\tassign out[{idx}] = r[{idx}]{inverts};\n"
        inverts += f" & ~r[{idx}]"
    file_string += "endmodule\n\n\n"
    return file_string


def gen_verilog_wbg_p2(precision, bipolar, invert_output):
    n = precision
    module_name = "wbg_p2_inv" if invert_output else "wbg_p2"
    file_string = f"module {module_name} (\n" \
                  f"\tinput  logic [{n-1}:0]     w,\n" \
                  f"\tinput  logic [{n-1}:0]     p,\n" \
                  f"\toutput logic           out\n);\n"

    if bipolar:  # invert MSB of control input
        p_str = "p2"
        file_string += f"\tlogic [{n - 1}:0] p2;\n"
        file_string += f"\tassign p2[{n-1}] = ~p[{n-1}];\n" \
                       f"\tassign p2[{n-2}:0] = p[{n-2}:0];\n"
    else:
        p_str = "p"

    file_string += f"\tlogic [{n-1}:0] u;\n"
    file_string += f"\tassign u = w & {p_str};\n"

    if invert_output:
        file_string += f"\tassign out = ~(|u);\n"
    else:
        file_string += f"\tassign out = |u;\n"

    file_string += f"endmodule\n\n\n"
    return file_string


def gen_verilog_wbg_array(precision, share_r, invert_list, full_correlation, ID=""):
    assert share_r, "Only sharing R is implemented at the moment."
    n = precision
    M = len(invert_list)

    file_string = f"module wbg_array{ID}  (\n" \
                  f"\tinput  logic [{n - 1}:0]     in  [{M-1}:0],\n" \
                  f"\tinput  logic [{n - 1}:0]     r,\n" \
                  f"\toutput logic           SNs  [{M-1}:0]\n);\n"

    file_string += f"\tlogic [{n-1}:0] w;\n"
    file_string += f"\twbg_p1 wbgp1(.r(r), .out(w));\n\n"

    if full_correlation:
        # Handle inverted WBG P1 with inverted R input if corr_adj is used
        file_string += f"\tlogic [{n-1}:0] w_neg;\n"
        file_string += f"\twbg_p1 wbgp1_inv(.r(~r), .out(w_neg));\n\n"
        w_neg = "w_neg"
        spacing = " " * 4
    else:
        w_neg = "w"
        spacing = ""

    for j in range(M):
        if invert_list[j] == 1:
            file_string += f"\twbg_p2_inv wbg{j}(.w({w_neg}), .p(in[{j}]), .out(SNs[{j}]));\n"
        else:
            file_string += f"\twbg_p2     wbg{j}(.w(w), {spacing}.p(in[{j}]), .out(SNs[{j}]));\n"

    file_string += f"endmodule\n\n\n"
    return file_string


##########################################
#   FIR Filter and Filterbank Modules   #
##########################################
def gen_verilog_filter_memory(quant_norm_coefs, pcc_precision, rns_precision, gated, ID=""):
    """
    Generate verilog code for an FIR filter's memory (control) module.
    :param quant_norm_coefs: FIR filter's quantized, normalized coefficients
    :param pcc_precision: bit-width of the filter's PCCs
    :param rns_precision: bit-width of the filter's RNS
    :param gated: whether to clock gate the flip flops (keep this as True for best performance)
    :param ID: identification number or string to append to the end of the module's name. This is used when you need more than
    one logically distinct comparator arrays in a single verilog file.
    :return: verilog code for the module in string form.
    """
    rns_n, pcc_n = rns_precision, pcc_precision

    # The filter's memory input size is the number of weights minus the number of 0 weights at end of filter.
    memory_size = len(quant_norm_coefs)
    while quant_norm_coefs[memory_size - 1] == 0:
        memory_size -= 1
    adjusted_quant_norm_weights = quant_norm_coefs[0:memory_size]

    # the output of filter memory is the input to PCC array. Only output the memory elements who have nonzero weights
    output_size = np.sum(adjusted_quant_norm_weights != 0)

    file_string = f"module memory{ID} (\n" \
                  f"\tinput  clock, reset,\n" \
                  f"\tinput  logic [{pcc_n - 1}:0] in,\n" \
                  f"\tinput  logic [{rns_n - 1}:0] count,\n" \
                  f"\toutput logic [{pcc_n - 1}:0] out [{output_size - 1}:0]\n);\n"
    file_string += f"\tlogic [{pcc_n - 1}:0] registers [{memory_size - 1}:0];\n"

    out_idx = 0
    for mem_idx in range(len(quant_norm_coefs)):
        if quant_norm_coefs[mem_idx] != 0:
            file_string += f"\tassign out[{out_idx}] = registers[{mem_idx}];\n"
            out_idx += 1
    assert out_idx == output_size

    # Set up the sequential logic
    if not gated:
        count_string = '1' * rns_n
        file_string += f"\n\talways_ff @(posedge clock) begin\n" \
                       f"\t\tif (reset == 1) begin\n" \
                       f"\t\t\tfor(int i=0; i<{memory_size}; i=i+1) registers[i] <= 'b0;\n" \
                       f"\t\tend\n" \
                       f"\t\telse begin\n" \
                       f"\t\t\tif (count == {rns_n}'b{count_string}) begin\n" \
                       f"\t\t\t\tregisters[{memory_size- 1}:1] <= registers[{memory_size- 2}:0];\n" \
                       f"\t\t\t\tregisters[0] <= in;\n" \
                       f"\t\t\tend\n" \
                       f"\t\tend\n" \
                       f"\tend\n" \
                       f"endmodule\n\n\n"
    else:
        file_string += f"\n\talways_ff @(posedge clock) begin\n" \
                       f"\t\tif (reset == 1) begin\n" \
                       f"\t\t\tfor(int i=0; i<{memory_size}; i=i+1) registers[i] <= 'b0;\n" \
                       f"\t\tend\n" \
                       f"\t\telse begin\n" \
                       f"\t\t\tregisters[{memory_size-1}:1] <= registers[{memory_size-2}:0];\n" \
                       f"\t\t\tregisters[0] <= in;\n" \
                       f"\t\tend\n" \
                       f"\tend\n" \
                       f"endmodule\n\n\n"

    return file_string


#########################
#   MUX/MAJ Adder Modules   #
#########################
# TODO Doc string
def gen_verilog_hardwired_mux_tree(quant_norm_weights, hdl_wire_map, tree_height, ID=""):
    M = (quant_norm_weights != 0).sum()
    m = tree_height

    file_string = f"module hw_tree{ID}  (\n" \
                  f"\tinput  logic           data_SNs  [{M-1}:0],\n" \
                  f"\tinput  logic [{m-1}:0]     mux_select,\n" \
                  f"\toutput logic           out_SN\n);\n"

    # Initialize an array of wires for every level in the mux tree. (for internal signals)
    num_mux = int(2 ** m)
    for level in range(m):
        num_mux //= 2
        file_string += f"\tlogic level{level}  [{num_mux - 1}:0];\n"

    # Assign the just initialized wires
    num_mux = int(2 ** m)
    for level in range(m):
        file_string += "\n"
        num_mux //= 2
        if level == 0:
            for mux_idx in range(num_mux):
                file_string += f"\tassign level{level}[{mux_idx}] = mux_select[{level}] ? data_SNs[{hdl_wire_map[2*mux_idx]}] :" \
                               f" data_SNs[{hdl_wire_map[2*mux_idx+1]}];\n"
        else:
            for mux in range(num_mux):
                file_string += f"\tassign level{level}[{mux}] = mux_select[{level}] ? level{level-1}[{2*mux}] :" \
                               f" level{level-1}[{2*mux+1}];\n"
    file_string += "\n"
    file_string += f"\tassign out_SN = level{m-1}[0];\n"
    file_string += f"endmodule\n\n\n"
    return file_string


def gen_verilog_hardwired_maj_tree(quant_norm_weights, hdl_wire_map, tree_height, ID=""):
    M = (quant_norm_weights != 0).sum()
    m = tree_height

    file_string = f"module hw_tree{ID}  (\n" \
                  f"\tinput  logic           data_SNs  [{M-1}:0],\n" \
                  f"\tinput  logic [{m-1}:0]     select_SN,\n" \
                  f"\toutput logic           out_SN\n);\n"

    # Initialize an array of wires for every level in the mux tree. (for internal signals)
    num_maj = int(2**m)
    for level in range(m):
        num_maj //= 2
        file_string += f"\tlogic level{level}  [{num_maj-1}:0];\n"

    # Assign the just initialized wires
    num_maj = int(2 ** m)
    for level in range(m):
        file_string += "\n"
        num_maj //= 2
        if level == 0:
            for maj_idx in range(num_maj):
                data_SN1 = f"data_SNs[{hdl_wire_map[2*maj_idx]}]"
                data_SN2 = f"data_SNs[{hdl_wire_map[2*maj_idx+1]}]"
                select = f"select_SN[{level}]"
                file_string += f"\tassign level{level}[{maj_idx}] = ({select}&{data_SN1}) | ({select}&{data_SN2}) | ({data_SN1}&{data_SN2});\n"
        else:
            for maj_idx in range(num_maj):
                data_SN1 = f"level{level-1}[{2*maj_idx}]"
                data_SN2 = f"level{level-1}[{2*maj_idx+1}]"
                select = f"select_SN[{level}]"
                file_string += f"\tassign level{level}[{maj_idx}] = ({select}&{data_SN1}) | ({select}&{data_SN2}) | ({data_SN1}&{data_SN2});\n"
    file_string += "\n"
    file_string += f"\tassign out_SN = level{m-1}[0];\n"
    file_string += f"endmodule\n\n\n"
    return file_string


##############################
#   Output Related Modules   #
##############################
# TODO Doc string
def gen_verilog_output_counter(precision, bipolar, ID=""):
    n = precision
    if bipolar:
        file_string = f"module en_counter{ID} (\n" \
                      f"\tinput  clock, reset,\n" \
                      f"\tinput  logic en,\n" \
                      f"\toutput logic [{n-1}:0] out\n);\n" \
                      f"\talways_ff @(posedge clock) begin\n" \
                      f"\t\tif      (reset == 1)   out <= 'b0;\n" \
                      f"\t\telse if (en == 1)      out <= out + 1;\n" \
                      f"\t\telse                   out <= out - 1;\n" \
                      f"\tend\n" \
                      f"endmodule\n\n\n"
    else:
        file_string = f"module en_counter(\n" \
                      f"\tinput  clock, reset,\n" \
                      f"\tinput  logic en,\n" \
                      f"\toutput logic [{n - 1}:0] out\n);\n" \
                      f"\talways_ff @(posedge clock) begin\n" \
                      f"\t\tif (reset == 1) out <= 'b0; else\n" \
                      f"\t\t                out <=  out + en;\n" \
                      f"\tend\n" \
                      f"endmodule\n\n\n"

    return file_string


##################################
#   High-level Circuit Modules   #
##################################
# TODO: Docstring
def gen_verilog_cemux_filtercore(precision, pcc_input_size, pcc_array_ID, tree_ID, output_ID, latency_factor=None):
    latency_factor = precision if latency_factor is None else latency_factor
    assert latency_factor >= precision, "Latency factor must be at least as large as the precision"
    rns_n, pcc_n = latency_factor, precision
    M = pcc_input_size

    file_string = f"module filter_core (\n" \
                   f"\tinput  clock, reset,\n" \
                   f"\tinput  logic [{pcc_n-1}:0]  pcc_in [{M-1}:0],\n" \
                   f"\tinput  logic [{rns_n-1}:0]  mux_select,\n" \
                   f"\tinput  logic [{rns_n-1}:0]  r,\n" \
                   f"\toutput logic [{rns_n-1}:0]  out\n);\n"

    # Initialize wires
    file_string += f"\tlogic       data_SNs         [{M-1}:0];\n" \
                   f"\tlogic       tree_out;\n\n"

    # Write the PCC array
    pcc_array_name = f"compara_array{pcc_array_ID}"
    file_string += f"\t{pcc_array_name} pccs{pcc_array_ID}(.in(pcc_in), .r(r[{rns_n-1}:{rns_n-pcc_n}]), .SNs(data_SNs));\n"

    # Write the mux tree
    file_string += "\n"
    file_string += f"\thw_tree{tree_ID} tree{tree_ID}(.data_SNs(data_SNs), .mux_select(mux_select), .out_SN(tree_out));\n"

    # Write the output counter
    file_string += f"\ten_counter{output_ID} est{output_ID}(.clock(clock), .reset(reset), .en(tree_out), .out(out));\n"
    file_string += "endmodule\n\n\n"
    return file_string


def gen_verilog_cemaj_filtercore(precision, pcc_input_size, pcc_array_ID, tree_ID, output_ID, latency_factor=None):
    latency_factor = precision if latency_factor is None else latency_factor
    assert latency_factor >= precision, "Latency factor must be at least as large as the precision"
    rns_n, pcc_n = latency_factor, precision

    M = pcc_input_size
    file_string = f"module filter_core (\n" \
                  f"\tinput  clock, reset,\n" \
                  f"\tinput  logic [{pcc_n-1}:0]  pcc_in [{M-1}:0],\n" \
                  f"\tinput  logic [{rns_n-1}:0]  mux_select,\n" \
                  f"\tinput  logic [{rns_n-1}:0]  r,\n" \
                  f"\toutput logic [{rns_n-1}:0]  out\n);\n"

    # Initialize wires
    file_string += f"\tlogic       data_SNs         [{M-1}:0];\n" \
                   f"\tlogic       tree_out;\n\n"

    # Write the PCC array
    pcc_array_name = f"wbg_array{pcc_array_ID}"
    file_string += f"\t{pcc_array_name} pccs{pcc_array_ID}(.in(pcc_in), .r(r[{rns_n-1}:{rns_n-pcc_n}]), .SNs(data_SNs));\n"

    # Write the mux tree
    file_string += "\n"
    file_string += f"\thw_tree{tree_ID} tree{tree_ID}(.data_SNs(data_SNs), .select_SN(mux_select), .out_SN(tree_out));\n"

    # Write the output counter
    file_string += f"\ten_counter{output_ID} est{output_ID}(.clock(clock), .reset(reset), .en(tree_out), .out(out));\n"
    file_string += "endmodule\n\n\n"
    return file_string


# TODO: Docstring
def gen_verilog_cemux_toplevel(precision, pcc_input_size, gated=True, latency_factor=None):
    latency_factor = precision if latency_factor is None else latency_factor
    assert latency_factor >= precision, "Latency factor must be at least as large as the precision"
    rns_n, pcc_n = latency_factor, precision

    # Generate the verilog for the top level module
    file_string = f"module filter (\n" \
                  f"\tinput  clock, reset,\n" \
                  f"\tinput  logic [{pcc_n-1}:0]  in,\n" \
                  f"\toutput logic [{rns_n-1}:0]  stored_out\n);\n"

    # Initialize wires
    file_string += f"\tlogic [{pcc_n-1}:0] pcc_in [{pcc_input_size-1}:0];\n" \
                   f"\tlogic [{rns_n-1}:0] mux_select;\n" \
                   f"\tlogic [{rns_n-1}:0] r;\n" \
                   f"\tlogic [{rns_n-1}:0] out;\n"
    if gated:
        file_string += f"\tlogic gated_clock;\n\n" \
                       f"\tassign gated_clock = clock & (&mux_select);\n"
    else:
        file_string += "\n"

    # Write the submodules
    # Write the RNS and MSG
    file_string += f"\tcounter cnt(.clock(clock), .reset(reset), .rev_state(r), .state(mux_select));\n"

    # Write the memory module
    if gated:
        file_string += f"\tmemory mem(.clock(gated_clock), .reset(reset), .in(in), .out(pcc_in), .count(mux_select));\n"
    else:
        file_string += f"\tcontrol ctrl(.clock(clock), .reset(reset), .in(in), .out(pcc_in), .count(mux_select));\n"

    # Write the filterbank core module
    file_string += f"\tfilter_core core(.clock(clock), .reset(reset), .mux_select(mux_select), .r(r), .pcc_in(pcc_in), .out(out));\n"

    # Write the output register
    file_string += "\n"

    if gated:
        file_string += f"\talways_ff @(posedge gated_clock) begin\n" \
                       f"\t\tif (reset == 1) begin\n" \
                       f"\t\t\tstored_out <= 'b0;\n" \
                       f"\t\tend\n" \
                       f"\t\telse begin\n" \
                       f"\t\t\tstored_out <= out;\n" \
                       f"\t\tend\n" \
                       f"\tend\n" \
                       f"endmodule"

    else:
        ones_string = '1' * rns_n
        file_string += f"\talways_ff @(posedge clock) begin\n" \
                       f"\t\tif (reset == 1) begin\n" \
                       f"\t\t\tstored_out <= 'b0;\n" \
                       f"\t\tend\n" \
                       f"\t\telse begin\n" \
                       f"\t\t\tif (mux_select == {rns_n}'b{ones_string}) begin\n" \
                       f"\t\t\t\tstored_out <= out;\n" \
                       f"\t\t\tend\n" \
                       f"\t\tend\n" \
                       f"\tend\n" \
                       f"endmodule"
    return file_string
