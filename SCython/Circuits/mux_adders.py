import numpy as np
from SCython.SNG import SNG, RNS, PCC, MSG
from SCython.Utilities import seq_utils

# TODO Implement symmetric coefficient in forward method
class HardwiredMux:
    def __init__(self, tree_height, weights, data_sng, select_gen, use_full_corr, use_ddg_tree, bipolar, symmetric_form):
        """
        :param int tree_height:
        :param numpy.ndarray weights:
        :param SNG.SNG data_sng:
        :param MSG.MSG select_gen:
        :param bool use_full_corr:
        :param bool use_ddg_tree:
        :param bool bipolar:
        :param bool symmetric_form:
        """
        self.tree_height = tree_height
        self.weights = weights
        self.data_sng = data_sng
        self.select_gen = select_gen
        self.use_full_corr = use_full_corr
        self.use_ddg_tree = use_ddg_tree
        self.bipolar = bipolar
        self.symmetric_form = symmetric_form

        # Unipolar (non-bipolar) hardwired muxes can only implement weighted addition using positive weights
        assert self.bipolar or (self.weights >= 0).all()

        # Tree height should be equal to RNS precision (for now)
        assert tree_height == data_sng.rns.n

        # Weights must be symmetric to use symmetric form
        if symmetric_form:
            for i in range(len(weights)//2):
                assert weights[i] == weights[-(i+1)]

        self.num_inputs = len(weights)
        self.inv_mask = weights < 0  # need to know which weights are negative for inverter array
        self.wire_map, self.quant_norm_weights, self.vhdl_wire_map = \
            HardwiredMux.get_wire_map_and_quant_norm_weights(weights, tree_height, use_ddg_tree, symmetric_form)

        # sng_mask is passed to the data_sng's gen_SN method to implement full correlation if full correlation is enabled.
        self.sng_mask = weights < 0 if self.use_full_corr else None

        # knowing the nonzero weights is helpful for Verilog code generation
        self.nonzero_weights = self.quant_norm_weights[self.quant_norm_weights != 0]

    @classmethod
    def get_wire_map_and_quant_norm_weights(cls, weights, tree_height, use_ddg_tree, symmetric_form):
        # TODO: Test symmetric form
        if symmetric_form:
            # tree height is cut in half if symmetric form is used
            tree_height = tree_height - 1
            if len(weights) % 2 == 1:  # Handle odd case
                mux_weights = weights[0:len(weights)//2+1].copy()
                mux_weights[-1] = mux_weights[-1]/2
            else:
                mux_weights = weights[0:len(weights)//2]
        else:
            mux_weights = weights

        num_mux_inputs = int(2 ** tree_height)

        abs_weights = np.abs(mux_weights)
        abs_normalized_weights = abs_weights / np.sum(abs_weights)
        target_numerators = abs_normalized_weights * num_mux_inputs
        quantized_numerators = np.round(target_numerators).astype(np.int)

        # rounding to nearest integer does not guarantee that exactly all 2^m mux inputs are used. need to fix this.
        while quantized_numerators.sum() > num_mux_inputs:
            diffs = quantized_numerators - target_numerators
            index = np.argmax(diffs)
            quantized_numerators[index] -= 1

        while quantized_numerators.sum() < num_mux_inputs:
            diffs = -(quantized_numerators - target_numerators)
            index = np.argmax(diffs)
            quantized_numerators[index] += 1

        '''
        - wire_map's i-th value tells us the index of the data input SN that is connected to the i-th mux tree slot
        - Ex: wire_map[i] = j says that the j-th input SN, X_j, is connected to i-th mux input slot meaning that X_j will
          will be sampled when the mux's select input is j.
        - we can construct the wire map such that the resulting mux is a DDG tree (this leads to lowest area) or we can
          construct the wire map top down and assign mux inputs to X_1, then to X_2 then to X_3 etc. 
        '''
        wire_map = np.zeros(num_mux_inputs, dtype=int)
        if use_ddg_tree:  # construct DDG tree wire map
            # get the binary expansions of the quantized normalized weight numerators
            binary_expansions = []
            for num in quantized_numerators:
                binary_expansions.append(np.binary_repr(num, width=tree_height))

            # now assign mux tree slots according to the binary coefficients
            next_available_mux_slot = 0
            for bit_position in range(tree_height):
                curr_slots = int(2 ** (tree_height - 1 - bit_position))
                for idx in range(len(binary_expansions)):
                    if binary_expansions[idx][bit_position] == '1':
                        end_slot = next_available_mux_slot + curr_slots
                        wire_map[next_available_mux_slot:end_slot] = idx
                        next_available_mux_slot = end_slot

        else:  # Construct a straightforward wire map
            next_available_mux_slot = 0
            for idx in range(len(mux_weights)):
                end_slot = next_available_mux_slot + quantized_numerators[idx]
                wire_map[next_available_mux_slot:end_slot] = idx
                next_available_mux_slot = end_slot

        # check to make sure we used up all 2^m mux slots
        assert next_available_mux_slot == num_mux_inputs, f"{use_ddg_tree}, {next_available_mux_slot}, {num_mux_inputs}"

        # we also compute the vhdl_wire_map which is helpful for VHDL generation where we ignore zero weighted inputs
        # and we have to handle symmetric form
        modifiers = np.zeros(len(quantized_numerators) + 1, dtype=np.int)
        vhdl_wire_map = wire_map.copy()
        for i in range(len(quantized_numerators)):
            modifiers[i + 1] = modifiers[i]
            if quantized_numerators[i] == 0:
                modifiers[i + 1] += 1

        modifiers = modifiers[1:]
        for i in range(len(wire_map)):
            vhdl_wire_map[i] -= modifiers[wire_map[i]]

        # Now we expand the symmetric form stuff which is helpful for simulation
        if symmetric_form:
            temp = np.empty(2*len(wire_map), dtype=int)
            temp[0:len(wire_map)] = wire_map
            temp[len(wire_map):] = len(weights) - 1 - wire_map
            wire_map = temp

        # we also compute the quantized, normalized weights
        quantized_numerators = np.zeros(len(weights))
        for i in range(len(wire_map)):
            quantized_numerators[wire_map[i]] += 1
        quant_norm_weights = quantized_numerators / len(wire_map)
        quant_norm_weights[weights < 0] *= -1  # don't forgot to undo the absolute value operation

        return wire_map, quant_norm_weights, vhdl_wire_map

    @classmethod
    def get_vdc_Rs(cls, wire_map, num_weights):
        num_mux_inputs = len(wire_map)
        m = np.log2(num_mux_inputs)
        assert m % 1 == 0, f"Hardwired mux wire_map must have a power of 2 number of entries. Given len: {num_mux_inputs}"
        m = int(m)

        vdc = seq_utils.get_vdc(m)
        Rs = [[] for _ in range(num_weights)]
        for idx in range(num_weights):
            Rs[idx].append(vdc[wire_map == idx])

        return Rs

    # TODO clean up this method
    def forward(self, input_values, SN_length) -> np.ndarray:
        """
        :param numpy.ndarray input_values:
        :param int SN_length:
        :return:
        """
        raise NotImplementedError("Forward method in Hardwired mux is not tested")

        assert input_values.ndim == 1
        num_input = input_values.shape

        # if bipolar, determine probabilities from values. Assuming 2's complement, this transform is implemented in hardware
        # by inverting the MSB (aka sign bit) and then treating the n-digits as an unsigned integer. Note we will
        # quantize these values later
        ps = (input_values + 1) / 2 if self.bipolar else input_values

        # Check to make sure that all probabilities are valid
        assert (ps >= 0).all() and (ps <= 1).all(), \
            f"Error, input values should be between [0,1] (unipolar) or [-1,1] (bipolar):\n{input_values}"

        # Get the shared RNS output
        Rs = self.data_sng.rns.gen_RN(SN_length, shape=(num_input,), share=True, mask=self.sng_mask)

        # Get the select inputs
        selects = self.select_gen.gen_selects(SN_length)

        # transform the selects into SN_indexes by using the hardwired mux wire_map
        # wire_map = self.wire_map[None, :].repeat((runs, 1))
        # SN_indexes = self.wire_map.gather(dim=1, index=selects)[:, None, :].long()
        SN_indexes = self.wire_map[selects]

        # Use the SN_indexes to pick out which input value is selected each clock cycle
        # selected_ps = ps[..., None].repeat((1, 1, SN_length)).gather(dim=1, index=SN_indexes).squeeze()
        selected_ps = ps[np.newaxis][SN_indexes, np.arange(SN_length)]

        # Use the SN_indexes to pick out which R values are used each clock cycle
        # This is done in case full_corr is used because some Rs will be R and some will be ~R
        selected_Rs = Rs.gather(dim=1, index=SN_indexes).squeeze()

        # Quantize both the Rs and the Bs and convert everything to an integer (quantization is done in fraction domain)
        # If bipolar is used then we have to drop the precision by 1
        if self.data_sng.pcc.n != self.data_sng.rns.n:
            fractional_Rs = selected_Rs/(2**self.data_sng.n)
            fractional_Rs = SNG.q_floor(fractional_Rs, precision=self.data_sng.pcc.n)
            selected_Rs = (fractional_Rs * (2 ** self.data_sng.pcc.n).astype(int))

        quant_n = self.data_sng.pcc.n - int(self.bipolar)
        selected_Bs = (self.data_sng.q_func(selected_ps, quant_n) * (2 ** quant_n)).astype(int)

        # Use PCC to convert the selected_Rs and input_values to SNs
        output = self.data_sng.pcc.forward(selected_Rs, selected_Bs)

        # account for XNOR arrays
        # selected_invs = self.inv_mask[..., None].repeat((1, 1, SN_length)).gather(dim=1, index=SN_indexes).squeeze()
        selected_invs = self.inv_mask[np.newaxis][SN_indexes, SN_length]
        output[selected_invs] = np.logical_not(output[selected_invs])

        return output

    def _gen_tree_verilog(self, ID=""):
        M = (self.quant_norm_weights != 0).sum()
        m = self.tree_height

        if self.symmetric_form:
            M = M//2 + (M % 2)
            m = m - 1

        file_string = f"module hw_tree{ID}  (\n" \
                      f"\tinput  logic           xs  [{M - 1}:0],\n" \
                      f"\tinput  logic [{m - 1}:0]     s,\n" \
                      f"\toutput logic           out\n);\n"

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
                    file_string += f"\tassign level{level}[{mux_idx}] = s[{level}] ? xs[{self.vhdl_wire_map[2 * mux_idx]}] :" \
                                   f" xs[{self.vhdl_wire_map[2 * mux_idx + 1]}];\n"
            else:
                for mux in range(num_mux):
                    file_string += f"\tassign level{level}[{mux}] = s[{level}] ? level{level - 1}[{2 * mux}] :" \
                                   f" level{level - 1}[{2 * mux + 1}];\n"
        file_string += "\n"
        file_string += f"\tassign out = level{m - 1}[0];\n"
        file_string += f"endmodule\n\n\n"
        return file_string

    def _gen_sym_verilog(self):
        num_inputs = len(self.nonzero_weights)
        pcc_n = self.data_sng.pcc.n
        # num_outputs = self.m - 1
        num_outputs = int(num_inputs//2 + (num_inputs % 2))
        # num_inputs = num_outputs/2
        file_string = f"module sym_select  (\n" \
                      f"\tinput  logic [{pcc_n-1}:0] in  [{num_inputs-1}:0],\n" \
                      f"\tinput  logic       s,\n" \
                      f"\toutput logic [{pcc_n-1}:0] out [{num_outputs-1}:0]\n);\n"

        for i in range(num_outputs):
            file_string += f"\tassign out[{i}] = s ? in[{num_inputs-1-i}] : in[{i}];\n"
        file_string += f"endmodule\n\n\n"
        return file_string

    def _gen_control_verilog(self):
        num_inputs = len(self.weights)
        nonzero_indexes = self.quant_norm_weights != 0
        num_nonzero_inputs = nonzero_indexes.sum()
        rns_n, pcc_n = self.data_sng.rns.n, self.data_sng.pcc.n

        file_string = f"module control  (\n" \
                      f"\tinput  clock, reset,\n" \
                      f"\tinput  logic [{pcc_n - 1}:0] in,\n" \
                      f"\toutput logic [{pcc_n-1}:0] out [{num_nonzero_inputs - 1}:0]\n);\n"
        file_string += f"\tlogic [{pcc_n-1}:0] registers [{num_inputs-1}:0];\n"

        # Wire the nonzero weighted inputs to the output of this control module
        out_idx = 0
        for in_idx, isnt_zero in enumerate(nonzero_indexes):
            if isnt_zero:
                file_string += f"\tassign out[{out_idx}] = registers[{in_idx}];\n"
                out_idx += 1

        # Set up the sequential logic
        file_string += f"\n\talways_ff @(posedge clock) begin\n" \
                       f"\t\tif (reset == 1) begin\n" \
                       f"\t\t\tfor(int i=0; i<{num_inputs}; i=i+1) registers[i] <= 'b0;\n" \
                       f"\t\tend\n" \
                       f"\t\telse begin\n" \
                       f"\t\t\tregisters[{num_inputs-1}:1] <= registers[{num_inputs-2}:0];\n" \
                       f"\t\t\tregisters[0] <= in;\n" \
                       f"\t\tend\n" \
                       f"\tend\n" \
                       f"endmodule\n\n\n"

        return file_string

    def gen_verilog(self, module_name):
        rns_n = self.data_sng.rns.n
        pcc_n = self.data_sng.pcc.n

        nonzero_inv_mask = self.inv_mask[self.quant_norm_weights != 0]
        if self.symmetric_form:
            nonzero_inv_mask = nonzero_inv_mask[0:np.ceil(len(nonzero_inv_mask)/2).astype(int)]

        file_string = ""

        # Generate the verilog for the control hardware
        file_string += self._gen_control_verilog()

        # Generate verilog for RNS
        file_string += "// Data RNS Module\n"
        file_string += self.data_sng.rns.gen_verilog(IDs=['data'])

        # Generate verilog for PCC (also handles the XNOR array)
        file_string += "// PCC Modules\n"
        file_string += self.data_sng.pcc.gen_verilog(nonzero_inv_mask, ID='data', share_r=True, write_submodules=True,
                                                     corr_adj=self.use_full_corr)

        # Generate verilog for mux select gen (if RNS is VDC and MSG is Counter, then they can share)
        if not (type(self.data_sng.rns) is RNS.VDC_RNS and type(self.select_gen) is MSG.Counter_MSG):
            file_string += "// Mux Select Generator Module\n"
            print(self.data_sng.rns, self.select_gen)
            # TODO Fix this case
            file_string += self.select_gen.gen_verilog()

        # Generate verilog for hardwired mux tree
        file_string += "// Hardwired Mux Tree Module\n"
        file_string += self._gen_tree_verilog()

        # Generate the verilog for the output counter
        file_string += "// Output Counter\n"
        file_string += SNG.output_counter_verilog(rns_n, self.bipolar)

        # Generate the verilog for symmetric logic if necessary
        pcc_array_len = len(nonzero_inv_mask)
        if self.symmetric_form:
            file_string += "// Symmetric coefficient logic\n"
            file_string += self._gen_sym_verilog()

        # Generate the verilog for the top level module
        file_string += f"// Top level module\n"
        file_string += f"module {module_name} (\n" \
                       f"\tinput  clock, reset,\n" \
                       f"\tinput  logic [{pcc_n-1}:0]  in,\n" \
                       f"\toutput logic [{rns_n-1}:0]  out\n);\n"

        # Initialize wires
        num_nonzero = len(self.nonzero_weights)
        file_string += f"\tlogic [{pcc_n-1}:0] nonzero_in [{num_nonzero-1}:0];\n" \
                       f"\tlogic       xs         [{pcc_array_len-1}:0];\n" \
                       f"\tlogic [{self.tree_height - 1}:0] s;\n" \
                       f"\tlogic [{rns_n-1}:0] r;\n" \
                       f"\tlogic       tree_out;\n" \
                       f"\tlogic [{pcc_n-1}:0] pcc_in     [{pcc_array_len-1}:0];\n\n"

        # Write the submodules
        pcc_array_module_name = f"{self.data_sng.pcc.name.lower()}_array_data"
        spacing = " " * (len(pcc_array_module_name) - len('en_counter'))

        # Write the control module
        file_string += f"\tcontrol   {spacing}crtl(.clock(clock), .reset(reset), .in(in), .out(nonzero_in));"

        # Write the RNS and MSG
        file_string += "\n"
        if type(self.data_sng.rns) is RNS.VDC_RNS and type(self.select_gen) is MSG.Counter_MSG:
            file_string += f"\tcounter    {spacing}cnt(.clock(clock), .reset(reset), .rev_state(r), .state(s));\n"
        else:
            if type(self.data_sng.rns) is RNS.VDC_RNS:
                f"\tcounter           cnt(.clock(clock), .reset(reset), .rev_state(r));\n"
            else:
                raise NotImplementedError
            raise NotImplementedError

        # Initialize symmetric selector if necessary
        if self.symmetric_form:
            s_idx = self.tree_height - 1
            file_string += f"\tsym_select {spacing}sym(.in(nonzero_in), .s(s[{self.tree_height - 1}]), .out(pcc_in));\n"
        else:
            file_string += "\tassign pcc_in = nonzero_in;\n"
            s_idx = self.tree_height

        file_string += f"\t{pcc_array_module_name} pccs(.in(pcc_in), .r(r[{pcc_n-1}:0]), .xs(xs));\n"
        file_string += f"\thw_tree    {spacing}tree(.xs(xs), .s(s[{s_idx-1}:0]), .out(tree_out));\n" \
                       f"\ten_counter {spacing}est(.clock(clock), .reset(reset), .en(tree_out), .out(out));\n" \
                       f"endmodule"

        return file_string

    def gen_tree_verilog_filterbank(self, ID=""):
        num_inputs = len(self.weights)
        tree_height = self.tree_height

        file_string = f"module hw_tree{ID}  (\n" \
                      f"\tinput  logic           pos_SNs  [{num_inputs - 1}:0],\n" \
                      f"\tinput  logic           neg_SNs  [{num_inputs - 1}:0],\n" \
                      f"\tinput  logic [{tree_height - 1}:0]     s,\n" \
                      f"\toutput logic           out\n);\n"

        # Initialize an array of wires for every level in the mux tree. (for internal signals)
        num_mux = int(2 ** tree_height)
        for level in range(tree_height):
            num_mux //= 2
            file_string += f"\tlogic level{level}  [{num_mux - 1}:0];\n"

        # Assign the just initialized wires
        num_mux = int(2 ** tree_height)
        for level in range(tree_height):
            file_string += "\n"
            num_mux //= 2
            if level == 0:
                for mux_idx in range(num_mux):
                    input1_idx = self.wire_map[2*mux_idx]
                    input2_idx = self.wire_map[2*mux_idx]
                    if self.inv_mask[input1_idx]:  # weight is negative
                        in1 = f"neg_SNs[{input1_idx}]"
                    else:
                        in1 = f"pos_SNs[{input1_idx}]"

                    if self.inv_mask[input2_idx]:  # weight is negative
                        in2 = f"neg_SNs[{input2_idx}]"
                    else:
                        in2 = f"pos_SNs[{input2_idx}]"

                    file_string += f"\tassign level{level}[{mux_idx}] = s[{level}] ? {in1} : {in2};\n"
            else:
                for mux in range(num_mux):
                    file_string += f"\tassign level{level}[{mux}] = s[{level}] ? level{level - 1}[{2 * mux}] :" \
                                   f" level{level - 1}[{2 * mux + 1}];\n"
        file_string += "\n"
        file_string += f"\tassign out = level{tree_height - 1}[0];\n"
        file_string += f"endmodule\n\n\n"
        return file_string


class CeMux(HardwiredMux):
    def __init__(self, tree_height, weights, pcc, use_ddg_tree=False, bipolar=True, symmetric_form=False,
                 vdc_n=None, pcc_n=None, vdc_seq=None):
        """
        :param int tree_height:
        :param numpy.ndarray weights:
        :param PCC.PCC pcc:
        :param bool use_ddg_tree:
        :param bool bipolar:
        :param bool symmetric_form:
        :param int vdc_n:
        :param int pcc_n:
        :param numpy.ndarray vdc_seq:
        """
        vdc_n = tree_height if pcc_n is None else vdc_n
        pcc_n = tree_height if pcc_n is None else pcc_n
        data_sng = SNG.SNG(RNS.VDC_RNS, pcc, vdc_n, pcc_n, vdc_seq=vdc_seq)
        select_gen = MSG.Counter_MSG(tree_height, tree_height, weights)
        use_full_corr = True
        super().__init__(tree_height, weights, data_sng, select_gen, use_full_corr, use_ddg_tree, bipolar, symmetric_form)

    def forward(self, input_values, SN_length) -> np.ndarray:
        assert input_values.ndim == 1
        assert self.data_sng.rns.n == np.log2(SN_length) and self.data_sng.rns.n == self.tree_height

        # if bipolar, determine probs from values. Assuming 2's complement, this transform is implemented in hardware
        # by inverting the MSB (aka sign bit) and then treating the n-digits as an unsigned integer. Quantize input values first
        quant_n = self.data_sng.pcc.n - int(self.bipolar)
        input_values = SNG.q_floor(input_values, quant_n)
        ps = (input_values + 1) / 2 if self.bipolar else input_values

        # Check to make sure that all probabilities are valid
        assert (ps >= 0).all() and (ps <= 1).all(), \
            f"Error, input values should be between [0,1] (unipolar) or [-1,1] (bipolar):\n{input_values}"

        # Get the shared RNS output (2D)
        posneg = self.data_sng.rns.gen_RN_posneg()

        # transform the selects into SN_indexes by using the hardwired mux wire_map
        SN_indexes = self.wire_map

        # Use the SN_indexes to pick out which input value is selected each clock cycle
        selected_ps = ps[SN_indexes]

        # Use the SN_indexes to pick out which whether the weight is positive or negative each clock cycle
        selected_mask = self.inv_mask[SN_indexes].astype(bool)

        # Use the SN_indexes to pick out which R values are used each clock cycle
        # This is done in case full_corr is used because some Rs will be R and some will be ~R
        selected_Rs = posneg[0]
        selected_Rs[selected_mask] = posneg[1][selected_mask]

        # Quantize both the Rs and convert everything to an integer
        # If bipolar is used then we have to drop the precision by 1
        if self.data_sng.pcc.n != self.data_sng.rns.n:
            print('pcc_n != rns_n')
            selected_Rs = selected_Rs // 2.0**(self.data_sng.rns.n - self.data_sng.pcc.n)
        selected_Bs = (selected_ps * (2 ** self.data_sng.pcc.n)).astype(int)

        # Use PCC to convert the selected_Rs and input_values to SNs
        output = self.data_sng.pcc.forward(selected_Rs, selected_Bs)

        output[selected_mask] = np.logical_not(output[selected_mask])

        return output.astype(int)


if __name__ == '__main__':
    precision = 8
    pcc = PCC.Comparator
    vdc_seq = seq_utils.get_vdc(precision)

    num_inputs = int(2**precision)
    weights = np.ones(num_inputs)
    values = np.zeros(num_inputs)

    cemux = CeMux(tree_height=precision, weights=weights, pcc=pcc, vdc_seq=vdc_seq)
    Z = cemux.forward(input_values=values, SN_length=num_inputs)
