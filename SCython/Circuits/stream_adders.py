import numpy as np
from SCython.SNG import SNG, RNS, PCC, MSG
from SCython.Utilities import seq_utils

def MAJ(SN_array1, SN_array2, SN_array3, do_checks=True):
    if do_checks:
        shapes = (SN_array1.shape, SN_array2.shape, SN_array3.shape)
        assert shapes[0] == shapes[1], f"SN_array1.shape:{shapes[0]} != SN_array2.shape:{shapes[1]}"
        assert (shapes[0] == shapes[2]) or (SN_array3.size == shapes[0][-1]), \
            f"SN_array1.shape:{SN_array1.shape} incompatible with SN_array3.shape:{SN_array3.shape}"
    return ((SN_array1 + SN_array2 + SN_array3) > 1).astype(int)


class HardwiredAdder:
    def __init__(self, tree_height, weights, data_rns, data_pcc, share_r, select_gen, full_corr, bipolar, ddg_tree=False):
        self.tree_height = tree_height
        self.weights = weights
        self.data_rns = data_rns
        self.data_pcc = data_pcc
        self.share_r = share_r
        self.select_gen = select_gen
        self.full_corr = full_corr
        self.ddg_tree = ddg_tree
        self.bipolar = bipolar

        # Unipolar (non-bipolar) hardwired muxes can only implement weighted addition using positive weights
        assert self.bipolar or (self.weights >= 0).all()

        # Tree height should be equal to RNS precision.
        assert tree_height == data_rns.n

        self.num_inputs = len(weights)
        self.inv_mask = weights < 0  # need to know which weights are negative for inverter array
        self.wire_map, self.quant_norm_weights, self.vhdl_wire_map = \
            HardwiredAdder.get_wire_map_and_quant_norm_weights(weights, tree_height, ddg_tree)

        # sng_mask is passed to the data_sng's gen_SN method to implement full correlation if full correlation is enabled.
        self.sng_mask = weights < 0 if self.full_corr else None

        # knowing the nonzero weights is helpful for Verilog code generation
        self.nonzero_weights = self.quant_norm_weights[self.quant_norm_weights != 0]

    @classmethod
    def get_wire_map_and_quant_norm_weights(cls, weights, tree_height, use_ddg_tree):
        num_tree_inputs = int(2**tree_height)

        abs_weights = np.abs(weights)
        abs_normalized_weights = abs_weights / np.sum(abs_weights)
        target_numerators = abs_normalized_weights * num_tree_inputs
        quantized_numerators = np.round(target_numerators).astype(np.int)

        # rounding to nearest integer does not guarantee that exactly all 2^m mux inputs are used. need to fix this.
        while quantized_numerators.sum() > num_tree_inputs:
            diffs = quantized_numerators - target_numerators
            index = np.argmax(diffs)
            quantized_numerators[index] -= 1

        while quantized_numerators.sum() < num_tree_inputs:
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
        wire_map = np.zeros(num_tree_inputs, dtype=int)
        if use_ddg_tree:  # construct DDG tree wire map
            # get the binary expansions of the quantized normalized weight numerators
            binary_expansions = []
            for num in quantized_numerators:
                binary_expansions.append(np.binary_repr(num, width=tree_height))

            # now assign mux tree slots according to the binary coefficients
            next_available_tree_slot = 0
            for bit_position in range(tree_height):
                curr_slots = int(2 ** (tree_height - 1 - bit_position))
                for idx in range(len(binary_expansions)):
                    if binary_expansions[idx][bit_position] == '1':
                        end_slot = next_available_tree_slot + curr_slots
                        wire_map[next_available_tree_slot:end_slot] = idx
                        next_available_tree_slot = end_slot

        else:  # Construct a straightforward wire map
            next_available_tree_slot = 0
            for idx in range(len(weights)):
                end_slot = next_available_tree_slot + quantized_numerators[idx]
                wire_map[next_available_tree_slot:end_slot] = idx
                next_available_tree_slot = end_slot

        # check to make sure we used up all 2^m mux slots
        assert next_available_tree_slot == num_tree_inputs, f"{use_ddg_tree}, {next_available_tree_slot}, {num_tree_inputs}"

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

        # we also compute the quantized, normalized weights
        quantized_numerators = np.zeros(len(weights))
        for i in range(len(wire_map)):
            quantized_numerators[wire_map[i]] += 1
        quant_norm_weights = quantized_numerators / len(wire_map)
        quant_norm_weights[weights < 0] *= -1  # don't forgot to undo the absolute value operation

        return wire_map, quant_norm_weights, vhdl_wire_map

    def forward(self, input_values, SN_length):
        raise NotImplementedError


class HardwiredMux(HardwiredAdder):
    def forward(self, input_values, SN_length):
        assert input_values.ndim == 1
        assert self.data_rns.n == np.log2(SN_length)

        # Quantize input values first
        quant_n = self.data_pcc.n
        input_values = SNG.q_floor(input_values, quant_n, self.bipolar)
        ps = (input_values + 1) / 2 if self.bipolar else input_values

        # Check to make sure that all probabilities are valid
        assert (ps >= 0).all() and (ps <= 1).all(), \
            f"Error, input values should be between [0,1] (unipolar) or [-1,1] (bipolar):\n{input_values}"

        # Generate the tree's select inputs
        select_inputs = self.select_gen.gen_selects(SN_length)

        # transform the selects into SN_indexes by using the hardwired mux wire_map
        SN_indexes = self.wire_map[select_inputs]

        # Use the SN_indexes to pick out which input value is selected each clock cycle
        selected_ps = ps[SN_indexes]

        # Use the RNS to generate the random inputs and pick out which R is valid each clock cycle.
        Rs = self.data_rns.gen_RN(SN_length, ps.shape, self.share_r, inv_mask=self.sng_mask)
        selected_Rs = Rs[SN_indexes, np.arange(SN_length)]

        # Quantize both the Rs and convert everything to an integer
        # If bipolar is used then we have to drop the precision by 1
        if self.data_pcc.n != self.data_rns.n:
            selected_Rs = selected_Rs // 2.0**(self.data_rns.n - self.data_pcc.n)
        selected_Bs = (selected_ps * (2 ** self.data_pcc.n)).astype(int)

        # Use PCC to convert the selected_Rs and input_values to SNs
        output_SN = self.data_pcc.forward(selected_Rs, selected_Bs)

        # Invert the appropriate SN bits (those who have corresponding negative weights).
        selected_mask = self.inv_mask[SN_indexes].astype(bool)

        output_SN[selected_mask] = np.logical_not(output_SN[selected_mask])

        return output_SN.astype(int)


class HardwiredMaj(HardwiredAdder):
    def __init__(self, tree_height, weights, data_rns, data_pcc, share_r, select_gen, full_corr, bipolar, ddg_tree=False):
        assert select_gen.n == data_rns.n

        super().__init__(tree_height, weights, data_rns, data_pcc, share_r, select_gen, full_corr, bipolar, ddg_tree)
        self.data_sng = SNG.SNG(data_rns, data_pcc)
        self.select_bit_streams = select_gen.gen_select_bit_streams()

    def forward(self, input_values, SN_length):
        assert input_values.ndim == 1
        assert self.data_rns.n == np.log2(SN_length)

        # Generate the data input SNs. Xs are the data_SNs
        Xs = self.data_sng.gen_SN(input_values, SN_length, self.bipolar, self.share_r, self.sng_mask)

        # Do the XNOR array
        Xs[self.inv_mask] = np.logical_not(Xs[self.inv_mask]).astype(int)

        # Compute the tree.
        if self.select_bit_streams is None:
            select_bit_streams = self.select_gen.gen_select_bit_streams()
        else:
            select_bit_streams = self.select_bit_streams

        prev_layer_Xs = Xs[self.wire_map]
        prev_layer_size = prev_layer_Xs.shape[0]

        for layer in range(self.tree_height):
            select_SN = select_bit_streams[layer]
            new_layer_size = prev_layer_size // 2
            new_layer_Xs = np.empty((new_layer_size, SN_length))
            for idx in range(new_layer_size):
                new_layer_Xs[idx] = MAJ(prev_layer_Xs[2*idx], prev_layer_Xs[2*idx+1], select_SN, do_checks=False)

            prev_layer_Xs = new_layer_Xs
            prev_layer_size = new_layer_size
        return prev_layer_Xs[0].astype(int)


class CeMux(HardwiredMux):  # HardwiredMux wrapper class for CeMux.
    def __init__(self, tree_height, weights, data_pcc, ddg_tree=False, bipolar=True, vdc_seq=None, dual=False):
        full_corr = True
        share_r = True
        if dual:
            data_rns = RNS.Counter_RNS(tree_height)
            select_gen = MSG.VDC_MSG(tree_height, tree_height, weights, vdc_seq=vdc_seq)
        else:
            data_rns = RNS.VDC_RNS(tree_height, vdc_seq=vdc_seq)
            select_gen = MSG.Counter_MSG(tree_height, tree_height, weights)
        super().__init__(tree_height, weights, data_rns, data_pcc, share_r, select_gen, full_corr, bipolar, ddg_tree)


class CeMaj(HardwiredMaj):  # HardwiredMux wrapper class for CeMux.
    def __init__(self, tree_height, weights, data_pcc, ddg_tree=False, bipolar=True, vdc_seq=None, dual=False):
        full_corr = False
        share_r = True
        if dual:
            data_rns = RNS.Counter_RNS(tree_height)
            select_gen = MSG.VDC_MSG(tree_height, tree_height, weights, vdc_seq=vdc_seq)
        else:
            data_rns = RNS.VDC_RNS(tree_height, vdc_seq=vdc_seq)
            # data_rns = RNS.Hypergeometric_RNS(tree_height)
            # select_gen = MSG.Counter_MSG(tree_height, tree_height, weights)
            select_gen = MSG.HYPER_MSG(tree_height, tree_height, weights)

        super().__init__(tree_height, weights, data_rns, data_pcc, share_r, select_gen, full_corr, bipolar, ddg_tree)
        self.select_bit_streams = None


if __name__ == '__main__':
    # Testing stuff (ignore)
    precision = 8
    pcc = PCC.Comparator
    vdc_seq = seq_utils.get_vdc(precision)

    num_inputs = int(2**precision)
    weights = np.ones(num_inputs)
    values = np.zeros(num_inputs)

    cemux = CeMux(tree_height=precision, weights=weights, data_pcc=pcc, vdc_seq=vdc_seq)
    Z = cemux.forward(input_values=values, SN_length=num_inputs)
