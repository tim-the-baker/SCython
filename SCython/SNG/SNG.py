# TODO update docstrings
import numpy as np
from SCython.SNG import RNS, PCC


#### Quantization Functions ###
def q_floor(x, precision, signed=False) -> np.ndarray:
    """
    truncates x to a specified precision
    :param numpy.ndarray x:
    :param int precision:
    :param bool signed:
    :return:
    """
    #  truncates x to specified precision
    pow2n = 2**(precision - int(signed))
    return np.floor(x*pow2n)/pow2n


def q_nearest(x, precision, signed=False) -> np.ndarray:
    """
    rounds x to specified precision (rounds to nearest)
    :param numpy.ndarray x:
    :param int precision:
    :return:
    """
    pow2n = 2**(precision-int(signed))
    return np.round(x*pow2n)/pow2n


def output_counter_verilog(n: int, bipolar: bool, ID: str = ""):
    if bipolar:
        file_string =  f"module en_counter{ID} (\n" \
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
        file_string =  f"module en_counter(\n" \
                       f"\tinput  clock, reset,\n" \
                       f"\tinput  logic en,\n" \
                       f"\toutput logic [{n - 1}:0] out\n);\n" \
                       f"\talways_ff @(posedge clock) begin\n" \
                       f"\t\tif (reset == 1) out <= 'b0; else\n" \
                       f"\t\t                out <=  out + en;\n" \
                       f"\tend\n" \
                       f"endmodule\n\n\n"

    return file_string


class SNG:
    def __init__(self, rns, pcc, rns_precision, pcc_precision, **kwargs):
        """
        :param RNS.RNS rns: RNS class for the SNG (see rns_utils)
        :param PCC.PCC pcc: PCC class for the SNG (see pcc_utils)
        :param int rns_precision: bit-width of RNS
        :param int pcc_precision: bit-width of PCC
        :param kwargs: Extra parameters mainly for hardware RNSs. Current choices include:
            feedback: LFSR feedback type (internal: 'i', external: 'e' or all: 'a')
            seqs: Pre-loaded seqs (seqs should correspond to n-bit RNS). (Note this class now automatically loads seqs.
            seq_idx: index of specific seq in seqs that should be used; seqs should be specified when using this.
        """
        assert rns_precision >= pcc_precision
        self.q_func = q_floor

        self.rns = rns(rns_precision, **kwargs)
        self.pcc = pcc(pcc_precision)

        self.vdc_seq = kwargs.get("vdc_seq")

    def gen_SN(self, values, SN_length, bipolar, share_RNS, RNS_mask) -> np.ndarray:
        """
        :param numpy.ndarray values:
        :param int SN_length:
        :param bool bipolar:
        :param bool share_RNS:
        :param np.ndarray RNS_mask:
        :return:
        """
        # quantize the input values
        values = self.q_func(values, self.pcc.n)

        # determine the SN probabilities based on values and whether SNs are bipolar or unipolar.
        ps = (values+1)/2 if bipolar else values

        assert (ps >= 0).all(), ps
        assert (ps <= 1).all(), ps

        # transform the fixed point probability values to integer format
        Cs = (ps * (2**self.pcc.n)).astype(int)

        # generate RNs
        Rs = self.rns.gen_RN(SN_length, ps.shape, share_RNS, RNS_mask)

        # the Rs must also quantized to fit into the PCC if rns.n != pcc.n. Do this by truncating the MSBs of the Rs.
        # To accomplish MSB truncation, a 1s mask of length pcc.n can be bitwise AND'ed with Rs
        if self.pcc.n != self.rns.n:
            truncation_mask = int(2**self.pcc.n) - 1
            Rs = np.bitwise_and(Rs, truncation_mask)

        SNs = self.pcc.forward(Rs, Cs)

        assert SNs.shape == (*values.shape, SN_length), f"SN Shape: {SNs.shape} was expected to be ({values.shape}, {SN_length})."
        return SNs

    def gen_verilog(self):
        raise NotImplementedError

    def info(self):
        return f"{self.rns.info()}\n{self.pcc.info()}"


if __name__ == '__main__':
    pass
