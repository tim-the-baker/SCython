import numpy as np

class PCC:
    """ Abstract class for stochastic computing probability conversion circuits """
    def __init__(self, n):
        """
        :param n:
        :param can_generate_all_1s:
        """
        self.n = n
        self.name: str = ""
        self.legend: str = ""

    def forward(self, Rs, Cs) -> np.ndarray:
        """
        Abstract method for implementation of the PCC.
        :param numpy.ndarray Rs: the random number inputs to the PCC.
        :param numpy.ndarray Cs: the control inputs to the PCC.
        :return: the generated SNs
        """
        raise NotImplementedError

    def gen_verilog(self, inv_outs, ID, share_r, write_submodules, corr_adj) -> str:
        """
        This method should generate verilog code for the PCC. Software PCCs should return 'None' for this method.
        This method can generate an array of PCCs. Each PCC in the array is given a unique ID indicated by IDs param.
        :param numpy.ndarray inv_outs: boolean array that indicates which SNs are inverted immediately following generation.
        :param list ID: list of IDs for the array of PCCs.
        :param bool share_r: when True, a single RNS is shared amongst all PCCs in PCC array.
        :param bool write_submodules: when True, generates verilog both the PCC array and for the PCC submodule.
        :param bool corr_adj: whether or not to generate Verilog code that implements correlation adjustment.
        :return:
        """
        raise NotImplementedError

    def __str__(self):
        return self.name

    def info(self):
        return f"### PCC Info: name={self.name} n={self.n}"

# TODO: Finish updating this class for NumPy
class WBG(PCC):
    def __init__(self, n):
        super().__init__(n)
        self.name = "WBG"
        self.legend = "WBG"

    def forward(self, Rs, Cs):
        SNs = np.full(Rs.shape, -1, dtype=int)
        np.log2(Rs,  where=(Rs != 0), out=SNs, casting='unsafe')
        SNs = np.bitwise_and(np.right_shift(Cs.T, SNs.T), 1).T

        # the following line handles the case when the input value is = 1. In real hardware, a real WBG cannot handle
        # this situation. For instance, a standard 4-bit WBG can generate unipolar SNs with value 0, 1/16, 2/16, ...,
        # 15/16, but not with value 16/16 because an extra input bit would be needed. However, relatively little hardware
        # modification would needed to handle the value = 16/16 case (just 1 OR gate would be needed).
        SNs[Cs == pow(2, self.n)] = np.ones(Rs.shape[-1])

        return SNs

    def gen_verilog(self, inv_outs, ID, share_r, write_submodules, corr_adj, share_p1=True):
        assert share_r, "Only sharing R is implemented at the moment."

        file_string = ""
        if write_submodules:  # write the wbg_p1 and wbg_p2 submodules if necessary (should only do this once)
            # WBG P1
            file_string = f"module wbg_p1  (\n" \
                          f"\tinput  logic [{self.n-1}:0]     r,\n" \
                          f"\toutput logic [{self.n-1}:0]     out\n);\n"
            inverts = ""
            for idx in range(self.n - 1, -1, -1):  # loop backwards
                file_string += f"\tassign out[{idx}] = r[{idx}]{inverts};\n"
                inverts += f" & ~r[{idx}]"
            file_string += "endmodule\n\n\n"

            # WBG P2
            file_string += f"module wbg_p2  (\n" \
                           f"\tinput  logic [{self.n-1}:0]     w,\n" \
                           f"\tinput  logic [{self.n-1}:0]     p,\n" \
                           f"\toutput logic           out\n);\n"
            file_string += f"\tlogic [{self.n-1}:0] u;\n"
            file_string += f"\tassign u = w & p;\n"
            file_string += f"\tassign out = |u;\n"
            file_string += f"endmodule\n\n\n"

            # WBG P2 with inverted output
            file_string += f"module wbg_p2_inv  (\n" \
                           f"\tinput  logic [{self.n-1}:0]     w,\n" \
                           f"\tinput  logic [{self.n-1}:0]     p,\n" \
                           f"\toutput logic           out\n);\n"
            file_string += f"\tlogic [{self.n-1}:0] u;\n"
            file_string += f"\tlogic       temp;\n"
            file_string += f"\tassign u = w & p;\n"
            file_string += f"\tassign temp = |u;\n"
            file_string += f"\tassign out = ~temp;\n"
            file_string += f"endmodule\n\n\n"

        # Write the WBG array
        arr_len = len(inv_outs)
        file_string += f"module {self.name.lower()}_array_{ID}  (\n" \
                       f"\tinput  logic [{self.n-1}:0]     in  [{arr_len-1}:0],\n" \
                       f"\tinput  logic [{self.n-1}:0]     r,\n" \
                       f"\toutput logic           xs  [{arr_len-1}:0]\n);\n"

        file_string += f"\tlogic [{self.n-1}:0] w;\n"
        file_string += f"\twbg_p1 wbgp1(.r(r), .out(w));\n\n"

        if corr_adj:
            # Handle inverted WBG P1 with inverted R input if corr_adj is used
            file_string += f"\tlogic [{self.n-1}:0] w_neg;\n"
            file_string += f"\twbg_p1 wbgp1_inv(.r(~r), .out(w_neg));\n\n"
            w_neg = "w_neg"
            spacing = " "*4
        else:
            w_neg = "w"
            spacing = ""

        for j in range(arr_len):
            if inv_outs[j] == 1:
                file_string += f"\twbg_p2_inv wbg{j}(.w({w_neg}), .p(in[{j}]), .out(xs[{j}]));\n"
            else:
                file_string += f"\twbg_p2     wbg{j}(.w(w), {spacing}.p(in[{j}]), .out(xs[{j}]));\n"

        file_string += f"endmodule\n\n\n"
        return file_string

class Comparator(PCC):
    """
    Class that implements comparator PCC.
    """
    def __init__(self, n):
        super().__init__(n)
        self.name = "Comparator"
        self.legend = "Comparator"

    def forward(self, Rs, Bs):
        SNs = (Rs.T < Bs.T).T.astype(np.int8)
        return SNs

    def gen_verilog(self, inv_outs, ID, share_r, write_submodules, corr_adj):
        assert share_r, "Only sharing R is implemented at the moment."

        file_string = ""
        if write_submodules:
            # Comparator
            file_string += f"module compara (\n" \
                           f"\tinput  logic [{self.n-1}:0] r,\n" \
                           f"\tinput  logic [{self.n-1}:0] p,\n" \
                           f"\toutput logic       out\n);\n" \
                           f"\tassign out = r < p;\n" \
                           f"endmodule\n\n\n"

            # Comparator with inverted output
            file_string += f"module compara_inv (\n" \
                           f"\tinput  logic [{self.n-1}:0] r,\n" \
                           f"\tinput  logic [{self.n-1}:0] p,\n" \
                           f"\toutput logic       out\n);\n" \
                           f"\tassign out = ~(r < p);\n" \
                           f"endmodule\n\n\n"

        # Comparator array
        arr_len = len(inv_outs)
        file_string += f"module {self.name.lower()}_array_{ID} (\n" \
                       f"\tinput  logic [{self.n-1}:0] in [{arr_len-1}:0],\n" \
                       f"\tinput  logic [{self.n-1}:0] r,\n" \
                       f"\toutput logic       xs [{arr_len-1}:0]\n);\n"

        if corr_adj:
            # Handle inverted WBG P1 with inverted R input if corr_adj is used
            file_string += f"\tlogic [{self.n-1}:0] r_inv;\n" \
                           f"\tassign r_inv = ~r;\n\n"
            r_neg = "r_inv"
            spacing = " "*4
        else:
            r_neg = "r"
            spacing = ""

        for j in range(arr_len):
            if inv_outs[j] == 1:
                file_string += f"\tcompara_inv comp{j}(.r({r_neg}), .p(in[{j}]), .out(xs[{j}]));\n"
            else:
                file_string += f"\tcompara     comp{j}(.r(r), {spacing}.p(in[{j}]), .out(xs[{j}]));\n"

        file_string += f"endmodule\n\n\n"
        return file_string


if __name__ == '__main__':
    pass
