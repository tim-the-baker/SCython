import numpy as np
from scipy import io
from scipy import signal
import matplotlib.pyplot as plt
from SCython.SNG import PCC, SNG
from SCython.IO import seq_utils, input_value_chooser
from SCython.Circuits import mux_adders
import SCython.SN_operations as SN_ops

def get_filter_coefs(mode):
    coefs = None
    if mode == 'ifir':
        # TODO load IFIR (two stage) coefficients.
        coefs1 = None  # TODO load in first filter coefs
        coefs2 = None  # TODO laod in second filter coefs
        coefs = coefs1, coefs2
        raise NotImplementedError
    elif mode == 'fir':
        # TODO load FIR (single stage) coefficients
        coefs = None
        raise NotImplementedError

    return coefs


if __name__ == '__main__':
    # simulation parameters that you will often want to change
    precisions = np.arange(8, 12)  # vary precision from 8 bits to 16 bits
    pcc = PCC.Comparator  # don't change this for now. tells code to use comparator PCCs.
    input_mode = 'rand'  # see SCython.IO.input_value_chooser.choose_input_values for more details on this mode.
    bipolar = True  # True means to use bipolar SNs rather than unipolar SNs
    do_BC = True  # if True, track data for binary computing
    ifir_coefs = get_filter_coefs(mode='ifir')  # get the two sets of filter coefficients for IFIR design
    fir_coefs = get_filter_coefs(mode='fir')  # get the one set of filter coefficients for FIR design
    fir_input_size = len(fir_coefs)
    ifir1_input_size = len(ifir_coefs[0])
    ifir2_input_size = len(ifir_coefs[1])
    num_runs = 500  # increase this after testing code
    random_signal_size = num_runs + max(len(ifir_coefs[0]) + len(ifir_coefs[1]), len(fir_coefs)) + 1
    random_signal_size = 500  # increase this after testing code
    num_runs = random_signal_size + fir_input_size + 1

    # arrays to store data and other bookkeeping
    Z_hats = np.zeros((4, 2, len(precisions), num_runs))  # array for holding SC and BC estimated output values
    Z_stars = np.zeros(Z_hats.shape)  # array for holding exact (floating point) target output values
    sc_idx, bc_idx = 0, 1
    fir_idx, ifir1_idx, ifir2_idx, ifir_overall_idx = 0, 1, 2, 3
    data_labels = [["FIR SC", "FIR BC"],
                   ["IFIR Stage 1 SC", "IFIR Stage 1 BC"],
                   ["IFIR Stage 2 SC", "IFIR Stage 2 BC"],
                   ["IFIR Overall SC", "IFIR Overall BC"]]

    random_signal = input_value_chooser.choose_input_values(num_runs, bipolar, input_mode)
    # get the input values
    for prec_idx, precision in enumerate(precisions):
        # for use with SC
        tree_height = precision
        vdc_seq = seq_utils.get_vdc(precision)
        SN_length = int(2**precision)
        fir_cemux = mux_adders.CeMux(tree_height, fir_coefs, pcc, bipolar=bipolar, vdc_seq=vdc_seq)
        ifir1_cemux = mux_adders.CeMux(tree_height, ifir_coefs[0], pcc, bipolar=bipolar, vdc_seq=vdc_seq)
        ifir2_cemux = mux_adders.CeMux(tree_height, ifir_coefs[1], pcc, bipolar=bipolar, vdc_seq=vdc_seq)

        # for use with binary computing
        quantized_fir_coefs = SNG.q_nearest(fir_coefs, precision, signed=bipolar)
        quantized_ifir1_coefs = SNG.q_nearest(ifir_coefs[0], precision, signed=bipolar)
        quantized_ifir2_coefs = SNG.q_nearest(ifir_coefs[1], precision, signed=bipolar)

        # in this loop, we do SC FIR, BC FIR, SC IFIR stage 1. Next loop we do SC IFIR stage 2
        for r_idx in range(random_signal_size):
            # get this run's input values then get target outputs
            fir_input_values = random_signal[r_idx:r_idx+fir_input_size]
            ifir1_input_values = random_signal[r_idx:r_idx+ifir1_input_size]

            # FIR filter
            Z_stars[fir_idx, bc_idx, prec_idx, r_idx] = np.inner(fir_coefs, fir_input_values)
            Z_stars[fir_idx, sc_idx, prec_idx, r_idx] = Z_stars[fir_idx, bc_idx, prec_idx, r_idx]/np.sum(np.abs(fir_coefs))

            # IFIR stage 1
            Z_stars[ifir1_idx, bc_idx, prec_idx, r_idx] = np.inner(ifir_coefs[0], ifir1_input_values)
            Z_stars[ifir1_idx, sc_idx, prec_idx, r_idx] = Z_stars[ifir1_idx, bc_idx, prec_idx, r_idx]/np.sum(np.abs(ifir_coefs[0]))

            # Now do estimated values. First do FIR
            Z_hats[fir_idx, bc_idx, prec_idx, r_idx] = np.inner(quantized_fir_coefs, SNG.q_floor(fir_input_values, precision, bipolar))
            SN_Z = fir_cemux.forward(fir_input_values, SN_length)
            Z_hats[fir_idx, sc_idx, prec_idx, r_idx] = SN_ops.get_SN_value(SN_Z, bipolar)

            # Now do IFIR stage 1
            Z_hats[ifir1_idx, bc_idx, prec_idx, r_idx] = np.inner(quantized_ifir1_coefs, SNG.q_floor(ifir1_input_values, precision, bipolar))
            SN_Z = ifir1_cemux.forward(ifir1_input_values, SN_length)
            Z_hats[ifir1_idx, sc_idx, prec_idx, r_idx] = SN_ops.get_SN_value(SN_Z, bipolar)

        # in this loop, we do SC IFIR stage 2 and SC IFIR total
        for r_idx in range(num_runs):
            # TODO: this loop
            pass

    # compute the errors. This array contains the error of every simulation run.
    errors = Z_hats - Z_stars

    # TODO: dont forget to scale the binary errors.

    # once you have the errors, you can compute whatever aggregate metric you want. MSE, RMSE, etc.
    MSEs = np.mean(np.square(errors), axis=-1)  # axis=-1 means average over the last dimension
    RMSEs = np.sqrt(MSEs)

    # code for plotting. can also print the results.
    # google "how do I _ with matplotlib" and stack overflow should come up an answer
    for i in range(4):  # loop over the 4 filters errors: FIR, IFIR stage 1, IFIR stage 2, IFIR total
        for j in range(2):  # loop over SC and BC
            plt.plot(precisions, RMSEs[i, j], label=data_labels[i][j], lw=2)

    fs = 14
    plt.xlabel("Precision $n$. SN length=$2^n$", fontsize=fs)
    plt.xticks(fontsize=fs-2)
    plt.yticks(fontsize=fs-2)
    plt.ylabel("RMSEs", fontsize=fs)
    plt.legend()
    plt.tight_layout()
    plt.show()

