import numpy as np
from scipy import io
from scipy import signal
import matplotlib.pyplot as plt
from SCython.SNG import PCC
from SCython.IO import seq_utils
from SCython.Circuits import mux_adders
import SCython.SN_operations as SN_ops

# not modified.
def get_coefficients():
    # returns two sets of coefficients. h is the model filter and m is the masking filter.
    all_coefs = io.loadmat('data/coefficients/ifir_filter_coeff_1_July-7-2021.mat')
    h = all_coefs['h'][0]
    m = all_coefs['m'][0]
    return h, m


def get_ecg_coefficients(num_taps):
    # returns two sets of coefficients. h is the model filter and m is the masking filter.
    all_coefs = io.loadmat('data/coefficients/blo_order_4_268_July-7-2021.mat')
    h = all_coefs['blo']
    index = num_taps - 5
    return h[index, :num_taps]

def get_hearing_aid_coefficients(which):
    # returns two sets of coefficients. h is the model filter and m is the masking filter.
    all_coefs = io.loadmat('data/coefficients/nonuniform500_v2_July-7-2021.mat')
    coefs = all_coefs['all_coefs'][which]
    return coefs

# not modified.
def get_cemux(coefs, precision):
    pcc = PCC.Comparator
    vdc_seq = seq_utils.get_vdc(precision)
    cemux = mux_adders.CeMux(tree_height=precision, weights=coefs, pcc=pcc, vdc_seq=vdc_seq)
    return cemux


def get_cemuxes(model_coefs, mask_coefs, precision):
    pcc = PCC.Comparator
    vdc_seq = seq_utils.get_vdc(precision)
    model_cemux = mux_adders.CeMux(tree_height=precision, weights=model_coefs, pcc=pcc, vdc_seq=vdc_seq)
    mask_cemux = mux_adders.CeMux(tree_height=precision, weights=mask_coefs, pcc=pcc, vdc_seq=vdc_seq)
    return model_cemux, mask_cemux

# not modified
def gen_Z_hat(input_values, cemux, precision):
    Z = cemux.forward(input_values=input_values, SN_length=int(2 ** precision))
    Z_hat = SN_ops.get_SN_value(Z, bipolar=True)
    return Z_hat

# not modified or checked. Susan results suggest this method works fine
def get_sa_db(x, resp_db):
    # x (512,
    # all_resp(1,512)
    resp_db = np.atleast_2d(resp_db)
    cutoff_freqs = [0.0031,0.0656,0.3344, 0.3969, 0.4031,0.4656,0.7344,0.7969,0.8031,0.8656]
    plt.plot(x, resp_db[0], '-', lw=2, label="Frequency Response in Linear")
    sa_db = np.zeros((1, 3))
    for i in range(3):
        if i == 2:
            indx_mask = (x > cutoff_freqs[9])
        else:
            indx_mask = (x > cutoff_freqs[i*4+1]) & (x < cutoff_freqs[i*4+2])
        true_index = np.where(indx_mask==True)
        sa_db[0,i] = max(resp_db[0,true_index][0])
        plt.plot(x[true_index], resp_db[0,true_index][0], '-', lw=2,color='red',label="Frequency Response in Linear")
    plt.grid()
    plt.show(block=True)
    max_sa_db = np.max(sa_db)
    return max_sa_db


def get_single_freq_response_sim_sc(coefs, sc, prec, show_plot, label=None, db=True):
    # ignore all function parameters except "all_coefs". The other params are for future code
    num_freq = 512  # number of frequencies to test (i.e., number of x-axis values in freq response plot)
    delta_freq = 0.5 / num_freq  # this is stepsize of normalized frequency. Normalized frequency various from 0 to 0.5.

    # declare 2D array to hold the frequency response. shape is (num_channels, num_freq), data type is complex.
    actual_resp = np.zeros((1, num_freq), dtype=np.complex)
    tar_resp = np.zeros((1, num_freq))

    num_taps = len(coefs)  # number of filter taps. CODE MIGHT ONLY WORK IF THIS NUMBER IS ODD.
    # assert num_taps % 2 == 1, f'Number of taps:{num_taps} should be odd.'
    delay = (num_taps-1)//2  # delay of filter assuming that num_taps is odd.
    warm_up = num_taps + delay  # warm up period of filter. "warm_up" is the 1st sample that is valid.

    # declare arrays to hold filter input and output. One input and one output corresponds to cosine input, the other
    # input and output corresponds to sine input.
    cos_idx, sin_idx = 0, 1
    ins = np.zeros((2, warm_up))
    outs = np.zeros((2, 1))

    cemux = get_cemux(coefs, prec)
    abs_sum = sum(abs(coefs))
    # loop over the frequencies
    for idx in range(num_freq):
        freq = idx * delta_freq  # current normalized frequency
        omega = 2 * np.pi * freq  # current normalized radial frequency

        '''
        Let w be omega. Our input is x(n) = e^(jwn + j*phi) where phi is a phase shift. We compute phi such that
        x(M-1) = e^(jw(M-1) + j*phi) = 1. This obviously happens when phi = -w(M-1). So by setting x(n) = e^(jwn - j(M-1)w)
        we have that y(M-1) = H(w)x(M-1) becomes y(M-1) = H(w).
        '''
        phi = -(num_taps - 1) * omega

        for jdx in range(warm_up):
            # As implied by the last comment, the angle given to sine or cosine is (wn + phi) where w is omega.
            angle = omega * jdx + phi
            ins[cos_idx, jdx] = np.cos(angle)
            ins[sin_idx, jdx] = np.sin(angle)

            # we only need to compute y(M-1).
            start = delay
            stop = delay + num_taps
            if sc:
                outs[cos_idx] = abs_sum * gen_Z_hat(ins[cos_idx, start:stop], cemux, prec)
                outs[sin_idx] = abs_sum * gen_Z_hat(ins[sin_idx, start:stop], cemux, prec)

            else:
                outs[cos_idx] = np.inner(coefs, ins[cos_idx, start:stop])
                outs[sin_idx] = np.inner(coefs, ins[sin_idx, start:stop])

            actual_resp[0, idx] = complex(outs[cos_idx], outs[sin_idx])

    actual_response = actual_resp[0]
    ws, target_response = signal.freqz(coefs)
    x = ws / np.pi
    tar_resp[0] = abs(target_response)
    y_targ = 20 * np.log10(abs(target_response)) if db else abs(target_response)
    y_est = 20 * np.log10(abs(actual_response)) if db else abs(actual_response)
    if show_plot:
        plt.plot(x, y_targ, '-', lw=2, color='k', label="Calculated Frequency Response (target)")
        plt.plot(x, y_est, '--', lw=2, color='red', label="Measured Frequency Response")
        fs = 14
        plt.grid()
        plt.ylim(-100, 20)
        plt.xlabel(f"Normalized Frequency (Hz) {label} Filter", fontsize=fs)
        plt.ylabel("Frequency Response", fontsize=fs)
        plt.show(block=True)
    return x, y_targ, y_est


if __name__ == '__main__':
    model_coefs, mask_coefs = get_coefficients()
    # ecg_coefs = get_ecg_coefficients(23)
    # ha_coefs = get_hearing_aid_coefficients(10)
    print(f"Num taps in model filter:{len(model_coefs)} Num taps in mask filter:{len(mask_coefs)}")
    sc = True
    show_plot = True
    db = True

    prec = 15
    chosen_coefs = mask_coefs
    label = 'mask'
    get_single_freq_response_sim_sc(chosen_coefs, sc, prec, show_plot, label=label, db=db)
