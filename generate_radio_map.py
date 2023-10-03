import numpy as np
import matplotlib.pyplot as plt
from radio_map_utils import generate_map, column_normalization
import argparse

parser = argparse.ArgumentParser(description='Generate a map for a wireless sensor network')

parser.add_argument('-R', '--emitters', type=int, default=6, help='Number of emitters')
parser.add_argument('-eta', '--shadow-variance', type=float, default=4.0, help='Shadow variance')
parser.add_argument('-Xc', '--decorrelation-distance', type=int, default=100, help='Decorrelation distance')
parser.add_argument('-s', '--snr', type=int, default=0, help='Signal-to-noise ratio')
parser.add_argument('-I', '--space-x', type=int, default=100, help='Map dimension I')
parser.add_argument('-J', '--space-y', type=int, default=100, help='Map dimension J')
parser.add_argument('-K', '--bandwidth-length', type=int, default=64, help='Map dimension K')
parser.add_argument('--psd-basis', type=str, default='g', help='Basis function for PSD generation, "s" for sinc, "g" for gaussian')
parser.add_argument('--save-file', type=str, default='radio_map', help='File name to save the generated map')
parser.add_argument('--visualize', type=bool, default=False, help='Visualize the radio map')

args = parser.parse_args()
print(vars(args))

R = args.emitters
shadow_sigma = args.shadow_variance
Xc = args.decorrelation_distance
snr = args.snr
I = args.space_x
J = args.space_y
K = args.bandwidth_length
save_file = args.save_file
visualize = args.visualize

def generate_radio_map():
    # the last argument determines the type of psd basis function 's': sinc 'g': gaussian
    T_true, Sc, C_true, _ = generate_map(I, J, K, R, shadow_sigma, Xc, args.psd_basis, dB=False)

    S_true = np.zeros((I, J, R))
    for rr in range(R):
        S_true[:, :, rr] = Sc[rr]

    C_true, _ = column_normalization(C_true)

    if snr != 0:
        Ps = np.linalg.norm(T_true)**2
        Pn = Ps*10**(-snr/10)
        sn = np.sqrt(Pn/I/J/K)
        if sn >= 1e2:
            sn = 0
        T = T_true + sn*1.73*np.random.randn(I, J, K)
    else:
        T = T_true

    if save_file != '':
        np.savez(save_file, T=T, C_true=C_true, S_true=S_true, T_true=T_true)


if __name__ == "__main__":
    generate_radio_map()