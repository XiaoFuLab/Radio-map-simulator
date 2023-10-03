from flask import Flask, request, jsonify, render_template, send_file
import io
import numpy as np
import matplotlib.pyplot as plt
from radio_map_utils import generate_map, column_normalization
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_radio_map')
def generate_radio_map():
    R = int(request.args.get('emitters', 2))
    shadow_sigma = float(request.args.get('shadow-variance', 4.0))
    Xc = int(request.args.get('decorrelation-distance', 100))
    snr = int(request.args.get('snr', 0))
    I = int(request.args.get('space-x', 51))
    J = int(request.args.get('space-y', 51))
    K = int(request.args.get('bandwidth-length', 64))
    visualize_K = int(request.args.get('visualize-bin', 25))
    psd_basis = request.args.get('psd-basis', 'g')

    args_dict = {}
    for key, value in request.args.items():
        args_dict[key] = value

    # the last argument determines the type of psd basis function 's': sinc 'g': gaussian
    T_true, Sc, C_true, _ = generate_map(I, J, K, R, shadow_sigma, Xc, psd_basis, dB=False)

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

    np.savez("radio_map.npz", T=T, C_true=C_true, S_true=S_true, T_true=T_true)

    # visualize the radio map
    rows = int(np.ceil(R/5.0))
    if rows==1:
        rows = 2
    cols = int(np.ceil((R+1)/ rows))
    fig, axs = plt.subplots(rows, cols, figsize=(cols*R, rows*R))
    for rr in range(R):
        i, j = np.unravel_index(rr, (rows, cols))
        axs[i, j].plot(C_true[:,rr])

    i, j = np.unravel_index(rr+1, (rows, cols))
    axs[i, j].imshow(np.log10(T_true[visualize_K].squeeze()), cmap="jet")
    plt.savefig('radio_map.png', bbox_inches='tight', pad_inches=0)

    # create download links for radio map and radio file
    radio_map_link = '/download_radio_map'
    radio_file_link = '/download_radio_file'

    # render HTML page with download links
    return render_template('download.html', radio_map_link=radio_map_link, radio_file_link=radio_file_link, variables= args_dict )


@app.route('/download_radio_map')
def download_radio_map():
    # return radio map file for download
    return send_file('radio_map.png', as_attachment=True)

@app.route('/download_radio_file')
def download_radio_file():
    # return radio file for download
    return send_file('radio_map.npz', as_attachment=True)

@app.route('/image')
def get_image():
    image_name = 'radio_map.png'
    return send_file(image_name, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)