import time
import numpy as np

def generate_random_array(a, b, size):
    numbs = []
    steps = np.ceil((b-a)/size)
    l1 = a
    l2 = a+steps
    for i in range(size):
        numbs.append(np.random.randint(l1, l2) )
        l1 = l1+steps
        l2 = l2+steps       
        
    return np.array(numbs)

def shadowing_data(Cloc, var, p=0.98):
    # correlation E(z(x)z(x')) = var^2*exp(-|x-x'|/Xc)
    # if var is 0, return a matrix of zeros with the same size as Cloc
    if var == 0:
        shadowing_correlation = np.zeros_like(Cloc)
    else:
        m, n = Cloc.shape
        vec_Cloc = Cloc.reshape(-1)
        shadowing_iid = var * np.random.randn(m * n)
        vec_shadowing_iid = shadowing_iid.reshape(-1)
        R = lambda d: p ** d
        distance_corr = np.abs(np.subtract.outer(vec_Cloc, vec_Cloc))
        S = np.linalg.cholesky(R(distance_corr))
        vec_shadowing_correlation = S @ vec_shadowing_iid
        shadowing_correlation = vec_shadowing_correlation.reshape((m, n))
    return shadowing_correlation


def generate_map(I, J, K, R, shadow_sigma, Xc, basis, dB):
    seed = int(sum(100 * np.array(time.localtime())))
    s = np.random.RandomState(seed)
    indK = np.arange(1, K + 1)
    if basis == 'g':
        Sx = lambda f0, sigma: np.exp(-(indK - f0) ** 2 / (2 * sigma ** 2))
    else:
        Sx = lambda f0, a: np.sinc((indK - f0) / a) ** 2 * (np.abs((indK - f0) / a) <= 1)

    Ctrue = np.empty((K, 0))
    num_peaks_per_psd = 3

    for rr in range(R):
        psd_peaks = generate_random_array(2, K-5, num_peaks_per_psd)
        am = 0.5 + 1.5 * s.rand(num_peaks_per_psd, 1)
        c = am[0] * Sx(psd_peaks[0], 2 + 2 * s.rand())
        for q in range(1, num_peaks_per_psd):
            c += am[q] * Sx(psd_peaks[q], 2 + 2 * s.rand())

        Ctrue = np.hstack((Ctrue, c.reshape(-1, 1)))

    Ctrue = Ctrue / np.linalg.norm(Ctrue, axis=0) 

    loss_f = lambda x, d, alpha: np.minimum(1, (x / d) ** (-alpha))
    d0 = 2

    gridLen_x = I-1
    gridLen_y = J-1
    gridResolution = 1
    x_grid = np.arange(0, gridLen_x + gridResolution, gridResolution)
    y_grid = np.arange(0, gridLen_y + gridResolution, gridResolution)
    Xmesh_grid, Ymesh_grid = np.meshgrid(x_grid, y_grid)
    Xgrid = Xmesh_grid + 1j * Ymesh_grid
    I, J = Xgrid.shape

    Svec = []
    Sc = []
    peaks = []
    for rr in range(R):
        location = 50 * s.rand() + 50j * s.rand()
        peaks.append([int(np.real(location)), int(np.imag(location))])
        loss_mat = np.abs(Xgrid - location)
        alpha = 2 + 0.5 * s.rand()
        p = np.exp(-1 / Xc)
        shadow = shadowing_data(Xgrid, shadow_sigma, p)
        shadow_linear = 10 ** (shadow / 10)
        Sc.append( loss_f(loss_mat, d0, alpha) * shadow_linear )
        Sc[rr] = Sc[rr] / np.linalg.norm(Sc[rr], ord='fro')
        Svec.append(Sc[rr].reshape(-1))

    Svec = np.vstack(Svec)
    if dB:
        for rr in range(R):
            Sc[rr] = np.real(10 * np.log10(Sc[rr]))

    X = np.zeros((I, J, K))
    Sc = np.vstack([np.expand_dims(i, axis=0) for i in Sc])
    X = np.matmul(Sc.T, Ctrue.T).T


    return X, Sc, Ctrue, peaks


def column_normalization(X):
    m, n = X.shape
    Y = np.zeros((m, n))
    D = np.zeros(n)

    for ii in range(n):
        D[ii] = np.linalg.norm(X[:, ii])
        if D[ii] == 0:
            Y[:, ii] = X[:, ii]
        else:
            Y[:, ii] = X[:, ii] / D[ii]

    return Y, D