"""Turbie 2-DOF model for DTU 46W38 Project 2.

y = [x_blade, x_tower, v_blade, v_tower]
"""
import numpy as np

PARAM_NAMES = (
    'mb', 'mn', 'mh', 'mt', 'c1', 'c2', 'k1', 'k2',
    'fb', 'ft', 'drb', 'drt', 'Dr', 'rho'
)


def read_parameters(path):
    # one value per line, # starts a comment
    vals = np.loadtxt(path, comments='#').ravel()
    if vals.size != len(PARAM_NAMES):
        raise ValueError(f'{path}: expected {len(PARAM_NAMES)} values, got {vals.size}')
    return dict(zip(PARAM_NAMES, vals))


def read_wind(path):
    data = np.loadtxt(path, skiprows=1)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(f'{path}: need two columns')
    t, u = data[:, 0], data[:, 1]
    if len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError(f'{path}: time must increase')
    return t, u


def read_ct_table(path):
    tab = np.loadtxt(path, comments='#')
    if tab.ndim != 2 or tab.shape[1] != 2:
        raise ValueError(f'{path}: need two columns')
    return tab[:, 0], tab[:, 1]


def interpolate_ct(mean_wind, speed, ct):
    # allow tiny rounding errors
    if mean_wind < speed[0] - 1e-3 or mean_wind > speed[-1] + 1e-3:
        raise ValueError(f'mean wind {mean_wind:.6f} outside CT table')
    mean_wind = np.clip(mean_wind, speed[0], speed[-1])
    return float(np.interp(mean_wind, speed, ct))


def build_matrices(p):
    m1 = 3 * p['mb']
    m2 = p['mn'] + p['mh'] + p['mt']
    c1, c2 = p['c1'], p['c2']
    k1, k2 = p['k1'], p['k2']
    M = np.diag([m1, m2])
    C = np.array([[c1, -c1], [-c1, c1 + c2]])
    K = np.array([[k1, -k1], [-k1, k1 + k2]])
    return M, C, K


def build_state_matrices(M, C, K):
    # y_dot = A*y + B*F
    Minv = np.diag(1 / np.diag(M))
    A = np.zeros((4, 4))
    A[:2, 2:] = np.eye(2)
    A[2:, :2] = -Minv @ K
    A[2:, 2:] = -Minv @ C
    # force only acts on blade mass
    B = np.zeros(4)
    B[2] = 1 / M[0, 0]
    return A, B


def thrust_force(w, ct, area, rho):
    # 0.5*rho*ct*A*w*|w|, w = u - v_blade
    return 0.5 * rho * ct * area * w * abs(w)


def state_derivative(t, y, A, B, wind_t, wind_u, ct, area, rho):
    u = np.interp(t, wind_t, wind_u)
    F = thrust_force(u - y[2], ct, area, rho)
    return A @ y + B * F