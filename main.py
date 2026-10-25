"""Run all Turbie 2DOF cases and save csv / plots."""

import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh

import turbie_mod as tm

ROOT = Path(__file__).resolve().parent
TURBIE_DIR = ROOT / 'inputs' / 'turbie_inputs'
WIND_DIR = ROOT / 'inputs' / 'wind_files'
OUT_DIR = ROOT / 'outputs'


def case_info(path):
    # get wind speed and TI from filename
    m = re.fullmatch(r'wind_(\d+)_ms_TI_(0\.\d+)\.txt', path.name)
    if not m:
        raise ValueError(f'bad filename: {path.name}')
    return int(m[1]), float(m[2])


def run_case(wind_file, params, A, B, ct_speed, ct_vals, out_dir,
             spinup=60.0, max_step=0.1, save_trace=True):
    t, u = tm.read_wind(wind_file)
    u_nom, ti = case_info(wind_file)
    mean_u = float(np.mean(u))
    ct = tm.interpolate_ct(mean_u, ct_speed, ct_vals)
    area = np.pi * (params['Dr'] / 2) ** 2

    sol = solve_ivp(
        tm.state_derivative,
        (t[0], t[-1]),
        np.zeros(4),
        t_eval=t,
        args=(A, B, t, u, ct, area, params['rho']),
        method='RK45',
        rtol=1e-4,
        atol=1e-6,
        max_step=max_step,
    )
    if not sol.success:
        raise RuntimeError(f'{wind_file.name}: {sol.message}')

    x1, x2, v1, v2 = sol.y
    F = tm.thrust_force(u - v1, ct, area, params['rho'])

    if save_trace:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            out_dir / f'result_{wind_file.name}',
            np.column_stack([t, u, x1, x2, v1, v2, F]),
            delimiter='\t',
            fmt='%.7f',
            header='time wind blade_x tower_x blade_v tower_v thrust',
            comments='',
        )

    # skip first spinup seconds to reduce start-up transient
    mask = t >= t[0] + spinup
    if not mask.any():
        raise ValueError('spinup is longer than simulation')

    row = {
        'nominal_wind_mps': u_nom,
        'measured_mean_wind_mps': mean_u,
        'nominal_TI': ti,
        'measured_TI': float(np.std(u) / mean_u),
        'CT': ct,
        'blade_mean_m': float(np.mean(x1[mask])),
        'blade_std_m': float(np.std(x1[mask])),
        'tower_mean_m': float(np.mean(x2[mask])),
        'tower_std_m': float(np.std(x2[mask])),
    }
    return row, (t, u, x1, x2)


def save_summary(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_example(path, t, u, x1, x2, title):
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()

    ax1.plot(t, u, color='gray', lw=0.7, alpha=0.8, label='Wind [m/s]')
    ax2.plot(t, x1, lw=1.1, label='Blade [m]')
    ax2.plot(t, x2, lw=1.1, label='Tower [m]')

    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Wind speed [m/s]')
    ax2.set_ylabel('Displacement [m]')
    ax1.set_title(title)
    ax1.grid(alpha=0.3)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='best')

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_comparison(path, all_rows, kind):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharex=True)

    for ti, rows in sorted(all_rows.items()):
        rows = sorted(rows, key=lambda r: r['nominal_wind_mps'])
        speed = [r['nominal_wind_mps'] for r in rows]
        axes[0].plot(speed, [r[f'blade_{kind}_m'] for r in rows],
                     'o-', ms=3, label=f'TI={ti:g}')
        axes[1].plot(speed, [r[f'tower_{kind}_m'] for r in rows],
                     'o-', ms=3, label=f'TI={ti:g}')

    label = 'Mean' if kind == 'mean' else 'Std'
    for ax, name in zip(axes, ['Blade', 'Tower']):
        ax.set_xlabel('Nominal wind speed [m/s]')
        ax.set_ylabel(f'{label} displacement [m]')
        ax.set_title(name)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle(f'{label} displacement vs wind speed and TI')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--single-ti', type=float)
    p.add_argument('--single-wind', type=int)
    p.add_argument('--spinup', type=float, default=60.0)
    p.add_argument('--max-step', type=float, default=0.1)
    p.add_argument('--no-traces', action='store_true',
                   help='skip per-case time series, useful for quick tests')
    args = p.parse_args()

    params = tm.read_parameters(TURBIE_DIR / 'turbie_parameters.txt')
    ct_speed, ct_vals = tm.read_ct_table(TURBIE_DIR / 'CT.txt')
    M, C, K = tm.build_matrices(params)
    A, B = tm.build_state_matrices(M, C, K)

    freq = np.sqrt(eigh(K, M)[0]) / (2 * np.pi)
    print('M =\n', M)
    print('C =\n', C)
    print('K =\n', K)
    print('Natural frequencies [Hz]:', freq)
    print('Rotor diameter [m]:', params['Dr'])

    files = sorted(
        WIND_DIR.glob('wind_TI_*/*.txt'),
        key=lambda f: (case_info(f)[1], case_info(f)[0]),
    )
    if args.single_ti is not None:
        files = [f for f in files if np.isclose(case_info(f)[1], args.single_ti)]
    if args.single_wind is not None:
        files = [f for f in files if case_info(f)[0] == args.single_wind]
    if not files:
        raise FileNotFoundError(f'no wind files found: {WIND_DIR}')

    OUT_DIR.mkdir(exist_ok=True)
    all_rows = {}
    sample_done = False

    for f in files:
        u_nom, ti = case_info(f)
        out = OUT_DIR / f'TI_{ti:g}'
        row, (t, u, x1, x2) = run_case(
            f, params, A, B, ct_speed, ct_vals, out,
            spinup=args.spinup,
            max_step=args.max_step,
            save_trace=not args.no_traces,
        )
        all_rows.setdefault(ti, []).append(row)

        print(f'TI={ti:.2f}, U={u_nom:2d} m/s; CT={row["CT"]:.3f}; '
              f'blade {row["blade_mean_m"]:.4f}/{row["blade_std_m"]:.4f} m; '
              f'tower {row["tower_mean_m"]:.4f}/{row["tower_std_m"]:.4f} m')

        # plot one sample case: 15 m/s, TI=0.1
        if not sample_done and np.isclose(ti, 0.1) and u_nom == 15:
            plot_example(OUT_DIR / 'example_wind_blade_tower.png',
                         t, u, x1, x2, '15 m/s, TI = 0.10')
            sample_done = True

    for ti, rows in sorted(all_rows.items()):
        save_summary(
            OUT_DIR / f'TI_{ti:g}' / 'summary.csv',
            sorted(rows, key=lambda r: r['nominal_wind_mps']),
        )

    plot_comparison(OUT_DIR / 'mean_displacement_vs_wind.png', all_rows, 'mean')
    plot_comparison(OUT_DIR / 'std_displacement_vs_wind.png', all_rows, 'std')
    print(f'Ran {len(files)} cases, results in {OUT_DIR}')


if __name__ == '__main__':
    main()