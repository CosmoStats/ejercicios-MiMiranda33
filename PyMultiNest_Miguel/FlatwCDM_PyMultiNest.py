# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np

from scipy.linalg import cho_factor, cho_solve
from scipy.integrate import cumulative_trapezoid

import pymultinest


C_LIGHT = 299792.458  # km/s

# Magnitud absoluta fija
M_FIXED = -19.291293562581636


def load_pantheon_covariance(filename):
    raw = np.loadtxt(filename)
    n = int(raw[0])
    values = raw[1:]

    if values.size != n * n:
        raise ValueError(
            f"El archivo de covarianza no tiene el tamano esperado. "
            f"N={n}, pero hay {values.size} elementos."
        )

    return values.reshape((n, n))


def mu_theory_flat_wcdm(z, H0, Om, w0, n_grid=6000):
 
    z = np.asarray(z, dtype=float)

    if np.any(z < 0.0):
        return None

    if H0 <= 0.0:
        return None

    if Om <= 0.0 or Om >= 1.0:
        return None

    Ode = 1.0 - Om

    zmax = np.max(z)

    if zmax <= 0.0:
        return np.zeros_like(z)

    zg = np.linspace(0.0, zmax, n_grid)

    Ez2 = (
        Om * (1.0 + zg) ** 3
        + Ode * (1.0 + zg) ** (3.0 * (1.0 + w0))
    )

    if np.any(Ez2 <= 0.0) or not np.all(np.isfinite(Ez2)):
        return None

    integral_grid = cumulative_trapezoid(
        1.0 / np.sqrt(Ez2),
        zg,
        initial=0.0
    )

    Iz = np.interp(z, zg, integral_grid)

    dL = (1.0 + z) * (C_LIGHT / H0) * Iz  # Mpc

    if np.any(dL <= 0.0) or not np.all(np.isfinite(dL)):
        return None

    mu = 5.0 * np.log10(dL) + 25.0

    return mu


def summarize_samples(samples, labels):
    print("\nResumen posterior:")
    print("-" * 80)

    for i, lab in enumerate(labels):
        q16, q50, q84 = np.percentile(samples[:, i], [16, 50, 84])
        print(f"{lab:15s} = {q50:.6f} -{q50 - q16:.6f} +{q84 - q50:.6f}")

    print("-" * 80)


def save_corner_plot(samples, labels, basename):
    try:
        import corner
        import matplotlib.pyplot as plt

        fig = corner.corner(
            samples,
            labels=labels,
            quantiles=[0.16, 0.50, 0.84],
            show_titles=True,
            title_fmt=".4f",
        )

        corner_file = basename + "corner.png"
        fig.savefig(corner_file, dpi=160, bbox_inches="tight")
        plt.close(fig)

        print("\nCorner plot guardado en:")
        print(corner_file)

    except Exception as exc:
        print("\nNo se pudo generar el corner plot.")
        print("Error:", exc)
        print("Puedes instalar corner con: pip install corner")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Inferencia Pantheon+SH0ES para un modelo wCDM plano "
            "usando PyMultiNest, con M fijo."
        )
    )

    parser.add_argument(
        "--data",
        type=str,
        default="Pantheon+SH0ES.dat",
        help="Archivo de datos Pantheon+SH0ES."
    )

    parser.add_argument(
        "--cov",
        type=str,
        default="Pantheon+SH0ES_STATONLY.cov",
        help="Archivo de covarianza."
    )

    parser.add_argument(
        "--nlive",
        type=int,
        default=1200,
        help="Numero de live points de MultiNest."
    )

    parser.add_argument(
        "--tol",
        type=float,
        default=0.1,
        help="Evidence tolerance de MultiNest."
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continuar una corrida previa si existe."
    )

    parser.add_argument(
        "--chains-dir",
        type=str,
        default="chains-wCDM",
        help="Carpeta donde se guardan las cadenas."
    )

    args = parser.parse_args()

    os.makedirs(args.chains_dir, exist_ok=True)

    print("\nModelo: wCDM plano")
    print("Caso: M fijo, H0, alpha, beta, Omega_M y w0 libres")
    print(f"M fijo = {M_FIXED}")
    print("Condicion: Omega_DE = 1 - Omega_M")
    print("No se separan calibradores de Cefeidas.")

    print("\nLeyendo datos:", args.data)

    data = np.genfromtxt(
        args.data,
        names=True,
        dtype=None,
        encoding=None
    )

    zHD = data["zHD"].astype(float)
    mB = data["mB"].astype(float)
    x1 = data["x1"].astype(float)
    color = data["c"].astype(float)
    biasCor = data["biasCor_m_b"].astype(float)

    n_data = len(zHD)

    print(f"Numero total de datos = {n_data}")
    print("Todos los datos se comparan contra mu_model(z).")

    print("\nLeyendo covarianza:", args.cov)

    cov = load_pantheon_covariance(args.cov)

    if cov.shape != (n_data, n_data):
        raise ValueError(
            f"La covarianza tiene forma {cov.shape}, "
            f"pero los datos tienen longitud {n_data}."
        )

    print("Factorizando covarianza con Cholesky...")

    cho = cho_factor(cov, lower=True, check_finite=False)

    def chi2_cov(residuals):
        sol = cho_solve(cho, residuals, check_finite=False)
        return float(np.dot(residuals, sol))

    labels = ["H0", "alpha", "beta", "Omega_M", "w0"]
    ndim = len(labels)

    bounds = {
        "H0": (50.0, 90.0),
        "alpha": (0.0, 0.4),
        "beta": (1.0, 5.0),
        "Omega_M": (0.05, 0.60),
        "w0": (-2.5, -0.3),
    }

    print("\nParametros inferidos:", labels)
    print("Priors uniformes:")

    for lab in labels:
        print(f"  {lab}: {bounds[lab]}")

    def prior(cube, ndim, nparams):
        for i, lab in enumerate(labels):
            lo, hi = bounds[lab]
            cube[i] = lo + cube[i] * (hi - lo)


    def loglike(cube, ndim, nparams):
        H0, alpha, beta, Om, w0 = [cube[i] for i in range(ndim)]

        if H0 <= 0.0:
            return -1.0e100

        if Om <= 0.0 or Om >= 1.0:
            return -1.0e100

        # Distancia observada tipo Tripp con M fijo
        mu_obs = mB + alpha * x1 - beta * color - M_FIXED - biasCor

        # Distancia teorica para wCDM plano
        mu_th = mu_theory_flat_wcdm(
            zHD,
            H0=H0,
            Om=Om,
            w0=w0
        )

        if mu_th is None:
            return -1.0e100

        # todos los datos se comparan contra el modelo cosmologico.
        residuals = mu_obs - mu_th

        if not np.all(np.isfinite(residuals)):
            return -1.0e100

        chi2 = chi2_cov(residuals)

        if not np.isfinite(chi2):
            return -1.0e100

        return -0.5 * chi2

    basename = os.path.join(
        args.chains_dir,
        "wcdm_flat_fixed_M_5params_"
    )

    print("\nCorriendo PyMultiNest...")
    print("Output basename:", basename)
    print("n_live_points:", args.nlive)
    print("evidence_tolerance:", args.tol)

    pymultinest.run(
        loglike,
        prior,
        ndim,
        outputfiles_basename=basename,
        n_live_points=args.nlive,
        evidence_tolerance=args.tol,
        sampling_efficiency="model",
        resume=args.resume,
        verbose=True,
    )


    print("\nLeyendo resultados de MultiNest...")

    analyzer = pymultinest.Analyzer(
        n_params=ndim,
        outputfiles_basename=basename
    )

    stats = analyzer.get_stats()

    print("\nEvidencia bayesiana:")
    print(
        "logZ =",
        stats["global evidence"],
        "+/-",
        stats["global evidence error"]
    )

    posterior = analyzer.get_equal_weighted_posterior()

    samples = posterior[:, :ndim]

    samples_file = basename + "equal_weighted_samples.txt"

    np.savetxt(
        samples_file,
        samples,
        header=" ".join(labels)
    )

    print("\nMuestras guardadas en:")
    print(samples_file)

    summarize_samples(samples, labels)
    save_corner_plot(samples, labels, basename)


if __name__ == "__main__":
    main()
