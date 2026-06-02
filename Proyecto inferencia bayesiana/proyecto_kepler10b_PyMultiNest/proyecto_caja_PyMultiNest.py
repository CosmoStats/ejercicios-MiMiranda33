# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pymultinest


def box_model_phase(x, Tc, delta, Tdur, F0):
    in_transit = np.abs(x - Tc) < 0.5 * Tdur
    model = np.ones_like(x) * F0
    model[in_transit] -= delta
    return model


def summarize_samples(samples, labels):
    print("\nResumen posterior:")
    print("-" * 80)

    for i, lab in enumerate(labels):
        q16, q50, q84 = np.percentile(samples[:, i], [16, 50, 84])
        print(f"{lab:15s} = {q50:.8f} -{q50 - q16:.8f} +{q84 - q50:.8f}")

    print("-" * 80)


def save_corner_plot(samples, labels, basename):
    try:
        import corner

        fig = corner.corner(
            samples,
            labels=labels,
            quantiles=[0.16, 0.50, 0.84],
            show_titles=True,
            title_fmt=".6f",
        )

        corner_file = basename + "corner.png"
        fig.savefig(corner_file, dpi=160, bbox_inches="tight")
        plt.close(fig)

        print("\nCorner plot guardado en:")
        print(corner_file)

    except Exception as exc:
        print("\nNo se pudo generar el corner plot.")
        print("Error:", exc)


def save_fit_plot(x_fit, y_fit, yerr_fit, best_params, basename):
    Tc, delta, Tdur, F0, sigma = best_params

    x_model = np.linspace(-0.12, 0.12, 1000)
    y_model = box_model_phase(x_model, Tc, delta, Tdur, F0)

    plt.figure(figsize=(9, 4))

    plt.errorbar(
        x_fit,
        y_fit,
        yerr=yerr_fit,
        fmt=".",
        markersize=5,
        capsize=2,
        label="Datos procesados"
    )

    plt.plot(
        x_model,
        y_model,
        lw=2,
        label="Modelo tipo caja"
    )

    plt.axvline(
        Tc,
        linestyle="--",
        alpha=0.5,
        label=r"$T_c$ inferido"
    )

    plt.xlabel("Fase [dias]")
    plt.ylabel("Flujo normalizado")
    plt.title("Kepler-10 b: ajuste PyMultiNest con modelo tipo caja")
    plt.xlim(-0.12, 0.12)
    plt.ylim(0.9997, 1.0002)
    plt.legend()

    fit_file = basename + "fit.png"
    plt.savefig(fit_file, dpi=160, bbox_inches="tight")
    plt.close()

    print("\nGrafica del ajuste guardada en:")
    print(fit_file)


def save_data_plot(x_fit, y_fit, yerr_fit, basename):
    plt.figure(figsize=(9, 4))

    plt.errorbar(
        x_fit,
        y_fit,
        yerr=yerr_fit,
        fmt=".",
        markersize=5,
        capsize=2,
        label="Datos usados en el ajuste"
    )

    plt.xlabel("Fase [dias]")
    plt.ylabel("Flujo normalizado")
    plt.title("Kepler-10 b: datos procesados para PyMultiNest")
    plt.xlim(-0.12, 0.12)
    plt.ylim(0.9997, 1.0002)
    plt.legend()

    data_plot_file = basename + "data.png"
    plt.savefig(data_plot_file, dpi=160, bbox_inches="tight")
    plt.close()

    print("\nGrafica de datos guardada en:")
    print(data_plot_file)


def load_equal_weighted_samples(basename, ndim):
    try:
        analyzer = pymultinest.Analyzer(
            n_params=ndim,
            outputfiles_basename=basename
        )

        posterior = analyzer.get_equal_weighted_posterior()

        samples = posterior[:, :ndim]
        loglike_values = posterior[:, -1]

        return samples, loglike_values

    except Exception as exc:
        print("\nAnalyzer.get_equal_weighted_posterior() fallo.")
        print("Error:", exc)
        print("Intentando leer directamente post_equal_weights.dat...")

        filename = basename + "post_equal_weights.dat"

        data_equal = np.loadtxt(filename)

        samples = data_equal[:, :ndim]
        loglike_values = data_equal[:, -1]

        return samples, loglike_values


def read_evidence(basename, ndim):
    try:
        analyzer = pymultinest.Analyzer(
            n_params=ndim,
            outputfiles_basename=basename
        )

        stats = analyzer.get_stats()

        logZ = stats["global evidence"]
        logZerr = stats["global evidence error"]

        return logZ, logZerr

    except Exception as exc:
        print("\nNo se pudo leer la evidencia con Analyzer.")
        print("Error:", exc)
        return np.nan, np.nan


def main():
    parser = argparse.ArgumentParser(
        description="Inferencia de parametros de transito de Kepler-10 b con PyMultiNest usando modelo tipo caja."
    )

    parser.add_argument(
        "--data",
        type=str,
        default="data/kepler10b_datos_binned.txt"
    )

    parser.add_argument(
        "--nlive",
        type=int,
        default=500
    )

    parser.add_argument(
        "--tol",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--resume",
        action="store_true"
    )

    parser.add_argument(
        "--chains-dir",
        type=str,
        default="chains_box"
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results"
    )

    parser.add_argument(
        "--figures-dir",
        type=str,
        default="figures"
    )

    args = parser.parse_args()

    os.makedirs(args.chains_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)

    print("\nModelo: transito tipo caja")
    print("Objeto: Kepler-10 b")
    print("Archivo de datos:", args.data)

    data = np.loadtxt(args.data, skiprows=1)

    x_bin = data[:, 0]
    y_bin = data[:, 1]
    yerr_bin = data[:, 2]

    print("\nDatos binned cargados:")
    print(f"Numero total de puntos binned = {len(x_bin)}")
    print("Primeras filas:")
    print(data[:5])

    mask_fit = (x_bin > -0.12) & (x_bin < 0.12)

    x_fit = x_bin[mask_fit]
    y_fit = y_bin[mask_fit]
    yerr_fit = yerr_bin[mask_fit]

    print("\nDatos usados en el likelihood:")
    print(f"Numero de puntos = {len(x_fit)}")
    print(f"Rango de fase = [{x_fit.min()}, {x_fit.max()}]")

    labels = ["Tc", "delta", "Tdur", "F0", "sigma"]
    ndim = len(labels)

    bounds = {
        "Tc": (-0.03, 0.03),
        "delta": (0.0, 8.0e-4),
        "Tdur": (0.01, 0.12),
        "F0": (0.9998, 1.0002),
        "sigma": (1.0e-8, 1.0e-3),
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
        Tc, delta, Tdur, F0, sigma = [cube[i] for i in range(ndim)]

        if delta <= 0.0:
            return -1.0e100

        if Tdur <= 0.0:
            return -1.0e100

        if sigma <= 0.0:
            return -1.0e100

        model = box_model_phase(
            x_fit,
            Tc=Tc,
            delta=delta,
            Tdur=Tdur,
            F0=F0
        )

        if not np.all(np.isfinite(model)):
            return -1.0e100

        s2 = yerr_fit**2 + sigma**2

        if np.any(s2 <= 0.0) or not np.all(np.isfinite(s2)):
            return -1.0e100

        residuals = y_fit - model

        if not np.all(np.isfinite(residuals)):
            return -1.0e100

        logL = -0.5 * np.sum(
            residuals**2 / s2 + np.log(2.0 * np.pi * s2)
        )

        if not np.isfinite(logL):
            return -1.0e100

        return float(logL)

    basename = os.path.join(
        args.chains_dir,
        "box_model_"
    )

    figure_basename = os.path.join(
        args.figures_dir,
        "kepler10b_box_model_"
    )

    result_basename = os.path.join(
        args.results_dir,
        "kepler10b_box_model_"
    )

    save_data_plot(
        x_fit=x_fit,
        y_fit=y_fit,
        yerr_fit=yerr_fit,
        basename=figure_basename
    )

    print("\nCorriendo PyMultiNest...")
    print("Output basename:", basename)
    print("n_live_points:", args.nlive)
    print("evidence_tolerance:", args.tol)
    print("resume:", args.resume)

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

    logZ, logZerr = read_evidence(
        basename=basename,
        ndim=ndim
    )

    print("\nEvidencia bayesiana:")
    print("logZ =", logZ, "+/-", logZerr)

    samples, loglike_values = load_equal_weighted_samples(
        basename=basename,
        ndim=ndim
    )

    samples_file = result_basename + "equal_weighted_samples.txt"

    np.savetxt(
        samples_file,
        samples,
        header=" ".join(labels)
    )

    print("\nMuestras guardadas en:")
    print(samples_file)

    summarize_samples(samples, labels)

    q50 = np.percentile(samples, 50, axis=0)
    Tc_best, delta_best, Tdur_best, F0_best, sigma_best = q50

    Rp_Rs_best = np.sqrt(delta_best)

    print("\nParametro derivado:")
    print("Rp/Rs =", Rp_Rs_best)

    summary_file = result_basename + "summary.txt"

    with open(summary_file, "w") as f:
        f.write("Modelo tipo caja para Kepler-10 b con PyMultiNest\n")
        f.write("Parametros: Tc delta Tdur F0 sigma Rp_Rs logZ logZerr\n")
        f.write(
            f"{Tc_best:.12e} "
            f"{delta_best:.12e} "
            f"{Tdur_best:.12e} "
            f"{F0_best:.12e} "
            f"{sigma_best:.12e} "
            f"{Rp_Rs_best:.12e} "
            f"{logZ:.12e} "
            f"{logZerr:.12e}\n"
        )

    print("\nResumen numerico guardado en:")
    print(summary_file)

    loglike_file = result_basename + "loglike_values.txt"

    np.savetxt(
        loglike_file,
        loglike_values,
        header="loglike"
    )

    print("\nLog-likelihoods guardados en:")
    print(loglike_file)

    save_corner_plot(
        samples=samples,
        labels=labels,
        basename=figure_basename
    )

    save_fit_plot(
        x_fit=x_fit,
        y_fit=y_fit,
        yerr_fit=yerr_fit,
        best_params=q50,
        basename=figure_basename
    )

    print("\nScript terminado correctamente.")


if __name__ == "__main__":
    main()
