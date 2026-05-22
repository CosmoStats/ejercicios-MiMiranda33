import os
import glob
import numpy as np

N_DATA = 1701

# Modelos a comparar
MODELS = {
    "FlatLambdaCDM": {
        "folder": "chains-FlatLambdaCDM",
        "k": 4,
    },
    "LambdaCDM": {
        "folder": "chains-LambdaCDM",
        "k": 5,
    },
    "FlatwCDM": {
        "folder": "chains-FlatwCDM",
        "k": 5,
    },
}


def find_post_equal_weights(folder):

    pattern = os.path.join(folder, "*post_equal_weights.dat")
    files = glob.glob(pattern)

    if len(files) == 0:
        return None

    if len(files) > 1:
        print(f"\nAdvertencia: encontre varios archivos en {folder}:")
        for f in files:
            print("  ", f)
        print("Usare el primero:", files[0])

    return files[0]


def get_loglike_max(filename, k):

    data = np.loadtxt(filename)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    ncols = data.shape[1]

    if ncols < k + 1:
        raise ValueError(
            f"El archivo {filename} tiene {ncols} columnas, "
            f"pero se esperaban al menos {k + 1} para k={k}."
        )

    loglikes = data[:, -1]
    return np.max(loglikes)


def compute_aic_bic(loglike_max, k, n_data):
    aic = 2.0 * k - 2.0 * loglike_max
    bic = k * np.log(n_data) - 2.0 * loglike_max
    return aic, bic


rows = []

for model_name, info in MODELS.items():
    folder = info["folder"]
    k = info["k"]

    filename = find_post_equal_weights(folder)

    if filename is None:
        print(f"\nNo encontre archivo *_post_equal_weights.dat en {folder}")
        continue

    print(f"\nModelo: {model_name}")
    print(f"Carpeta: {folder}")
    print(f"Archivo usado: {filename}")
    print(f"Numero de parametros k = {k}")

    loglike_max = get_loglike_max(filename, k)
    aic, bic = compute_aic_bic(loglike_max, k, N_DATA)

    rows.append({
        "model": model_name,
        "folder": folder,
        "k": k,
        "loglike_max": loglike_max,
        "AIC": aic,
        "BIC": bic,
    })


if len(rows) == 0:
    raise RuntimeError("No se encontro ningun modelo valido para comparar.")


aic_min = min(r["AIC"] for r in rows)
bic_min = min(r["BIC"] for r in rows)

for r in rows:
    r["Delta_AIC"] = r["AIC"] - aic_min
    r["Delta_BIC"] = r["BIC"] - bic_min


print("\n" + "=" * 90)
print("Comparacion de modelos usando AIC")
print("=" * 90)
print(f"{'Modelo':20s} {'k':>3s} {'lnLmax':>15s} {'AIC':>15s} {'Delta AIC':>15s}")

for r in sorted(rows, key=lambda x: x["AIC"]):
    print(
        f"{r['model']:20s} "
        f"{r['k']:3d} "
        f"{r['loglike_max']:15.6f} "
        f"{r['AIC']:15.6f} "
        f"{r['Delta_AIC']:15.6f}"
    )


print("\n" + "=" * 90)
print("Comparacion de modelos usando BIC")
print("=" * 90)
print(f"{'Modelo':20s} {'k':>3s} {'lnLmax':>15s} {'BIC':>15s} {'Delta BIC':>15s}")

for r in sorted(rows, key=lambda x: x["BIC"]):
    print(
        f"{r['model']:20s} "
        f"{r['k']:3d} "
        f"{r['loglike_max']:15.6f} "
        f"{r['BIC']:15.6f} "
        f"{r['Delta_BIC']:15.6f}"
    )


best_aic = min(rows, key=lambda x: x["AIC"])
best_bic = min(rows, key=lambda x: x["BIC"])

print("\nModelo preferido por AIC:", best_aic["model"])
print("Modelo preferido por BIC:", best_bic["model"])
