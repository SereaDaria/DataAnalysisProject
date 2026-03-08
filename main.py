import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hclust
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import fcluster


def nan_replace(tabel):
    """Tratarea valorilor lipsa folosind media pentru coloane numerice."""
    for col in tabel.columns:
        if tabel[col].isna().any():
            if is_numeric_dtype(tabel[col]):
                tabel[col].fillna(tabel[col].mean(), inplace=True)
            else:
                tabel[col].fillna(tabel[col].mode()[0], inplace=True)


def partitie(h, nr_clusteri, p, instante):
    """Extragerea clusterelor si desenarea dendrogramei colorate."""
    # Pozitia sectionarii in linkage matrix
    k_diff = p - nr_clusteri
    # Pragul unde sectionam dendrograma
    prag = (h[k_diff, 2] + h[k_diff + 1, 2]) / 2

    # Desenare
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_title(f"Clusterizare ierarhica - {nr_clusteri} clusteri optimi")
    hclust.dendrogram(h, labels=instante, ax=ax, color_threshold=prag, leaf_rotation=90)

    # Numar observatii
    n = p + 1
    # Initializam labels
    c = np.arange(n)

    # Simulare a reuniunilor intre clusteri
    for i in range(n - nr_clusteri):
        k1 = int(h[i, 0])
        k2 = int(h[i, 1])
        c[c == k1] = n + i
        c[c == k2] = n + i

    coduri = pd.Categorical(c).codes
    return np.array([f"C{cod + 1}" for cod in coduri])


def histograma(x, variabila, partitia):
    """Vizualizarea distributiei variabilelor pe fiecare cluster."""
    fig, axs = plt.subplots(1, len(np.unique(partitia)), figsize=(12, 4), sharey=True)
    fig.suptitle(f"Distributia variabilei: {variabila} pe clustere")

    for ax, cluster in zip(axs, np.unique(partitia)):
        ax.hist(x[partitia == cluster], bins=10, rwidth=0.9, color='skyblue', edgecolor='black')
        ax.set_title(cluster)


def execute():
    # Am setat index_col='Country' pentru ca aceasta sa fie eticheta pe grafic
    tabel = pd.read_csv("DateDSAD.csv", sep=';', decimal=',', index_col='Country')

    instante = list(tabel.index)
    variabile = list(tabel.columns)

    # Inlocuim eventualele valori lipsa
    nan_replace(tabel)

    # 3. Extragere date si Standardizare (Esentiala pentru Metoda Ward)
    x = tabel[variabile].values
    scaler = StandardScaler()
    x_std = scaler.fit_transform(x)

    # 4. Construim ierarhia de clusteri folosind metoda Ward
    h = hclust.linkage(x_std, method='ward')
    print(f'Linkage matrix (Matricea de legaturi): \n{h}')

    n = len(instante)
    p = n - 1

    # 5. Alegerea automata a numarului optim de clusteri (Metoda Elbow/Diferenta Maxima)
    # Comparam salturile de distanta intre etapele de agregare
    k_diff_max = np.argmax(h[1:, 2] - h[:-1, 2])
    nr_clusteri_optim = p - k_diff_max
    print(f"\n>>> Numar optim de clusteri detectat: {nr_clusteri_optim}")

    # 6. Extractie si export partitie optima
    partitie_optima = partitie(h, nr_clusteri_optim, p, instante)

    # Salvam rezultatul intr-un CSV pentru a-l folosi in tabelele din articol
    rezultate_df = pd.DataFrame(data={"Cluster": partitie_optima}, index=instante)
    rezultate_df.to_csv("Rezultate_Final_Clustere_UE.csv")
    print("\nRezultatele au fost exportate in 'Rezultate_Final_Clustere_UE.csv'")

    # 7. Grafice (Histograme pentru primii 3 indicatori din lista)
    for i in range(min(3, x.shape[1])):
        histograma(x[:, i], variabile[i], partitie_optima)

    # Afisare toate graficele
    plt.show()


if __name__ == '__main__':
    execute()