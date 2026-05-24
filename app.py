# ============================================
# TASK 1 : Iris Flower Classification App
# ============================================

import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ---- Entraîner le modèle ----
df = pd.read_csv("C:/Users/user/Downloads/Iris data - Copie.zip")
df = df.drop('Id', axis=1)

X = df.drop('Species', axis=1).values
le = LabelEncoder()
y = le.fit_transform(df['Species'].values)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# ============================================
# INTERFACE TKINTER
# ============================================

# ---- Fenêtre principale ----
fenetre = tk.Tk()
fenetre.title("🌸 Iris Flower Classification")
fenetre.geometry("500x600")
fenetre.configure(bg="#f0f4f8")
fenetre.resizable(False, False)

# ---- Titre ----
titre = tk.Label(
    fenetre,
    text="🌸 Iris Flower Classification",
    font=("Arial", 18, "bold"),
    bg="#f0f4f8",
    fg="#2c3e50"
)
titre.pack(pady=20)

sous_titre = tk.Label(
    fenetre,
    text="Entre les mesures de la fleur :",
    font=("Arial", 12),
    bg="#f0f4f8",
    fg="#7f8c8d"
)
sous_titre.pack()

# ---- Cadre des entrées ----
cadre = tk.Frame(fenetre, bg="white", 
                  relief="ridge", bd=2)
cadre.pack(pady=20, padx=30, fill="x")

# ---- Fonction pour créer les champs ----
def creer_champ(parent, label, valeur_defaut):
    frame = tk.Frame(parent, bg="white")
    frame.pack(pady=10, padx=20, fill="x")
    
    tk.Label(
        frame, text=label,
        font=("Arial", 11),
        bg="white", fg="#2c3e50",
        width=20, anchor="w"
    ).pack(side="left")
    
    entry = tk.Entry(
        frame,
        font=("Arial", 11),
        width=10,
        relief="solid",
        bd=1
    )
    entry.insert(0, valeur_defaut)
    entry.pack(side="left")
    
    tk.Label(
        frame, text="cm",
        font=("Arial", 11),
        bg="white", fg="#7f8c8d"
    ).pack(side="left", padx=5)
    
    return entry

# ---- Créer les 4 champs ----
sepal_length = creer_champ(cadre, "Sepal Length :", "5.1")
sepal_width  = creer_champ(cadre, "Sepal Width  :", "3.5")
petal_length = creer_champ(cadre, "Petal Length :", "1.4")
petal_width  = creer_champ(cadre, "Petal Width  :", "0.2")

# ---- Cadre résultat ----
cadre_resultat = tk.Frame(fenetre, bg="white",
                           relief="ridge", bd=2)
cadre_resultat.pack(pady=10, padx=30, fill="x")

resultat_label = tk.Label(
    cadre_resultat,
    text="Espèce : ---",
    font=("Arial", 14, "bold"),
    bg="white",
    fg="#2c3e50",
    pady=15
)
resultat_label.pack()

proba_label = tk.Label(
    cadre_resultat,
    text="",
    font=("Arial", 10),
    bg="white",
    fg="#7f8c8d",
    justify="left"
)
proba_label.pack(pady=5)

# ---- Fonction prédiction ----
def predire():
    try:
        # Récupérer les valeurs
        mesures = np.array([[
            float(sepal_length.get()),
            float(sepal_width.get()),
            float(petal_length.get()),
            float(petal_width.get())
        ]])

        # Prédire
        prediction = model.predict(mesures)
        espece = le.inverse_transform(prediction)[0]
        probas = model.predict_proba(mesures)[0]

        # Choisir couleur et emoji
        if 'setosa' in espece.lower():
            emoji = '🌸'
            couleur = '#e74c3c'
        elif 'versicolor' in espece.lower():
            emoji = '🌺'
            couleur = '#27ae60'
        else:
            emoji = '🌼'
            couleur = '#2980b9'

        # Afficher résultat
        resultat_label.config(
            text=f"{emoji} {espece}",
            fg=couleur
        )

        # Afficher probabilités
        proba_text = "Probabilités :\n"
        for i, classe in enumerate(le.classes_):
            proba_text += f"  {classe} : {probas[i]*100:.1f}%\n"
        proba_label.config(text=proba_text)

    except ValueError:
        messagebox.showerror(
            "Erreur",
            "Entre des nombres valides !"
        )

# ---- Fonction reset ----
def reset():
    sepal_length.delete(0, tk.END)
    sepal_width.delete(0, tk.END)
    petal_length.delete(0, tk.END)
    petal_width.delete(0, tk.END)
    sepal_length.insert(0, "5.1")
    sepal_width.insert(0, "3.5")
    petal_length.insert(0, "1.4")
    petal_width.insert(0, "0.2")
    resultat_label.config(text="Espèce : ---", fg="#2c3e50")
    proba_label.config(text="")

# ---- Boutons ----
cadre_boutons = tk.Frame(fenetre, bg="#f0f4f8")
cadre_boutons.pack(pady=20)

btn_predire = tk.Button(
    cadre_boutons,
    text="🌸 Prédire",
    command=predire,
    font=("Arial", 12, "bold"),
    bg="#27ae60",
    fg="white",
    width=12,
    height=2,
    relief="flat",
    cursor="hand2"
)
btn_predire.pack(side="left", padx=10)

btn_reset = tk.Button(
    cadre_boutons,
    text="🔄 Reset",
    command=reset,
    font=("Arial", 12, "bold"),
    bg="#e74c3c",
    fg="white",
    width=12,
    height=2,
    relief="flat",
    cursor="hand2"
)
btn_reset.pack(side="left", padx=10)

# ---- Lancer la fenêtre ----
fenetre.mainloop()