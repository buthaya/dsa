"""
EXERCICE 2 : Biais-Variance & Validation Croisée
================================================================
Objectif : Observer l'overfitting, tracer les courbes d'apprentissage,
et comprendre l'impact de la complexité du modèle.

Ce que vous allez apprendre :
- Visualiser underfitting et overfitting
- Tracer et interpréter des courbes d'apprentissage
- Utiliser cross_val_score pour comparer des modèles
- Diagnostiquer un modèle avec validation_curve

Temps estimé : 20 minutes
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (
    cross_val_score, learning_curve, validation_curve, KFold, TimeSeriesSplit
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# =============================================================
# PARTIE 1 : VISUALISER LE BIAIS-VARIANCE
# =============================================================
print("=" * 60)
print("PARTIE 1 : Visualisation Biais-Variance")
print("=" * 60)

# Générer des données synthétiques : y = sin(2πx) + bruit
np.random.seed(42)
n_samples = 50
X = np.sort(np.random.uniform(0, 1, n_samples))
y_true = np.sin(2 * np.pi * X)  # La vraie fonction (inconnue en pratique)
y = y_true + np.random.normal(0, 0.3, n_samples)  # Avec bruit

X = X.reshape(-1, 1)
X_plot = np.linspace(0, 1, 200).reshape(-1, 1)

# Fitter des polynômes de différents degrés
degrees = [1, 4, 15]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, degree in zip(axes, degrees):
    # Créer un pipeline : PolynomialFeatures + LinearRegression
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    
    # Prédictions
    y_pred_plot = model.predict(X_plot)
    y_pred_train = model.predict(X)
    
    # Erreur sur le train
    mse_train = mean_squared_error(y, y_pred_train)
    
    # Tracer
    ax.scatter(X, y, color='blue', alpha=0.5, label='Données (avec bruit)')
    ax.plot(X_plot, np.sin(2 * np.pi * X_plot), 'g--', label='Vraie fonction')
    ax.plot(X_plot, y_pred_plot, 'r-', linewidth=2, label=f'Polynôme degré {degree}')
    ax.set_title(f'Degré {degree}\nMSE train = {mse_train:.4f}')
    ax.set_ylim(-2, 2)
    ax.legend(fontsize=8)

plt.suptitle("Le dilemme Biais-Variance : 3 degrés de complexité", fontsize=14)
plt.tight_layout()
plt.savefig("biais_variance_demo.png", dpi=150)
plt.show()

# TODO : Répondez aux questions suivantes :
# 1. Quel modèle a la plus faible erreur d'entraînement ?
# 2. Quel modèle généralisera le mieux à de nouvelles données ? Pourquoi ?
# 3. Le modèle de degré 15 a une MSE train presque nulle. Est-ce bon signe ?

# =============================================================
# PARTIE 2 : COURBES D'APPRENTISSAGE
# =============================================================
print("\n" + "=" * 60)
print("PARTIE 2 : Courbes d'apprentissage")
print("=" * 60)

# Générer plus de données pour les learning curves
np.random.seed(42)
n_samples_lc = 300
X_lc = np.sort(np.random.uniform(0, 1, n_samples_lc)).reshape(-1, 1)
y_lc = np.sin(2 * np.pi * X_lc.ravel()) + np.random.normal(0, 0.3, n_samples_lc)

# Comparer deux modèles : un biaisé et un à haute variance
models = {
    'Polynôme degré 1 (biais élevé)': make_pipeline(PolynomialFeatures(1), LinearRegression()),
    'Polynôme degré 15 (variance élevée)': make_pipeline(PolynomialFeatures(15), LinearRegression()),
    'Polynôme degré 4 (bon compromis)': make_pipeline(PolynomialFeatures(4), LinearRegression()),
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, (name, model) in zip(axes, models.items()):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_lc, y_lc, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, 
        scoring='neg_mean_squared_error',
        random_state=42
    )
    
    # Convertir en erreur positive
    train_errors = -train_scores.mean(axis=1)
    val_errors = -val_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_std = val_scores.std(axis=1)
    
    ax.plot(train_sizes, train_errors, 'b-', label='Erreur Train')
    ax.fill_between(train_sizes, train_errors - train_std, train_errors + train_std, alpha=0.1, color='blue')
    ax.plot(train_sizes, val_errors, 'r-', label='Erreur Validation')
    ax.fill_between(train_sizes, val_errors - val_std, val_errors + val_std, alpha=0.1, color='red')
    ax.set_xlabel('Nombre d\'exemples d\'entraînement')
    ax.set_ylabel('MSE')
    ax.set_title(name)
    ax.legend()

    ax.set_yscale('log')

plt.suptitle("Courbes d'apprentissage — Diagnostic Biais vs Variance", fontsize=14)
plt.tight_layout()
plt.savefig("learning_curves.png", dpi=150)
plt.show()

# TODO : Interprétez les courbes :
# 1. Degré 1 : les courbes convergent-elles vers un score élevé ou faible ?
#    → Diagnostic : biais ou variance ?
#    → Action recommandée ?
#
# 2. Degré 15 : y a-t-il un grand écart entre train et validation ?
#    → Diagnostic : biais ou variance ?
#    → Action recommandée ?
#
# 3. Degré 4 : que voyez-vous ? Pourquoi est-ce le meilleur compromis ?

# =============================================================
# PARTIE 3 : VALIDATION CROISÉE POUR COMPARER DES MODÈLES
# =============================================================
print("\n" + "=" * 60)
print("PARTIE 3 : Comparaison par Cross-Validation")
print("=" * 60)

# Comparer les scores CV de plusieurs modèles
models_cv = {
    'Linéaire (degré 1)': make_pipeline(PolynomialFeatures(1), LinearRegression()),
    'Polynôme degré 3': make_pipeline(PolynomialFeatures(3), LinearRegression()),
    'Polynôme degré 4': make_pipeline(PolynomialFeatures(4), LinearRegression()),
    'Polynôme degré 10': make_pipeline(PolynomialFeatures(10), LinearRegression()),
    'Polynôme degré 4 + Ridge': make_pipeline(PolynomialFeatures(4), Ridge(alpha=1.0)),
    'Decision Tree (depth=3)': DecisionTreeRegressor(max_depth=3, random_state=42),
    'Decision Tree (depth=None)': DecisionTreeRegressor(max_depth=None, random_state=42),
}

print(f"\n{'Modèle':<30} {'Score CV (MSE)':<20} {'± Écart-type'}")
print("-" * 70)

results = {}
for name, model in models_cv.items():
    scores = cross_val_score(model, X_lc, y_lc, cv=5, scoring='neg_mean_squared_error')
    mean_score = -scores.mean()
    std_score = scores.std()
    results[name] = (mean_score, std_score)
    print(f"{name:<30} {mean_score:<20.4f} ± {std_score:.4f}")

# TODO : Répondez :
# 1. Quel modèle a le meilleur score CV ?
# 2. La Ridge sur le degré 4 améliore-t-elle ou empire-t-elle les choses ?
# 3. L'arbre profond a-t-il un bon ou mauvais score ? Pourquoi ?
# 4. Si vous deviez choisir UN modèle pour la production, lequel et pourquoi ?

# =============================================================
# PARTIE 4 : VALIDATION CURVE (impact d'un hyperparamètre)
# =============================================================
print("\n" + "=" * 60)
print("PARTIE 4 : Validation Curve — Impact de max_depth")
print("=" * 60)

# Tracer le score en fonction de la profondeur de l'arbre
param_range = [1, 2, 3, 4, 5, 7, 10, 15, 20, None]
# Note: validation_curve ne gère pas None, on va le faire manuellement

depths = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30]
train_scores_vc = []
val_scores_vc = []

for depth in depths:
    model = DecisionTreeRegressor(max_depth=depth, random_state=42)
    # Score sur le train (overfitting indicator)
    model.fit(X_lc, y_lc)
    train_pred = model.predict(X_lc)
    train_scores_vc.append(mean_squared_error(y_lc, train_pred))
    # Score CV
    scores = cross_val_score(model, X_lc, y_lc, cv=5, scoring='neg_mean_squared_error')
    val_scores_vc.append(-scores.mean())

plt.figure(figsize=(10, 6))
plt.plot(depths, train_scores_vc, 'b-o', label='Erreur Train')
plt.plot(depths, val_scores_vc, 'r-o', label='Erreur Validation (CV)')
plt.xlabel('max_depth de l\'arbre')
plt.ylabel('MSE')
plt.title('Validation Curve — Impact de max_depth sur un Decision Tree')
plt.legend()
plt.axvline(x=depths[np.argmin(val_scores_vc)], color='green', linestyle='--', 
            label=f'Optimal: depth={depths[np.argmin(val_scores_vc)]}')
plt.legend()
plt.savefig("validation_curve.png", dpi=150)
plt.show()

print(f"\nProfondeur optimale : {depths[np.argmin(val_scores_vc)]}")
print(f"MSE correspondante  : {min(val_scores_vc):.4f}")

# =============================================================
# PARTIE BONUS : TimeSeriesSplit vs KFold
# =============================================================
print("\n" + "=" * 60)
print("BONUS : TimeSeriesSplit vs KFold")
print("=" * 60)

# Simuler des données temporelles (tendance + saisonnalité + bruit)
np.random.seed(42)
n_time = 200
t = np.arange(n_time)
X_time = t.reshape(-1, 1)
# Signal : tendance + saisonnalité
y_time = 0.01 * t + np.sin(2 * np.pi * t / 12) + np.random.normal(0, 0.3, n_time)

# Comparaison KFold vs TimeSeriesSplit
model_time = DecisionTreeRegressor(max_depth=5, random_state=42)

# KFold classique (MAUVAIS pour données temporelles)
kfold_scores = cross_val_score(model_time, X_time, y_time, cv=KFold(5, shuffle=True), 
                                scoring='neg_mean_squared_error')

# TimeSeriesSplit (CORRECT pour données temporelles)
tscv_scores = cross_val_score(model_time, X_time, y_time, cv=TimeSeriesSplit(5), 
                               scoring='neg_mean_squared_error')

print(f"Score KFold (shuffle)    : {-kfold_scores.mean():.4f} ± {kfold_scores.std():.4f}")
print(f"Score TimeSeriesSplit    : {-tscv_scores.mean():.4f} ± {tscv_scores.std():.4f}")
print(f"\n→ Le KFold donne un score OPTIMISTE car il 'voit le futur'")
print(f"→ Le TimeSeriesSplit est plus réaliste pour un déploiement réel")

# =============================================================
# QUESTIONS FINALES DE RÉFLEXION
# =============================================================
"""
1. Pourquoi le score sur le train set n'est-il PAS un bon indicateur 
   de performance ?
   → Réponse : Parce qu'un modèle suffisamment complexe peut toujours 
     atteindre 0 erreur sur le train en mémorisant les données.

2. Que se passe-t-il si vous augmentez le nombre de folds dans la CV ?
   → Réponse : L'estimation devient plus précise mais plus coûteuse.
     À la limite (LOO), chaque observation est utilisée une fois en val.

3. En assurance, pourquoi un TimeSeriesSplit est-il souvent plus 
   approprié qu'un KFold ?
   → Réponse : Les sinistres ont une dimension temporelle (trends, 
     inflation, changements réglementaires). Un modèle déployé en 
     2024 ne verra que des données passées, pas futures.

4. Votre modèle a 95% d'accuracy sur le train et 60% sur le test.
   Diagnostic ? Action ?
   → Réponse : Overfitting (haute variance). Actions : simplifier le 
     modèle, ajouter de la régularisation, ou collecter plus de données.

5. Votre modèle a 62% d'accuracy sur le train ET sur le test.
   Diagnostic ? Action ?
   → Réponse : Underfitting (haut biais). Actions : modèle plus 
     complexe, plus de features, moins de régularisation.
"""