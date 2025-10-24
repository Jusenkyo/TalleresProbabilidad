import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from math import comb, factorial
from scipy.stats import poisson

# Problema 1: Probabilidades de selección de estudiantes
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('Problema 1: Probabilidades de Selección de Estudiantes', fontsize=16, fontweight='bold')

casos = ['a) 3 Elect.', 'b) 3 Sist.', 'c) 3 Ind.', 'd) Ningún Elect.', 'e) 1E, 1S, 1I', 'f) 2E, 1S']
sin_sust = [
    comb(8,3) / comb(20,3),
    comb(3,3) / comb(20,3),
    comb(9,3) / comb(20,3),
    comb(12,3) / comb(20,3),
    (comb(8,1)*comb(3,1)*comb(9,1)) / comb(20,3),
    (comb(8,2)*comb(3,1)) / comb(20,3)
]
con_sust = [
    (8/20)**3,
    (3/20)**3,
    (9/20)**3,
    (12/20)**3,
    6 * (8/20) * (3/20) * (9/20),
    3 * (8/20)**2 * (3/20)
]
x = np.arange(len(casos))
width = 0.35
ax = axes[0, 0]
ax.bar(x - width/2, sin_sust, width, label='Sin sustitución')
ax.bar(x + width/2, con_sust, width, label='Con sustitución')
ax.set_ylabel('Probabilidad')
ax.set_xticks(x)
ax.set_xticklabels(casos, rotation=45, ha='right')
ax.legend()

# Problema 2: Permutaciones de libros
libros = {'Ingeniería': 4, 'Inglés': 6, 'Física': 2}
perm_grupos_juntos = factorial(3) * factorial(4) * factorial(6) * factorial(2)
perm_total = factorial(sum(libros.values()))
fig2, ax2 = plt.subplots()
ax2.bar(['Grupos juntos', 'Sin restricciones'], [perm_grupos_juntos, perm_total])
ax2.set_yscale('log')
ax2.set_ylabel('Permutaciones')
ax2.set_title('Comparación de permutaciones (Tarea Libros)')

# Problema 3: Comité (5 ing, 7 abog, comité de 2 ing y 3 abog)
total_ing, total_abog, comite_ing, comite_abog = 5, 7, 2, 3
comites = comb(total_ing, comite_ing) * comb(total_abog, comite_abog)
fig3, ax3 = plt.subplots()
ax3.bar(['Total posibles'], [comites])
ax3.set_title('Número de comités posibles de 2 ingenieros y 3 abogados')

# Problema 6: Distribución de Poisson (λ = 1.5)
lambda_param = 1.5
k_vals = np.arange(0, 12)
pmf_vals = [poisson.pmf(k, lambda_param) for k in k_vals]
fig4, ax4 = plt.subplots()
ax4.bar(k_vals, pmf_vals)
ax4.set_xlabel('k (número de eventos)')
ax4.set_ylabel('P(X=k)')
ax4.set_title('Distribución de Poisson (λ=1.5)')

plt.tight_layout()
plt.show()