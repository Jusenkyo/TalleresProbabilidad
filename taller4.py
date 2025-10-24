import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm
import math

def pregunta_3_4_poisson():
    """Preguntas 3 y 4: Distribución Poisson P(X=3) con M=4"""
    print("=== PREGUNTAS 3 y 4: DISTRIBUCIÓN POISSON ===")
    print("X ~ Poisson(M=4), calcular P(X=3)")
    
    M = 4
    x = 3
    
    # Usando scipy
    prob_scipy = poisson.pmf(x, M)
    
    # Cálculo manual usando la fórmula
    prob_manual = (math.exp(-M) * (M ** x)) / math.factorial(x)
    
    print(f"Parámetros: M = {M}, x = {x}")
    print(f"Fórmula: P(X={x}) = (e^(-{M}) * {M}^{x}) / {x}!")
    print(f"Resultado scipy: {prob_scipy:.6f}")
    print(f"Resultado manual: {prob_manual:.6f}")
    print(f"Porcentaje: {prob_scipy * 100:.2f}%")
    
    return prob_scipy

def pregunta_5_6_poisson():
    """Preguntas 5 y 6: Distribución Poisson P(X=5) con M=4"""
    print("\n=== PREGUNTAS 5 y 6: DISTRIBUCIÓN POISSON ===")
    print("X ~ Poisson(M=4), calcular P(X=5)")
    
    M = 4
    x = 5
    
    # Usando scipy
    prob_scipy = poisson.pmf(x, M)
    
    # Cálculo manual usando la fórmula
    prob_manual = (math.exp(-M) * (M ** x)) / math.factorial(x)
    
    print(f"Parámetros: M = {M}, x = {x}")
    print(f"Fórmula: P(X={x}) = (e^(-{M}) * {M}^{x}) / {x}!")
    print(f"Resultado scipy: {prob_scipy:.6f}")
    print(f"Resultado manual: {prob_manual:.6f}")
    print(f"Porcentaje: {prob_scipy * 100:.2f}%")
    
    return prob_scipy

def pregunta_7_comparacion():
    """Pregunta 7: Comparación de probabilidades"""
    print("\n=== PREGUNTA 7: COMPARACIÓN ===")
    
    prob_3 = poisson.pmf(3, 4)
    prob_5 = poisson.pmf(5, 4)
    
    print(f"P(X=3) = {prob_3:.4f} = {prob_3 * 100:.2f}%")
    print(f"P(X=5) = {prob_5:.4f} = {prob_5 * 100:.2f}%")
    
    # Según el PDF menciona 19.54% y 15.63%
    print("\nComparación con valores del PDF:")
    print(f"Nuestro P(X=3): {prob_3 * 100:.2f}% vs PDF: 19.54%")
    print(f"Nuestro P(X=5): {prob_5 * 100:.2f}% vs PDF: 15.63%")
    
    diferencia_3 = abs(prob_3 * 100 - 19.54)
    diferencia_5 = abs(prob_5 * 100 - 15.63)
    
    print(f"Diferencia para P(X=3): {diferencia_3:.2f}%")
    print(f"Diferencia para P(X=5): {diferencia_5:.2f}%")

def problema_normal():
    """Problema de distribución normal - peso de baterías"""
    print("\n=== PROBLEMA DISTRIBUCIÓN NORMAL ===")
    print("Peso de baterías ~ N(μ=69, σ=2), P(X > 8)")
    
    # Del PDF parece que μ=6, σ=2, calcular P(X>8)
    # Pero hay inconsistencia en los valores, usaré los que parecen lógicos
    
    mu = 6  # media
    sigma = 2  # desviación estándar
    x_valor = 8  # valor para calcular P(X > 8)
    
    print(f"Parámetros: μ = {mu}, σ = {sigma}")
    print(f"Calcular: P(X > {x_valor})")
    
    # Calcular Z-score
    z_score = (x_valor - mu) / sigma
    print(f"Z-score: Z = ({x_valor} - {mu}) / {sigma} = {z_score}")
    
    # Calcular probabilidad usando scipy
    prob = 1 - norm.cdf(x_valor, mu, sigma)
    prob_z = 1 - norm.cdf(z_score)
    
    print(f"\nResultados:")
    print(f"P(X > {x_valor}) = 1 - P(X ≤ {x_valor})")
    print(f"Usando scipy directo: {prob:.6f}")
    print(f"Usando Z-score: {prob_z:.6f}")
    print(f"Porcentaje: {prob * 100:.2f}%")
    
    # Verificar con valores de la tabla normal
    print(f"\nVerificación con valores del PDF:")
    print(f"Nuestro resultado: {prob * 100:.2f}%")
    print(f"Valor en PDF: 15.87%")
    
    return prob, z_score

def graficar_poisson():
    """Graficar distribución de Poisson"""
    print("\n=== GRÁFICA DISTRIBUCIÓN POISSON ===")
    
    M = 4
    x_vals = np.arange(0, 11)
    probs = poisson.pmf(x_vals, M)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(x_vals, probs, color='skyblue', alpha=0.7)
    plt.axvline(x=3, color='red', linestyle='--', alpha=0.7, label='P(X=3)')
    plt.axvline(x=5, color='green', linestyle='--', alpha=0.7, label='P(X=5)')
    plt.xlabel('Número de clientes (x)')
    plt.ylabel('Probabilidad P(X=x)')
    plt.title(f'Distribución Poisson (M={M})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Destacar las probabilidades calculadas
    plt.text(3, poisson.pmf(3, M) + 0.01, f'P(X=3)={poisson.pmf(3, M):.3f}', 
             ha='center', color='red')
    plt.text(5, poisson.pmf(5, M) + 0.01, f'P(X=5)={poisson.pmf(5, M):.3f}', 
             ha='center', color='green')
    
    return x_vals, probs

def graficar_normal():
    """Graficar distribución normal"""
    print("\n=== GRÁFICA DISTRIBUCIÓN NORMAL ===")
    
    mu = 6
    sigma = 2
    x_critico = 8
    
    x_vals = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    y_vals = norm.pdf(x_vals, mu, sigma)
    
    plt.subplot(1, 2, 2)
    plt.plot(x_vals, y_vals, 'b-', linewidth=2, label=f'N({mu},{sigma}²)')
    
    # Sombrear área P(X > 8)
    x_fill = np.linspace(x_critico, mu + 4*sigma, 100)
    y_fill = norm.pdf(x_fill, mu, sigma)
    plt.fill_between(x_fill, y_fill, color='red', alpha=0.3, label=f'P(X > {x_critico})')
    
    plt.axvline(x=mu, color='black', linestyle='--', alpha=0.5, label=f'μ = {mu}')
    plt.axvline(x=x_critico, color='red', linestyle='--', alpha=0.7, label=f'x = {x_critico}')
    
    plt.xlabel('Peso de baterías')
    plt.ylabel('Densidad de probabilidad')
    plt.title('Distribución Normal - Peso de Baterías')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def calcular_todas_probabilidades_poisson():
    """Calcular todas las probabilidades Poisson para M=4"""
    print("\n=== TODAS LAS PROBABILIDADES POISSON PARA M=4 ===")
    
    M = 4
    print(f"Distribución Poisson con M = {M}")
    print("x\tP(X=x)\t\tPorcentaje")
    print("-" * 40)
    
    for x in range(0, 11):
        prob = poisson.pmf(x, M)
        print(f"{x}\t{prob:.6f}\t{prob * 100:.2f}%")
    
    # Probabilidades acumuladas
    print("\nProbabilidades acumuladas:")
    for x in range(0, 11):
        prob_acum = poisson.cdf(x, M)
        print(f"P(X ≤ {x}) = {prob_acum:.6f} = {prob_acum * 100:.2f}%")

def main():
    """Función principal"""
    print("SOLUCIÓN DEL TALLER - DISTRIBUCIÓN POISSON Y NORMAL")
    print("=" * 70)
    
    # Problemas de Poisson
    prob_3 = pregunta_3_4_poisson()
    prob_5 = pregunta_5_6_poisson()
    pregunta_7_comparacion()
    
    # Problema de distribución normal
    prob_normal, z_score = problema_normal()
    
    # Cálculos adicionales
    calcular_todas_probabilidades_poisson()
    
    print("\n" + "=" * 70)
    print("RESUMEN DE RESULTADOS:")
    print(f"Poisson P(X=3) = {prob_3:.6f} = {prob_3 * 100:.2f}%")
    print(f"Poisson P(X=5) = {prob_5:.6f} = {prob_5 * 100:.2f}%")
    print(f"Normal P(X>8) = {prob_normal:.6f} = {prob_normal * 100:.2f}%")
    print(f"Z-score = {z_score:.2f}")
    
    # Graficar distribuciones
    graficar_poisson()
    graficar_normal()

if __name__ == "__main__":
    main()