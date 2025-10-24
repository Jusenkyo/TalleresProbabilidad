import sympy as sp
import numpy as np
from sympy import binomial, integrate, symbols, Rational
import matplotlib.pyplot as plt

def problema_1_conjunta_discreta():
    """Problema 1: Distribución conjunta discreta - selección de estudiantes"""
    print("=== PROBLEMA 1: DISTRIBUCIÓN CONJUNTA DISCRETA ===")
    print("Selección de 2 estudiantes de: 5 Sistemas, 3 Electrónica, 3 Industrial")
    print("x = número de estudiantes de Sistemas")
    print("y = número de estudiantes de Electrónica")
    print("Total estudiantes: 5 + 3 + 3 = 11")
    
    # Espacio muestral total
    total_combinaciones = binomial(11, 2)
    print(f"\nTotal de formas de elegir 2 estudiantes: C(11,2) = {total_combinaciones}")
    
    # Función de probabilidad conjunta
    print("\n--- a) Función de probabilidad conjunta ---")
    
    # Valores posibles
    casos_posibles = []
    
    # Caso (0,0): 0 Sistemas, 0 Electrónica, 2 Industrial
    p_00 = binomial(3, 2) / total_combinaciones
    casos_posibles.append(((0,0), p_00))
    print(f"P(0,0) = C(3,2)/C(11,2) = {binomial(3,2)}/{total_combinaciones} = {p_00:.4f}")
    
    # Caso (0,1): 0 Sistemas, 1 Electrónica, 1 Industrial  
    p_01 = (binomial(3, 1) * binomial(3, 1)) / total_combinaciones
    casos_posibles.append(((0,1), p_01))
    print(f"P(0,1) = [C(3,1)*C(3,1)]/C(11,2) = {binomial(3,1)*binomial(3,1)}/{total_combinaciones} = {p_01:.4f}")
    
    # Caso (0,2): 0 Sistemas, 2 Electrónica, 0 Industrial
    p_02 = binomial(3, 2) / total_combinaciones
    casos_posibles.append(((0,2), p_02))
    print(f"P(0,2) = C(3,2)/C(11,2) = {binomial(3,2)}/{total_combinaciones} = {p_02:.4f}")
    
    # Caso (1,0): 1 Sistemas, 0 Electrónica, 1 Industrial
    p_10 = (binomial(5, 1) * binomial(3, 1)) / total_combinaciones
    casos_posibles.append(((1,0), p_10))
    print(f"P(1,0) = [C(5,1)*C(3,1)]/C(11,2) = {binomial(5,1)*binomial(3,1)}/{total_combinaciones} = {p_10:.4f}")
    
    # Caso (1,1): 1 Sistemas, 1 Electrónica, 0 Industrial
    p_11 = (binomial(5, 1) * binomial(3, 1)) / total_combinaciones
    casos_posibles.append(((1,1), p_11))
    print(f"P(1,1) = [C(5,1)*C(3,1)]/C(11,2) = {binomial(5,1)*binomial(3,1)}/{total_combinaciones} = {p_11:.4f}")
    
    # Caso (2,0): 2 Sistemas, 0 Electrónica, 0 Industrial
    p_20 = binomial(5, 2) / total_combinaciones
    casos_posibles.append(((2,0), p_20))
    print(f"P(2,0) = C(5,2)/C(11,2) = {binomial(5,2)}/{total_combinaciones} = {p_20:.4f}")
    
    # Verificar que suma = 1
    suma_total = sum(p for _, p in casos_posibles)
    print(f"\nVerificación: Suma de todas las probabilidades = {suma_total:.6f}")
    
    return casos_posibles, total_combinaciones

def problema_1_parte_b(casos_posibles):
    """Parte b: P(x,y) ∈ R donde R = {(x,y) | x + y ≤ 1}"""
    print("\n--- b) P(x,y) ∈ R donde R = {(x,y) | x + y ≤ 1} ---")
    
    probabilidad_R = 0
    puntos_en_R = []
    
    for (x, y), p in casos_posibles:
        if x + y <= 1:
            probabilidad_R += p
            puntos_en_R.append(((x, y), p))
            print(f"Punto ({x},{y}) está en R (x+y={x+y} ≤ 1), P = {p:.4f}")
    
    print(f"\nP((x,y) ∈ R) = {probabilidad_R:.4f}")
    return probabilidad_R, puntos_en_R

def problema_2_conjunta_continua():
    """Problema 2: Distribución conjunta continua - fábrica de dulces"""
    print("\n=== PROBLEMA 2: DISTRIBUCIÓN CONJUNTA CONTINUA ===")
    print("Función de densidad: f(x,y) = (2/5)(2x + 3y) para 0 ≤ x,y ≤ 1")
    
    x, y = symbols('x y')
    f_xy = Rational(2,5) * (2*x + 3*y)
    
    print(f"\n--- a) Verificar que f(x,y) es función de densidad ---")
    
    # Integrar en todo el dominio
    integral_doble = integrate(f_xy, (x, 0, 1), (y, 0, 1))
    print(f"∫∫ f(x,y) dx dy = ∫₀¹∫₀¹ ({f_xy}) dx dy = {integral_doble}")
    print(f"¿Es función de densidad? {integral_doble == 1}")
    
    return f_xy, x, y

def problema_2_parte_b(f_xy, x, y):
    """Parte b: Calcular probabilidad en región R"""
    print("\n--- b) Calcular P((x,y) ∈ R) ---")
    print("Basado en el PDF, parece que R = {(x,y) | 0 ≤ x ≤ 1/2, 1/4 ≤ y ≤ 1/2}")
    
    # Definir los límites basados en el PDF
    x_lim_inf = 0
    x_lim_sup = Rational(1, 2)  # y/2 según el PDF
    y_lim_inf = Rational(1, 4)
    y_lim_sup = Rational(1, 2)
    
    print(f"Región R: {x_lim_inf} ≤ x ≤ {x_lim_sup}, {y_lim_inf} ≤ y ≤ {y_lim_sup}")
    
    # Calcular la probabilidad
    probabilidad_R = integrate(f_xy, (x, x_lim_inf, x_lim_sup), (y, y_lim_inf, y_lim_sup))
    print(f"P((x,y) ∈ R) = ∫∫_R f(x,y) dx dy")
    print(f"= ∫_{y_lim_inf}²{ y_lim_sup} ∫_{x_lim_inf}²{ x_lim_sup} ({f_xy}) dx dy")
    print(f"= {probabilidad_R}")
    print(f"≈ {float(probabilidad_R):.6f}")
    
    return probabilidad_R

def problema_3_variables_independientes():
    """Problema 3: Análisis de variables independientes"""
    print("\n=== PROBLEMA 3: ANÁLISIS DE VARIABLES INDEPENDIENTES ===")
    print("f(x,y) = x + y = 30")
    
    # Esta parte del PDF parece incompleta, pero analicemos lo que hay
    print("\nAnálisis de los enunciados:")
    
    print("a) f(x,y=4) = 0 porque xy debe dar 30 no 4")
    print("   - Si x + y = 30, entonces f(26,4) sería válido matemáticamente")
    print("   - Probablemente hay un error en la interpretación")
    
    print("\nb) p(x,y=9) con x≥16, y≤15 hasta x=30, y=0")
    print("   - Si y=9, entonces x=21 para que x+y=30")
    print("   - x=21 ≥ 16 y y=9 ≤ 15, por lo tanto es un punto válido")
    
    print("\nc) p(x≥2, y≤1) - contradicción con x+y=30")
    print("   - Si y≤1 y x≥2, máximo x+y=31, mínimo=2")
    print("   - Para cumplir x+y=30, necesitamos y=1, x=29 o y=0, x=30")
    
    print("\nd) f(x≠2, y=1) = 0 - no es posible")
    print("   - Si y=1, entonces x=29 para que x+y=30")
    print("   - x=29 ≠ 2, pero SÍ es posible matemáticamente")

def graficar_distribucion_conjunta():
    """Graficar la distribución conjunta continua"""
    print("\n=== GRÁFICA DE LA DISTRIBUCIÓN CONJUNTA CONTINUA ===")
    
    # Crear malla de puntos
    x_vals = np.linspace(0, 1, 50)
    y_vals = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Evaluar función de densidad
    Z = (2/5) * (2*X + 3*Y)
    
    # Graficar
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    contour = plt.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(contour)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Función de Densidad Conjunta f(x,y)')
    
    # Marcar región R
    x_R = [0, 0.5, 0.5, 0]
    y_R = [0.25, 0.25, 0.5, 0.5]
    plt.fill(x_R, y_R, 'red', alpha=0.3, label='Región R')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')
    ax.set_title('Superficie de Densidad Conjunta')
    
    plt.tight_layout()
    plt.show()

def main():
    """Función principal"""
    print("SOLUCIÓN DEL TALLER 3 - DISTRIBUCIONES CONJUNTAS")
    print("=" * 70)
    
    # Problema 1: Distribución conjunta discreta
    casos_posibles, total_combinaciones = problema_1_conjunta_discreta()
    prob_R, puntos_R = problema_1_parte_b(casos_posibles)
    
    # Problema 2: Distribución conjunta continua
    f_xy, x, y = problema_2_conjunta_continua()
    prob_R_continua = problema_2_parte_b(f_xy, x, y)
    
    # Problema 3: Análisis adicional
    problema_3_variables_independientes()
    
    print("\n" + "=" * 70)
    print("RESUMEN DE RESULTADOS:")
    print(f"Problema 1 - P((x,y) ∈ R) = {prob_R:.4f}")
    print(f"Problema 2 - Verificación: ∫∫f(x,y)dxdy = 1 ✓")
    print(f"Problema 2 - P((x,y) ∈ R) = {float(prob_R_continua):.6f}")
    
    # Graficar (opcional)
    # graficar_distribucion_conjunta()

if __name__ == "__main__":
    main()