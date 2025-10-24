import sympy as sp
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

def problema_valor_esperado():
    """Valor esperado de un dado y suma de dos dados"""
    print("=== PROBLEMA 1: VALOR ESPERADO Y VARIANZA ===")
    
    # Valor esperado de un dado
    resultados_dado = [1, 2, 3, 4, 5, 6]
    E_xi = np.mean(resultados_dado)
    print(f"E[X_i] = (1+2+3+4+5+6)/6 = {sum(resultados_dado)}/6 = {E_xi}")
    
    # Valor esperado de la suma S = X1 + X2
    E_S = E_xi + E_xi
    print(f"E[S] = E[X1] + E[X2] = {E_xi} + {E_xi} = {E_S}")
    
    # Varianza de un dado
    # E[X_i^2]
    E_xi2 = np.mean([x**2 for x in resultados_dado])
    print(f"E[X_i²] = (1+4+9+16+25+36)/6 = 91/6 = {E_xi2:.4f}")
    
    Var_xi = E_xi2 - E_xi**2
    print(f"Var(X_i) = E[X_i²] - (E[X_i])² = {E_xi2:.4f} - ({E_xi})² = {Var_xi:.4f}")
    
    # Verificación con fórmula exacta
    Var_xi_exacta = sp.Rational(35, 12)
    print(f"Var(X_i) exacta = 35/12 = {float(Var_xi_exacta):.4f}")
    
    # Varianza de la suma S = X1 + X2
    Var_S = Var_xi + Var_xi
    print(f"Var(S) = Var(X1) + Var(X2) = {Var_xi:.4f} + {Var_xi:.4f} = {Var_S:.4f}")
    Var_S_exacta = sp.Rational(35, 6)
    print(f"Var(S) exacta = 35/6 = {float(Var_S_exacta):.4f}")

def problema_funcion_densidad():
    """Función de densidad de probabilidad f(x) = kx²"""
    print("\n=== PROBLEMA 2: FUNCIÓN DE DENSIDAD ===")
    
    # Definir símbolos
    x, k = sp.symbols('x k')
    
    # Función de densidad: f(x) = kx² para 0 < x < 6
    f_x = k * x**2
    
    # Calcular k tal que ∫f(x)dx = 1 en [0,6]
    integral = sp.integrate(f_x, (x, 0, 6))
    print(f"∫₀⁶ kx² dx = {integral}")
    
    # Resolver para k
    ecuacion = sp.Eq(integral, 1)
    k_valor = sp.solve(ecuacion, k)[0]
    print(f"k = {k_valor} = {float(k_valor):.4f}")
    
    # Función de densidad completa
    f_x_completa = f_x.subs(k, k_valor)
    print(f"f(x) = {f_x_completa} para 0 < x < 6")
    
    return f_x_completa, k_valor

def calcular_probabilidad(f_x_completa, a, b):
    """Calcular P(a < X < b)"""
    x = sp.symbols('x')
    probabilidad = sp.integrate(f_x_completa, (x, a, b))
    print(f"P({a} < X < {b}) = ∫_{a}^{b} {f_x_completa} dx = {probabilidad}")
    print(f"Valor numérico: {float(probabilidad):.4f}")
    return probabilidad

def problema_grafica_densidad(f_x_completa, k_valor):
    """Graficar la función de densidad"""
    print("\n=== GRÁFICA DE LA FUNCIÓN DE DENSIDAD ===")
    
    # Convertir a función numérica
    x_vals = np.linspace(0, 6, 100)
    f_num = sp.lambdify(sp.symbols('x'), f_x_completa, 'numpy')
    y_vals = f_num(x_vals)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, 'b-', linewidth=2, label=f'f(x) = {float(k_valor):.4f}x²')
    plt.fill_between(x_vals, y_vals, alpha=0.3, color='blue')
    plt.title('Función de Densidad de Probabilidad')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 6)
    plt.ylim(0, max(y_vals) * 1.1)
    plt.show()

def verificaciones_adicionales(f_x_completa):
    """Verificaciones adicionales de la función de densidad"""
    print("\n=== VERIFICACIONES ADICIONALES ===")
    
    x = sp.symbols('x')
    
    # Verificar que ∫f(x)dx = 1 en [0,6]
    integral_total = sp.integrate(f_x_completa, (x, 0, 6))
    print(f"Verificación: ∫₀⁶ f(x)dx = {integral_total} = {float(integral_total)}")
    
    # Calcular valor esperado E[X]
    E_X = sp.integrate(x * f_x_completa, (x, 0, 6))
    print(f"E[X] = ∫₀⁶ x·f(x)dx = {E_X} = {float(E_X):.4f}")
    
    # Calcular E[X²]
    E_X2 = sp.integrate(x**2 * f_x_completa, (x, 0, 6))
    print(f"E[X²] = ∫₀⁶ x²·f(x)dx = {E_X2} = {float(E_X2):.4f}")
    
    # Calcular varianza Var(X)
    Var_X = E_X2 - E_X**2
    print(f"Var(X) = E[X²] - (E[X])² = {float(E_X2):.4f} - ({float(E_X):.4f})² = {float(Var_X):.4f}")

def main():
    """Función principal"""
    print("SOLUCIÓN DEL TALLER DE PROBABILIDAD - VALOR ESPERADO Y FUNCIONES DE DENSIDAD")
    print("=" * 70)
    
    # Resolver problema 1: Valor esperado y varianza
    problema_valor_esperado()
    
    # Resolver problema 2: Función de densidad
    f_x_completa, k_valor = problema_funcion_densidad()
    
    # Calcular probabilidad P(1 < X < 5)
    print("\n--- Cálculo de P(1 < X < 5) ---")
    P_1_5 = calcular_probabilidad(f_x_completa, 1, 5)
    
    # Otras probabilidades útiles
    print("\n--- Otras probabilidades ---")
    calcular_probabilidad(f_x_completa, 0, 3)
    calcular_probabilidad(f_x_completa, 3, 6)
    calcular_probabilidad(f_x_completa, 2, 4)
    
    # Verificaciones adicionales
    verificaciones_adicionales(f_x_completa)
    
    # Graficar (opcional - descomentar si quieres ver la gráfica)
    # problema_grafica_densidad(f_x_completa, k_valor)
    
    print("\n" + "=" * 70)
    print("RESUMEN DE RESULTADOS:")
    print(f"• E[X_i] = 3.5")
    print(f"• E[S] = 7.0") 
    print(f"• Var(X_i) = 35/12 ≈ 2.9167")
    print(f"• Var(S) = 35/6 ≈ 5.8333")
    print(f"• k = 1/72 ≈ 0.01389")
    print(f"• P(1 < X < 5) = {float(P_1_5):.4f}")

if __name__ == "__main__":
    main()