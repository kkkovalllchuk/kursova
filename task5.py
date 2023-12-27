import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Функція Розенброка
def rosenbrock(x):
    return 100 * (x[0]**2 - x[1])**2 + (x[0] - 1)**2

# Функція штрафу
def penalty(x):
    penalty = 0
    if np.any(x[0] < 1):
        penalty += (1 - x[0])**2
    if np.any(x[1] < 1):
        penalty += (1 - x[1])**2
    if np.any(x[0] > 2):
        penalty += (x[0] - 2)**2
    if np.any(x[1] > 3):
        penalty += (x[1] - 3)**2
    return penalty

# Функція, яка об'єднує функцію Розенброка та штрафну функцію
def objective(x):
    return rosenbrock(x) + penalty(x)

# Початкова точка
x0 = np.array([0, 0])

# Мінімізація функції з використанням методу Марквардта
result = minimize(objective, x0, method='trust-constr', options={'verbose': 1})

# Отримані результати
print("Мінімум:", result.x)
print("Значення функції в мінімумі:", result.fun)

# Візуалізація
x = np.linspace(-2, 4, 100)
y = np.linspace(-2, 4, 100)
X, Y = np.meshgrid(x, y)
Z = rosenbrock([X, Y]) + penalty([X, Y])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Y, X, Z, cmap='viridis')
ax.set_xlabel('x[0]')
ax.set_ylabel('x[1]')
ax.set_zlabel('Objective Function')
ax.scatter(result.x[0], result.x[1], result.fun, color='red', label='Minimum')
ax.legend()
plt.show()
