import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

# Визначення функції Розенброка
def rosenbrock(x):
    return 100 * (x[0]**2 - x[1])**2 + (x[0] - 1)**2

# Початкова точка
x0 = np.array([-1.2, 0])

# Визначення області (невипуклої) - півкола
def constraint(x):
    return x[0]**2 + x[1]**2 - 1

# Масштаб для візуалізації
x1 = np.linspace(-2, 2, 100)
x2 = np.linspace(-2, 2, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = rosenbrock([X2, X1])

# Мінімізація функції Розенброка з обмеженням eq (півколо)
result = minimize(rosenbrock, x0, method='trust-constr', constraints={'fun': constraint, 'type': 'eq'})

# Виведення результатів
print("Значення мінімізованої функції: ", result.fun)
print("Отримані значення змінних: ", result.x)

# Візуалізація функції та області
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Z, cmap='viridis')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('Мінімізація функції Розенброка з обмеженням eq (півколо)')

# Візуалізація області (півкола)
theta = np.linspace(0, np.pi, 100)
x = np.cos(theta)
y = np.sin(theta)
zeros = np.zeros_like(x)
ax.plot(x, y, zeros, color='red', label='Область (півколо)')

# Візуалізація мінімума
ax.scatter(result.x[0], result.x[1], result.fun, color='red', label='Мінімум')

plt.tight_layout()
plt.show()
