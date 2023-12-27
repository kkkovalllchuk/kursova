import numpy as np

# Функція Розенброка
def rosenbrock(x):
    return 100 * (x[0]**2 - x[1])**2 + (x[0] - 1)**2

# Градієнт функції Розенброка
def rosenbrock_gradient(x):
    grad = np.zeros(2)
    grad[0] = 400 * (x[0]**2 - x[1]) * x[0] + 2 * (x[0] - 1)
    grad[1] = -200 * (x[0]**2 - x[1])
    return grad

# Метод Марквардта
def marquardt_method(f, gradient, x0, lamda, epsilon, max_iter):
    x = x0.copy()
    for i in range(max_iter):
        grad = gradient(x)
        hessian = np.array([[1200 * x[0]**2 - 400 * x[1] + 2, -400 * x[0]], [-400 * x[0], 200]])
        hessian_inv = np.linalg.inv(hessian + lamda * np.eye(2))
        p = -np.dot(hessian_inv, grad)
        x_new = x + p
        if np.linalg.norm(p) < epsilon:
            return x_new, f(x_new), True  # Збіжність до розв'язку
        if f(x_new) < f(x):
            x = x_new
            lamda /= 10  # Зменшуємо параметр Марквардта
        else:
            lamda *= 10  # Збільшуємо параметр Марквардта

    return x, f(x), False  # Досягнуто максимальну кількість ітерацій

# Початкова точка
x0 = np.array([-1.2, 0])

# Параметри для дослідження збіжності
lambda_values = [0.001, 0.01, 0.1, 1, 10]
epsilon = 1e-6
max_iter = 100

# Збіжність методу для кожного значення параметра
for lamda in lambda_values:
    x, f, success = marquardt_method(rosenbrock, rosenbrock_gradient, x0, lamda, epsilon, max_iter)
    print(f"lambda = {lamda}: x = {x}, f(x) = {f}, success = {success}")
