import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Rosenbrock function
def rosenbrock(x):
    return 100*(x[0]**2 - x[1])**2 + (x[0] - 1)**2

# Gradient of Rosenbrock function
def rosenbrock_gradient(x):
    return np.array([400*x[0]*(x[0]**2 - x[1]) + 2*(x[0] - 1), -200*(x[0]**2 - x[1])])

# Termination criterion: ||∇f(x^((k)))||≤ε
def termination_criterion(gradient, epsilon):
    return np.linalg.norm(gradient) <= epsilon

# Markwardt method with the specified termination criterion
def markwardt_method(rosenbrock, rosenbrock_gradient, x0, epsilon):
    xk = x0
    fk = rosenbrock(xk)
    history = [fk]

    while True:
        gradient = rosenbrock_gradient(xk)

        result = minimize(rosenbrock, xk, method='trust-constr', jac=rosenbrock_gradient, options={'disp': False}, tol=None)
        xk = result.x
        fk = result.fun
        history.append(fk)

        if termination_criterion(gradient, epsilon):
            break

    return history

# Initial point
x0 = np.array([-1.2, 0])

# Termination criterion: ||∇f(x^((k)))||≤ε
epsilon = 1e-6

# Run Markwardt method with the specified termination criterion
history = markwardt_method(rosenbrock, rosenbrock_gradient, x0, epsilon)

# Print the minimized value
minimized_value = history[-1]
print("Мінімізоване значення цільової функції:", minimized_value)

# Plot the convergence history
plt.plot(history, label='Convergence')
plt.xlabel('Iterations')
plt.ylabel('Objective Function Value')
plt.legend()
plt.show()
