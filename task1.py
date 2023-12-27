import numpy as np
from scipy.optimize import minimize

def rosenbrock(x):
    return 100 * (x[0]**2 - x[1])**2 + (x[0]-1)**2

def rosenbrock_gradient(x, h):
    grad = np.zeros(2)

    grad[0] = (rosenbrock([x[0] + h, x[1]]) - rosenbrock([x[0] - h, x[1]])) / (2 * h)
    grad[1] = (rosenbrock([x[0], x[1] + h]) - rosenbrock([x[0], x[1] - h])) / (2 * h)

    return grad

def rosenbrock_hessian(x, h):
    hess = np.zeros((2, 2))
    hess[0, 0] = (rosenbrock([x[0] + h, x[1]]) - 2 * rosenbrock([x[0], x[1]]) + rosenbrock([x[0] - h, x[1]])) / (h**2)
    hess[0, 1] = (rosenbrock([x[0] + h, x[1] + h]) - rosenbrock([x[0] + h, x[1] - h]) - rosenbrock([x[0] - h, x[1] + h]) + rosenbrock([x[0] - h, x[1] - h])) / (4 * h**2)
    hess[1, 0] = hess[0, 1]
    hess[1, 1] = (rosenbrock([x[0], x[1] + h]) - 2 * rosenbrock([x[0], x[1]]) + rosenbrock([x[0], x[1] - h])) / (h**2)

    return hess

def investigate_convergence(h):
    x0 = np.array([-1.2, 0])  # Initial point
    method = 'trust-exact'

    result = minimize(rosenbrock, x0, method=method, jac=lambda x: rosenbrock_gradient(x, h), hess=lambda x: rosenbrock_hessian(x, h), options={'xtol': 1e-8, 'disp': True})
    
    return result.success, result.nit, result.fun

h_values = [0.1, 0.01, 0.001, 0.0001]

for h in h_values:
    success, iterations, minimum = investigate_convergence(h)
    
    if success:
        print(f"For step size h = {h}: The trust-exact method converged to the minimum after {iterations} iterations.")
        print(f"Minimum value: {minimum}\n")
    else:
        print(f"For step size h = {h}: The trust-exact method did not converge to the minimum.\n")
