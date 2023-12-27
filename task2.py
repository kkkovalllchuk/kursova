import numpy as np
from scipy.optimize import minimize

# Definition of the Rosenbrock function
def rosenbrock(x):
    return 100 * (x[0]**2 - x[1])**2 + (x[0] - 1)**2

# Definition of the gradient of the Rosenbrock function using central differencing scheme
def rosenbrock_gradient_central(x, epsilon=1e-6):
    gradient = np.zeros_like(x)
    for i in range(len(x)):
        delta = np.zeros_like(x)
        delta[i] = epsilon
        gradient[i] = (rosenbrock(x + delta) - rosenbrock(x - delta)) / (2 * epsilon)
    return gradient

# Definition of the Hessian of the Rosenbrock function using central differencing scheme
def rosenbrock_hessian_central(x, epsilon=1e-6):
    hessian = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        delta_i = np.zeros_like(x)
        delta_i[i] = epsilon
        for j in range(len(x)):
            delta_j = np.zeros_like(x)
            delta_j[j] = epsilon
            hessian[i, j] = (rosenbrock(x + delta_i + delta_j) - rosenbrock(x + delta_i - delta_j) -
                             rosenbrock(x - delta_i + delta_j) + rosenbrock(x - delta_i - delta_j)) / (4 * epsilon**2)
    return hessian

# Initial point
x0 = np.array([-1.2, 0])

# Call minimize function with the trust-exact method and central differencing scheme for derivatives
result_central = minimize(rosenbrock, x0, method='trust-exact', jac=rosenbrock_gradient_central, hess=rosenbrock_hessian_central)

# Definition of the gradient of the Rosenbrock function using forward differencing scheme
def rosenbrock_gradient_forward(x, epsilon=1e-6):
    gradient = np.zeros_like(x)
    for i in range(len(x)):
        delta = np.zeros_like(x)
        delta[i] = epsilon
        gradient[i] = (rosenbrock(x + delta) - rosenbrock(x)) / epsilon
    return gradient

# Definition of the Hessian of the Rosenbrock function using forward differencing scheme
def rosenbrock_hessian_forward(x, epsilon=1e-6):
    hessian = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        delta_i = np.zeros_like(x)
        delta_i[i] = epsilon
        for j in range(len(x)):
            delta_j = np.zeros_like(x)
            delta_j[j] = epsilon
            hessian[i, j] = (rosenbrock(x + delta_i + delta_j) - rosenbrock(x + delta_i)) / epsilon
    return hessian

# Call minimize function with the trust-exact method and forward differencing scheme for derivatives
result_forward = minimize(rosenbrock, x0, method='trust-exact', jac=rosenbrock_gradient_forward, hess=rosenbrock_hessian_forward)

# Definition of the analytical gradient of the Rosenbrock function
def rosenbrock_gradient_analytical(x):
    return np.array([400 * x[0] * (x[0]**2 - x[1]) + 2 * (x[0] - 1), -200 * (x[0]**2 - x[1])])

# Definition of the analytical Hessian of the Rosenbrock function
def rosenbrock_hessian_analytical(x):
    return np.array([[1200 * x[0]**2 - 400 * x[1] + 2, -400 * x[0]], [-400 * x[0], 200]])

# Call minimize function with the trust-exact method and analytical derivatives
result_analytical = minimize(rosenbrock, x0, method='trust-exact', jac=rosenbrock_gradient_analytical, hess=rosenbrock_hessian_analytical)

# Display results
print("Result with central differencing scheme:")
print(result_central)
print("Minimum value of the function:", rosenbrock(result_central.x))
print()

print("Result with forward differencing scheme:")
print(result_forward)
print("Minimum value of the function:", rosenbrock(result_forward.x))
print()

print("Result with analytical derivatives:")
print(result_analytical)
print("Minimum value of the function:", rosenbrock(result_analytical.x))
