import numpy as np

# From "5.2 Thermodynamic Bethe Ansatz (TBA) equations"

class HubbardTBA:

    """
    Self consistent solution of the Thermodynamic Bethe Ansatz(TBA) equations 
    of the one dimensional Hubbard model 
    """
    
    def __init__(self, u=4.0, t=1.0, mu=0.0, B=0.0, k_points=100, Lambda_points=100, Lambda_max=5.0):

        """
        Parameters:
        U: On-site interaction strength
        t: Hopping parameter
        mu: Chemical potential
        B: Magnetic field
        """

        self.u = u
        self.t = t
        self.mu = mu
        self.B = B
        self.k_points = k_points
        self.Lambda_points = Lambda_points
        self.Lambda_max = Lambda_max
    
    def theta_func(self, x):
        return 2 * np.arctan(x)
    
    def Theta_nm(self, x, n, m):

        """
        c.f. Equation (4.39)
        
        Parameters:
        x: input variable
        n, m: string length parameters
        """

        if n != m:
            start = abs(n - m)
            stop = n + m
            result = self.theta_func(x / start)
            for k in range(1, (stop - start) // 2):
                result += 2 * self.theta_func(x / (start + 2 * k))
            result += self.theta_func(x / stop)
            return result
        else:
            result = 0.0
            for k in range(1, n):
                result += 2 * self.theta_func(x / (2 * k))
            result += self.theta_func(x / (2 * n))
            return result
    
    def a_n(self, x, n):

        """
        c.f. Equation (5.42)
        """

        return (1/(2*np.pi)) * (2*n*self.u) / ((n*self.u)**2 + x**2)
    
    def s_function(self, x):
        
        """
        c.f. Equation (5.59)
        """

        return 1/(4*self.u * np.cosh(np.pi*x/(2*self.u)))
    
    def A_nm_operator(self, f, x, n, m, Lambda_grid):

        """
        c.f. Equation (5.43)
        
        Parameters:
        f: the target function
        x: input variable
        n, m: string length parameters
        Lambda_grid: discrete grid for Λ
        """

        dLambda = Lambda_grid[1] - Lambda_grid[0]
        
        # δ_{nm} * f(x)
        result = (1.0 if n == m else 0.0) * f(x)
        
        # ∫(dy/2π) * f(y) * (d/dx)Θ_{nm}((x-y)/u) 
        integral_term = 0.0

        for y in Lambda_grid:
            # Compute the derivative of Θ_{nm} with respect to x
            h = 1e-8 
            theta_p = self.Theta_nm((x + h - y)/self.u, n, m)
            theta_m = self.Theta_nm((x - h - y)/self.u, n, m)
            dtheta_dx = (theta_p - theta_m) / (2 * h)
            
            integral_term += dtheta_dx * f(y) * dLambda
        
        result += integral_term / (2 * np.pi)
        return result
    
    def solve_TBA_equations(self, T, n_iter=100, tol=1e-6, max_strings=3):

        """
        Solve the TBA equations iteratively.
        
        Parameters:
        T: Temperature
        n_iter: maximum number of iterations
        tol: convergence tolerance
        max_strings: maximum number of strings
        """
        
        k_grid = np.linspace(-np.pi, np.pi, self.k_points)
        dk = k_grid[1]-k_grid[0]
        Lambda_grid = np.linspace(-self.Lambda_max, self.Lambda_max, self.Lambda_points)
        dLambda = Lambda_grid[1]-Lambda_grid[0]
        
        # Initialize the dressed energies from the non-interacting case
        zeta = np.exp((-2 * self.t * np.cos(k_grid) - self.mu - self.B)/T)
        
        # Initialize η[n][Λ] and η'[n][Λ]
        eta = {}
        eta_prime = {}
        for n in range(1, max_strings + 1):
            eta[n] = np.ones(self.Lambda_points) * np.exp(2 * n * self.B/T)
            eta_prime[n] = np.ones(self.Lambda_points) * np.exp((-2 * n * self.mu - 4 * n * self.u)/T)
        
        print("Starting TBA iterations...")
        
        for iteration in range(n_iter):

            zeta_old = zeta.copy()
            eta_old = eta.copy()
            eta_prime_old = eta_prime.copy()
            
            # Update ζ(k) - c.f. Equation (5.54)
            for i, k in enumerate(k_grid):
                sum1 = 0.0
                sum2 = 0.0
                for n in range(1, max_strings + 1):
                    for j, lam in enumerate(Lambda_grid):
                        sum1 += self.a_n(np.sin(k) - lam, n) * np.log(1 + 1/eta_prime[n][j]) * dLambda
                        sum2 += self.a_n(np.sin(k) - lam, n) * np.log(1 + 1/eta[n][j]) * dLambda
                
                zeta[i] = np.exp((-2 * self.t * np.cos(k) - self.mu - 2 * self.u - self.B)/T + sum1 - sum2)
            
            # Update η_n(Λ) - c.f. Equation (5.55)
            for n in range(1, max_strings + 1):
                for j, lam in enumerate(Lambda_grid):
                    k_integral = 0.0
                    for i, k_val in enumerate(k_grid):
                        k_integral += np.cos(k_val) * self.a_n(np.sin(k_val) - lam, n) * np.log(1 + 1/zeta[i]) * dk
                    
                    A_integral = 0.0
                    for m in range(1, max_strings + 1):
                        def f_func(y):
                            idx = np.argmin(np.abs(Lambda_grid - y))
                            return np.log(1 + 1/eta[m][idx])
                        
                        A_integral += self.A_nm_operator(f_func, lam, n, m, Lambda_grid)
                    
                    eta[n][j] = np.exp((2 * n * self.B / T - k_integral + A_integral)) - 1
            
            # Update η'_n(Λ) - c.f. Equation (5.56)
            for n in range(1, max_strings + 1):
                for j, lam in enumerate(Lambda_grid):
                    # 4Re√(1-(Λ-inu)²) - 2nμ - 4nu
                    real_part = 4 * np.real(np.sqrt(1 - (lam - 1j * n * self.u)**2))
                    driving_term = real_part - 2 * n * self.mu - 4 * n * self.u
                    
                    k_integral = 0.0
                    for i, k_val in enumerate(k_grid):
                        k_integral += np.cos(k_val) * self.a_n(np.sin(k_val) - lam, n) * np.log(1 + 1/zeta[i]) * dk
                    
                    # A_{nm} integral
                    A_integral = 0.0
                    for m in range(1, max_strings + 1):
                        def f_func(y):
                            idx = np.argmin(np.abs(Lambda_grid - y))
                            return np.log(1 + 1/eta_prime[m][idx])
                        
                        A_integral += self.A_nm_operator(f_func, lam, n, m, Lambda_grid)
                    
                    eta_prime[n][j] = np.exp((driving_term/T - k_integral + A_integral)) - 1
            
            # Check for convergence
            zeta_diff = np.max(np.abs(zeta - zeta_old))
            max_eta_diff = np.zeros(max_strings)
            max_eta_prime_diff = np.zeros(max_strings)
            for n in range(1, max_strings + 1):
                max_eta_diff[n-1] = np.max(np.abs(eta[n] - eta_old[n]))
                max_eta_prime_diff[n-1] = np.max(np.abs(eta_prime[n] - eta_prime_old[n]))
            
            max_diff = max(zeta_diff, np.max(max_eta_diff), np.max(max_eta_prime_diff))
            
            if iteration % 10 == 0:
                print(f"Iteration No. {iteration}, Maximal difference: {max_diff:.6f}")
            
            if max_diff < tol:
                print(f"Convergence is reached after {iteration} iterations !")
                break
        
        return zeta, eta, eta_prime, k_grid, Lambda_grid
    
    def calculate_free_energy(self, zeta, eta_prime, k_grid, Lambda_grid, T):

        """
        Gibbs free energy per site, c.f. Equation (5.57)
        """

        dk = k_grid[1] - k_grid[0]
        dLambda = Lambda_grid[1] - Lambda_grid[0]
        
        # The first term : -T ∫dk/(2π) ln(1 + 1/ζ(k))
        term1 = 0.0
        for i, _ in enumerate(k_grid):
            term1 += np.log(1 + 1/zeta[i]) * dk
        term1 = -T * term1 / (2*np.pi)
        
        # The second term : u
        term2 = self.u
        
        # The third term : -T ∑∫dΛ/π ln(1 + 1/η'_n(Λ)) Re[1/√(1 - (Λ - inu)^2)]
        term3 = 0.0
        for n in eta_prime.keys():
            for j, lam in enumerate(Lambda_grid):
                denominator = np.sqrt(1 - (lam - 1j * n * self.u)**2)
                real_part = np.real(1/denominator) if np.abs(denominator) > 1e-10 else 0
                term3 += np.log(1 + 1/eta_prime[n][j]) * real_part * dLambda
        
        term3 = -T * term3 / np.pi
        
        free_energy = term1 + term2 + term3

        return free_energy

# 使用示例
if __name__ == "__main__":

    # Initialize the HubbardTBA solver
    HubbardTBA_solver = HubbardTBA(u=4.0, t=1.0, mu=-1.0, B=0.0, k_points=100, Lambda_points=100, Lambda_max=5.0)
    
    # Define temperature range
    # T_range = np.linspace(0.5, 3.0, 6) 
    T = 10.0
    
    zeta, _, eta_prime, k_grid, Lambda_grid = HubbardTBA_solver.solve_TBA_equations(T, n_iter=200, tol=1e-6, max_strings=2)
    free_energy = HubbardTBA_solver.calculate_free_energy(zeta, eta_prime, k_grid, Lambda_grid, T)
    
    print("Temperature: ", T, "   free_energy : ", free_energy)