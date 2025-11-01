import numpy as np
from scipy.integrate import quad
from scipy.optimize import root
import matplotlib.pyplot as plt

class HubbardTBA:
    """
    一维Hubbard模型热力学Bethe Ansatz求解器
    包含正确的Θ函数和θ函数定义
    """
    
    def __init__(self, U=4.0, t=1.0):
        """
        初始化参数
        
        Parameters:
        U: 在位排斥能
        t: 跃迁振幅 (设为1作为能量单位)
        """
        self.U = U
        self.t = t
        self.u = U/(4*t)  # 书中使用的参数
    
    def theta_func(self, x):
        """θ函数定义: θ(x) = 2 arctan(x)"""
        return 2 * np.arctan(x)
    
    def Theta_nm(self, x, n, m):
        """
        Θ函数定义 (方程4.39)
        
        Parameters:
        x: 自变量
        n, m: 弦的长度参数
        """
        if n != m:
            # n ≠ m 的情况
            result = self.theta_func(x / abs(n - m))
            for k in range(1, (n + m - 2) // 2 + 1):
                result += 2 * self.theta_func(x / (abs(n - m) + 2 * k))
            result += self.theta_func(x / (n + m))
            return result
        else:
            # n = m 的情况
            result = 0.0
            for k in range(1, n):
                result += 2 * self.theta_func(x / (2 * k))
            result += self.theta_func(x / (2 * n))
            return result
    
    def a_n(self, x, n):
        """核函数 a_n(x) (方程5.42)"""
        return (1/(2*np.pi)) * (2*n*self.u) / ((n*self.u)**2 + x**2)
    
    def s_function(self, x):
        """s函数 (方程5.59)"""
        return 1/(4*self.u * np.cosh(np.pi*x/(2*self.u)))
    
    def A_nm_operator(self, f, x, n, m, Lambda_grid):
        """
        A_{nm}积分算子 (方程5.43)
        
        Parameters:
        f: 被作用的函数
        x: 自变量
        n, m: 弦的长度参数
        Lambda_grid: Λ的离散网格
        """
        dLambda = Lambda_grid[1] - Lambda_grid[0]
        
        # δ_{nm} f(x) 项
        result = (1.0 if n == m else 0.0) * f(x)
        
        # 积分项: ∫(dy/2π) (d/dx)Θ_{nm}((x-y)/u) f(y)
        integral_term = 0.0
        for y in Lambda_grid:
            # 计算 Θ_{nm} 对 x 的导数
            h = 1e-6  # 数值微分的步长
            theta_plus = self.Theta_nm((x + h - y)/self.u, n, m)
            theta_minus = self.Theta_nm((x - h - y)/self.u, n, m)
            dtheta_dx = (theta_plus - theta_minus) / (2 * h)
            
            integral_term += dtheta_dx * f(y) * dLambda
        
        result += integral_term / (2 * np.pi)
        return result
    
    def solve_tba_equations(self, T, mu, B, n_iter=100, tol=1e-6, max_strings=3):
        """
        求解TBA方程的主函数
        
        Parameters:
        T: 温度
        mu: 化学势
        B: 磁场
        n_iter: 最大迭代次数
        tol: 收敛容差
        max_strings: 考虑的最大弦长度
        """
        # 离散化动量空间和自旋空间
        k_points = 100
        Lambda_points = 100
        
        k_grid = np.linspace(-np.pi, np.pi, k_points)
        Lambda_grid = np.linspace(-5, 5, Lambda_points)  # Λ的截断范围
        
        # 初始化 dressed energies (从自由费米子出发)
        zeta = np.exp((-2*self.t*np.cos(k_grid) - mu - B)/T)
        
        # 初始化各种弦的η和η'
        eta = {}
        eta_prime = {}
        for n in range(1, max_strings + 1):
            eta[n] = np.ones(Lambda_points) * np.exp(2*n*B/T)
            eta_prime[n] = np.ones(Lambda_points) * np.exp((-2*n*mu - 4*n*self.U)/T)
        
        print("开始迭代求解TBA方程...")
        
        for iteration in range(n_iter):
            zeta_old = zeta.copy()
            for n in range(1, max_strings + 1):
                eta_old = eta[n].copy()
                eta_prime_old = eta_prime[n].copy()
            
            # 更新 ζ(k) - 方程(5.54)
            for i, k in enumerate(k_grid):
                sum1 = 0.0
                sum2 = 0.0
                for n in range(1, max_strings + 1):
                    for j, lam in enumerate(Lambda_grid):
                        sum1 += self.a_n(np.sin(k) - lam, n) * np.log(1 + 1/eta_prime[n][j]) * (Lambda_grid[1]-Lambda_grid[0])
                        sum2 += self.a_n(np.sin(k) - lam, n) * np.log(1 + 1/eta[n][j]) * (Lambda_grid[1]-Lambda_grid[0])
                
                zeta[i] = np.exp((-2*self.t*np.cos(k) - mu - 2*self.U - B + sum1 - sum2)/T)
            
            # 更新 η_n(Λ) - 方程(5.55)
            for n in range(1, max_strings + 1):
                for j, lam in enumerate(Lambda_grid):
                    # k积分项
                    k_integral = 0.0
                    for i, k_val in enumerate(k_grid):
                        k_integral += np.cos(k_val) * self.a_n(np.sin(k_val) - lam, n) * np.log(1 + 1/zeta[i]) * (k_grid[1]-k_grid[0])
                    
                    # A_{nm}卷积项
                    A_integral = 0.0
                    for m in range(1, max_strings + 1):
                        # 简化处理：使用数值方法计算A_{nm}算子的作用
                        def f_func(y):
                            idx = np.argmin(np.abs(Lambda_grid - y))
                            return np.log(1 + 1/eta[m][idx])
                        
                        A_integral += self.A_nm_operator(f_func, lam, n, m, Lambda_grid)
                    
                    eta[n][j] = np.exp((2*n*B/T + k_integral + A_integral))
            
            # 更新 η'_n(Λ) - 方程(5.56)
            for n in range(1, max_strings + 1):
                for j, lam in enumerate(Lambda_grid):
                    # 第一项：4Re√(1-(Λ-inu)²) - 2nμ - 4nu
                    real_part = 4 * np.real(np.sqrt(1 - (lam - 1j*n*self.u)**2))
                    driving_term = real_part - 2*n*mu - 4*n*self.U
                    
                    # k积分项
                    k_integral = 0.0
                    for i, k_val in enumerate(k_grid):
                        k_integral += np.cos(k_val) * self.a_n(np.sin(k_val) - lam, n) * np.log(1 + 1/zeta[i]) * (k_grid[1]-k_grid[0])
                    
                    # A_{nm}卷积项
                    A_integral = 0.0
                    for m in range(1, max_strings + 1):
                        def f_func(y):
                            idx = np.argmin(np.abs(Lambda_grid - y))
                            return np.log(1 + 1/eta_prime[m][idx])
                        
                        A_integral += self.A_nm_operator(f_func, lam, n, m, Lambda_grid)
                    
                    eta_prime[n][j] = np.exp((driving_term/T + k_integral + A_integral))
            
            # 检查收敛性
            zeta_diff = np.max(np.abs(zeta - zeta_old))
            max_eta_diff = 0.0
            for n in range(1, max_strings + 1):
                max_eta_diff = max(max_eta_diff, np.max(np.abs(eta[n] - eta_old)))
                max_eta_diff = max(max_eta_diff, np.max(np.abs(eta_prime[n] - eta_prime_old)))
            
            max_diff = max(zeta_diff, max_eta_diff)
            
            if iteration % 10 == 0:
                print(f"迭代 {iteration}, 最大误差: {max_diff:.6f}")
            
            if max_diff < tol:
                print(f"在 {iteration} 次迭代后收敛")
                break
        
        return zeta, eta, eta_prime, k_grid, Lambda_grid
    
    def calculate_free_energy(self, zeta, eta_prime, k_grid, Lambda_grid, T):
        """
        计算自由能密度 (方程5.57)
        """
        dk = k_grid[1] - k_grid[0]
        dLambda = Lambda_grid[1] - Lambda_grid[0]
        
        # 第一项: -T ∫dk/(2π) ln(1 + 1/ζ(k))
        term1 = 0.0
        for i, k in enumerate(k_grid):
            term1 += np.log(1 + 1/zeta[i]) * dk
        term1 = -T * term1 / (2*np.pi)
        
        # 第二项: u
        term2 = self.U/4  # 注意: 书中u = U/4
        
        # 第三项: -T ∑∫dΛ/π ln(1 + 1/η'_n(Λ)) Re[1/√(1 - (Λ - inu)^2)]
        term3 = 0.0
        for n in eta_prime.keys():
            for j, lam in enumerate(Lambda_grid):
                denominator = np.sqrt(1 - (lam - 1j*n*self.u)**2)
                real_part = np.real(1/denominator) if np.abs(denominator) > 1e-10 else 0
                term3 += np.log(1 + 1/eta_prime[n][j]) * real_part * dLambda
        
        term3 = -T * term3 / np.pi
        
        free_energy = term1 + term2 + term3
        return free_energy
    
    def solve_and_plot(self, T_range, mu, B=0.0, max_strings=3):
        """
        在温度范围内求解并绘制自由能
        """
        free_energies = []
        
        for T in T_range:
            print(f"\n求解 T = {T}")
            zeta, eta, eta_prime, k_grid, Lambda_grid = self.solve_tba_equations(
                T, mu, B, max_strings=max_strings)
            f = self.calculate_free_energy(zeta, eta_prime, k_grid, Lambda_grid, T)
            free_energies.append(f)
            print(f"自由能 f = {f:.6f}")
        
        # 绘制结果
        plt.figure(figsize=(10, 6))
        plt.plot(T_range, free_energies, 'o-', linewidth=2, markersize=6)
        plt.xlabel('温度 T', fontsize=12)
        plt.ylabel('自由能密度 f', fontsize=12)
        plt.title(f'一维Hubbard模型自由能 (U={self.U}, μ={mu})', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return free_energies

# 使用示例
if __name__ == "__main__":
    # 初始化Hubbard模型求解器
    hubbard = HubbardTBA(U=4.0, t=1.0)
    
    # 定义温度范围
    T_range = np.linspace(0.5, 3.0, 6)  # 减少点数以加快计算
    
    # 求解特定化学势下的自由能
    mu = -1.0  # 化学势
    B = 0.0    # 零磁场
    
    # 使用较少的弦数量以加快计算
    free_energies = hubbard.solve_and_plot(T_range, mu, B, max_strings=2)
    
    # 输出结果
    print("\n最终结果:")
    for T, f in zip(T_range, free_energies):
        print(f"T = {T:.2f}, f = {f:.6f}")