import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def euler_lotka_error(r, ages, l_vals, m_vals, dx=5):
    terms = np.exp(-r * ages) * l_vals * m_vals
    lhs = np.sum(terms * dx)
    return (lhs - 1)**2

def compute_r(ages, l_vals, m_vals, dx=5):
    """给定ages, l_vals, m_vals, 通过Euler-Lotka方程最小化求解r"""
    res = minimize(
        euler_lotka_error,
        x0=[0.01],
        args=(ages, l_vals, m_vals, dx),
        method="Powell",
        bounds=[(-5, 5)],
        tol=1e-6
    )
    if res.success:
        return res.x[0]
    else:
        return np.nan

def main():
    # 假设你已经有以下数据：ages, m_vals, death_rates, l_vals, 以及基准的 r_est
    # 以下是示例，需替换为你实际代码中的处理
    ages = np.array([17.5, 22.5, 27.5, 32.5, 37.5, 42.5])
    m_vals = np.array([0.008, 0.0386, 0.0764, 0.0942, 0.0563, 0.0149])
    death_rates = np.array([0.000252, 0.000355, 0.000448, 0.000654, 0.000997, 0.001515])
    l_vals = np.ones_like(ages)  # 假设先随便给个存活概率
    # ... 这里应替换成你完整的代码来计算 l_vals

    # 计算基准 r
    r_est = compute_r(ages, l_vals, m_vals)
    print(f"Base r = {r_est}")

    # 准备对每个年龄段单独加小扰动
    delta_m = 0.001
    dr_dm_numerical = np.zeros_like(ages)

    for i in range(len(ages)):
        # 复制生育率数组
        m_perturbed = m_vals.copy()
        m_perturbed[i] += delta_m
        # 计算新的 r
        r_new = compute_r(ages, l_vals, m_perturbed)
        # 差分
        dr_dm_numerical[i] = (r_new - r_est) / delta_m

    # 现在 dr_dm_numerical 长度和 ages 相同，可以绘图
    plt.plot(ages, dr_dm_numerical, marker='o', label="Numerical dr/dm(x)")
    plt.xlabel("Age (midpoint)")
    plt.ylabel("dr/dm(x)")
    plt.title("Numerical Sensitivity of r to Fertility Changes")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
