import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def euler_lotka_error(r, ages, l_vals, m_vals, dx=5):
    """
    计算 Euler-Lotka 方程的误差，用于求解内禀增长率 r
    """
    terms = np.exp(-r * ages) * l_vals * m_vals
    lhs = np.sum(terms * dx)
    return (lhs - 1) ** 2


def compute_r(ages, l_vals, m_vals, dx=5):
    """
    给定年龄、存活概率和生育率，使用 Euler-Lotka 方程求解内禀增长率 r
    """
    res = minimize(euler_lotka_error, x0=[0.01], args=(ages, l_vals, m_vals, dx),
                   method="Powell", bounds=[(-5, 5)], tol=1e-6)
    if res.success:
        return res.x[0]
    else:
        return np.nan


def compute_l(death_rates, ages, dx=5):
    """
    根据各年龄段死亡率和区间宽度 dx，计算存活概率 l(x)
    假设从第一个年龄段开始的存活概率为 1
    """
    l_vals = np.zeros_like(ages, dtype=float)
    l_vals[0] = 1.0
    for i in range(1, len(ages)):
        cum_death = min(dx * death_rates[i - 1], 0.99)
        l_vals[i] = l_vals[i - 1] * (1 - cum_death)
    return l_vals


def main():
    # 1. 读取 Excel 数据
    df = pd.read_excel(r"/Users/snowie/Documents/data.xlsx")
    print("Data preview:")
    print(df.head())

    # 2. 将年龄组转换为中点值
    age_mid = {
        "15-19": 17.5,
        "20-24": 22.5,
        "25-29": 27.5,
        "30-34": 32.5,
        "35-39": 37.5,
        "40-44": 42.5
    }
    df["x"] = df["Age Groups"].map(age_mid)

    # 3. 将 ASFR-female 转换为每位女性的生育率 m(x)
    df["m"] = df["ASFR-female"] / 1000.0

    # 4. 提取数据，并按年龄排序
    ages = df["x"].values
    m_vals = df["m"].values
    death_rates = df["death_rate"].values
    sort_idx = np.argsort(ages)
    ages = ages[sort_idx]
    m_vals = m_vals[sort_idx]
    death_rates = death_rates[sort_idx]

    dx = 5
    l_vals = compute_l(death_rates, ages, dx)

    # 5. 计算基准的内禀增长率 r
    r_est = compute_r(ages, l_vals, m_vals, dx)
    print(f"\nEstimated intrinsic growth rate r = {r_est:.6f}")

    # 6. 计算世代时间 T
    T = np.sum(ages * np.exp(-r_est * ages) * l_vals * m_vals * dx) / np.sum(
        np.exp(-r_est * ages) * l_vals * m_vals * dx)
    print(f"Generation Time (T) = {T:.6f}")

    # 7. 数值验证 Hamilton 公式关于死亡率敏感性（3.1.2）
    # 对每个年龄段，加一个小扰动 delta_mu，然后计算新的 r 值，再计算差分近似导数
    delta_mu = 1e-5
    dr_dmu_numerical = np.zeros_like(ages, dtype=float)

    for i in range(len(ages)):
        # 复制原始死亡率数组并对第 i 个年龄段加扰动
        death_rates_perturbed = death_rates.copy()
        death_rates_perturbed[i] += delta_mu
        # 根据扰动后的死亡率重新计算存活概率
        l_vals_perturbed = compute_l(death_rates_perturbed, ages, dx)
        # 计算新的内禀增长率
        r_new = compute_r(ages, l_vals_perturbed, m_vals, dx)
        dr_dmu_numerical[i] = (r_new - r_est) / delta_mu

    # 计算理论上的 dr/dmu，公式为:
    # dr/dmu(x) \approx -\frac{\int_{x}^{\infty} e^{-ry} l(y) m(y) dy}{T}
    dr_dmu_theoretical = np.zeros_like(ages, dtype=float)
    for i in range(len(ages)):
        integral_value = np.sum(np.exp(-r_est * ages[i:]) * l_vals[i:] * m_vals[i:] * dx)
        dr_dmu_theoretical[i] = -integral_value / T

    # 8. 绘图比较
    plt.figure(figsize=(8, 5))
    plt.plot(ages, dr_dmu_numerical, marker='o', label="Numerical dr/dmu(x)")
    plt.plot(ages, dr_dmu_theoretical, linestyle="dashed", label="Theoretical dr/dmu(x)")
    plt.xlabel("Age (midpoint)")
    plt.ylabel("dr/dmu(x)")
    plt.title("Validation of Hamilton's Mortality Sensitivity Equation")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
