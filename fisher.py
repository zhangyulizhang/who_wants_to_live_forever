import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def main():
    df = pd.read_excel(r"/Users/snowie/Documents/data.xlsx")
    print("Data preview:")
    print(df.head())

    age_mid = {
        "15-19": 17.5,
        "20-24": 22.5,
        "25-29": 27.5,
        "30-34": 32.5,
        "35-39": 37.5,
        "40-44": 42.5
    }
    df["x"] = df["Age Groups"].map(age_mid)

    df["m"] = df["ASFR-female"] / 1000.0
    ages = df["x"].values
    m_vals = df["m"].values
    death_rates = df["death_rate"].values
    sort_idx = np.argsort(ages)
    ages = ages[sort_idx]
    m_vals = m_vals[sort_idx]
    death_rates = death_rates[sort_idx]

    l_vals = np.zeros_like(ages, dtype=float)
    l_vals[0] = 1.0
    for i in range(1, len(ages)):
        cum_death = min(5 * death_rates[i - 1], 0.99)
        l_vals[i] = l_vals[i - 1] * (1 - cum_death)

    # 5. 定义 Euler-Lotka 误差函数
    def euler_lotka_error(r, ages, l_vals, m_vals, dx=5):
        terms = np.exp(-r * ages) * l_vals * m_vals
        lhs = np.sum(terms * dx)
        return (lhs - 1) ** 2

    # 6. 求解 r
    res = minimize(
        euler_lotka_error,
        x0=[0.01],
        args=(ages, l_vals, m_vals, 5),
        method="Powell",
        bounds=[(-5, 5)],
        tol=1e-6
    )
    if res.success:
        r_est = res.x[0]
        print(f"\nEstimated intrinsic growth rate r = {r_est:.6f}")
    else:
        print("Optimization for r failed.")
        return

    # 7. 计算 Euler-Lotka 方程的平衡情况
    euler_lotka_sum = np.sum(np.exp(-r_est * ages) * l_vals * m_vals * 5)
    print(f"Euler-Lotka sum: {euler_lotka_sum:.6f}")

    # 8. 计算 Fisher 生殖价值
    def fisher_value(ages, l_vals, m_vals, r, dx=5):
        v_list = []
        for i in range(len(ages)):
            x_i = ages[i]
            future_mask = np.arange(i, len(ages))
            discount = np.exp(-r * ages[future_mask]) * l_vals[future_mask] * m_vals[future_mask]
            integral_part = np.sum(discount * dx)
            v_i = np.exp(r * x_i) * integral_part
            v_list.append(v_i)
        return np.array(v_list)

    v_vals = fisher_value(ages, l_vals, m_vals, r_est)

    # 9. 绘制生殖价值曲线
    plt.figure(figsize=(8, 5))
    plt.plot(ages, v_vals, marker='o', label="Fisher Reproductive Value")
    plt.xlabel("Age (midpoint)")
    plt.ylabel("Reproductive Value v(x)")
    plt.title("Fisher's Reproductive Value by Age Group")
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()


