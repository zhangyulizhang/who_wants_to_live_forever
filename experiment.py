import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ----------------------------------
# 1. 计算存活概率 l(x)（从每日死亡率生成）
# ----------------------------------
def compute_l_from_mu(mu_array, ages, dx=1):
    """
    根据每个年龄段（每日）的死亡率 mu_array[i]，
    计算存活概率 l(x)，假设 l(0)=1.
    """
    l_vals = np.zeros_like(ages, dtype=float)
    l_vals[0] = 1.0
    for i in range(1, len(ages)):
        # 这里 dx=1, 每天死亡概率为 mu_array[i-1]
        l_vals[i] = l_vals[i-1] * (1 - mu_array[i-1])
    return l_vals

# ----------------------------------
# 2. 模拟果蝇数据（统一死亡率和生育率）
# ----------------------------------
def simulate_drosophila_unified():
    """
    模拟果蝇（Drosophila melanogaster）的年龄特异性数据。
    - 生命期：0~50 天，步长1天。
    - 每日连续死亡速率 0.12 转换为离散死亡概率 mu_daily = 1 - exp(-0.12)。
    - 存活率 l(x) = ∏_{i=0}^{x-1} (1 - mu_daily)。
    - 生育率 m(x)：在 5-10 天达到峰值（约50～80 eggs/day），10-20 天呈指数下降，30 天后趋于 0。
    """
    ages = np.arange(0, 51, 1)  # 0到50天
    dx = 1  # 天为单位

    # 连续死亡速率 lambda = 0.12/day，转换为离散死亡概率
    mu_daily = 1 - np.exp(-0.12)  # ~0.1131
    mu_array = np.full_like(ages, mu_daily, dtype=float)  # 每天死亡率相同

    # 根据每日死亡率计算存活概率 l(x)
    l_vals = compute_l_from_mu(mu_array, ages, dx)

    # 模拟生育率 m(x)
    m_vals = np.zeros_like(ages, dtype=float)
    # 5-10天：使用正弦函数模拟高峰（峰值约80）
    idx_peak = (ages >= 5) & (ages <= 10)
    m_vals[idx_peak] = 50 + 30 * np.sin((ages[idx_peak] - 5) * np.pi / 5)
    # 10-20天：指数下降
    idx_decay = (ages > 10) & (ages <= 20)
    m_vals[idx_decay] = 80 * np.exp(-0.2 * (ages[idx_decay] - 10))
    # 20天以后：趋近于0（30天后完全为0）
    m_vals[ages > 30] = 0

    print("daily_mu =", mu_daily)
    print("mu_array[:10] =", mu_array[:10])
    print("l_vals[:20] =", l_vals[:20])

    # 绘图：显示存活率和生育率
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel("Age (days)")
    ax1.set_ylabel("Survival Probability l(x)", color='b')
    ax1.plot(ages, l_vals, color='b', label="l(x)")
    ax1.tick_params(axis='y', labelcolor='b')
    ax2 = ax1.twinx()
    ax2.set_ylabel("Fertility Rate m(x) (eggs/day)", color='r')
    ax2.plot(ages, m_vals, color='r', label="m(x)")
    ax2.tick_params(axis='y', labelcolor='r')
    fig.tight_layout()
    plt.title("Simulated Age-Specific Survival and Fertility for Drosophila")
    plt.show()

    return ages, mu_array, l_vals, m_vals

# ----------------------------------
# 3. Euler-Lotka、世代时间与 Fisher 生殖价值计算
# ----------------------------------
def euler_lotka_error(r, ages, l_vals, m_vals, dx=1):
    lhs = np.sum(np.exp(-r * ages) * l_vals * m_vals * dx)
    return (lhs - 1)**2

def compute_r(ages, l_vals, m_vals, dx=1):
    res = minimize(euler_lotka_error, x0=[0.01], args=(ages, l_vals, m_vals, dx),
                   method="Powell", bounds=[(-5, 5)], tol=1e-6)
    return res.x[0] if res.success else np.nan

def compute_generation_time(ages, r, l_vals, m_vals, dx=1):
    numerator = np.sum(ages * np.exp(-r * ages) * l_vals * m_vals * dx)
    denominator = np.sum(np.exp(-r * ages) * l_vals * m_vals * dx)
    return numerator / denominator

def fisher_reproductive_value(ages, r, l_vals, m_vals, dx=1):
    v_vals = []
    for i in range(len(ages)):
        integral = np.sum(np.exp(-r * ages[i:]) * l_vals[i:] * m_vals[i:] * dx)
        v_vals.append(np.exp(r * ages[i]) * integral)
    return np.array(v_vals)

# ----------------------------------
# 4. Hamilton 敏感性分析：死亡率对 r 的敏感性
# ----------------------------------
def hamilton_sensitivity_mortality(ages, r, l_vals, m_vals, base_mu, dx=1, delta_mu=1e-5):
    """
    对于每个年龄 x_i，在 base_mu[x_i] 上加上小扰动 delta_mu，
    重新计算 l(x) 和 r，从而用有限差分近似 dr/dmu(x)。
    同时计算理论值:
       dr/dmu(x) ≈ -[∫_x^∞ e^{-r*y} l(y) m(y) dy] / T
    """
    dr_dmu_num = np.zeros_like(ages, dtype=float)
    for i in range(len(ages)):
        mu_perturbed = base_mu.copy()
        mu_perturbed[i] += delta_mu
        l_perturbed = compute_l_from_mu(mu_perturbed, ages, dx)
        r_new = compute_r(ages, l_perturbed, m_vals, dx)
        dr_dmu_num[i] = (r_new - r) / delta_mu

    dr_dmu_theo = np.zeros_like(ages, dtype=float)
    T = compute_generation_time(ages, r, l_vals, m_vals, dx)
    for i in range(len(ages)):
        integral = np.sum(np.exp(-r * ages[i:]) * l_vals[i:] * m_vals[i:] * dx)
        dr_dmu_theo[i] = -integral / T

    return dr_dmu_num, dr_dmu_theo

# ----------------------------------
# 5. 主函数
# ----------------------------------
def main():
    # 1. 模拟果蝇数据
    ages, mu_array, l_vals, m_vals = simulate_drosophila_unified()
    dx = 1  # 单位为天

    # 2. 计算 Euler-Lotka 方程得到内禀增长率 r
    r_fly = compute_r(ages, l_vals, m_vals, dx)
    print(f"Drosophila: Estimated intrinsic growth rate r = {r_fly:.6f}")

    # 3. 计算世代时间 T
    T_fly = compute_generation_time(ages, r_fly, l_vals, m_vals, dx)
    print(f"Drosophila: Generation Time T = {T_fly:.6f}")

    # 4. 计算 Fisher 生殖价值 v(x)
    v_fly = fisher_reproductive_value(ages, r_fly, l_vals, m_vals, dx)
    plt.figure(figsize=(8,5))
    plt.plot(ages, v_fly, marker='o', label="Reproductive Value v(x)")
    plt.xlabel("Age (days)")
    plt.ylabel("Fisher Reproductive Value v(x)")
    plt.title("Fisher's Reproductive Value for Drosophila")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 5. Hamilton 敏感性分析（死亡率）
    delta_mu = 1e-5
    dr_dmu_num, dr_dmu_theo = hamilton_sensitivity_mortality(ages, r_fly, l_vals, m_vals, mu_array, dx, delta_mu)
    plt.figure(figsize=(8,5))
    plt.plot(ages, dr_dmu_num, marker='o', label="Numerical dr/dmu(x)")
    plt.plot(ages, dr_dmu_theo, linestyle="--", label="Theoretical dr/dmu(x)")
    plt.xlabel("Age (days)")
    plt.ylabel("dr/dmu(x)")
    plt.title("Hamilton's Mortality Sensitivity for Drosophila")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
