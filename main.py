import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(42)

#Функция для генерации матрицы с заданным числом обусловленности
def generate_matrix_with_cond(m, n, cond):
    mat_m = np.array([[random.gauss(0.0, 1.0) for _ in range(m)] for _ in range(m)])
    mat_n = np.array([[random.gauss(0.0, 1.0) for _ in range(n)] for _ in range(n)])
    U, _ = np.linalg.qr(mat_m)
    V, _ = np.linalg.qr(mat_n)
    r = min(m, n)
    S = np.logspace(0, -np.log10(cond), r)
    Sigma = np.zeros((m, n))
    np.fill_diagonal(Sigma, S)
    return U.dot(Sigma).dot(V.T)


#Функция для отрисовки векторов
def plot_vector(ax, v, color, label, linestyle='-'):
    ax.quiver(0, 0, 0, v[0], v[1], v[2], color=color, label=label, arrow_length_ratio=0.1, linestyle=linestyle,
              linewidth=2)

def main():
    print("ЧАСТЬ 1 и 2: Генерация матрицы и вычисление псевдообратной через SVD")
    m, n = 30, 100
    A = np.array([[random.gauss(0.0, 1.0) for _ in range(n)] for _ in range(m)])
    print(f"Матрица A размером {A.shape}:")

    #A = U * Sigma * V^T
    U, S, Vt = np.linalg.svd(A, full_matrices=True)
    Sigma = np.zeros((m, n))
    np.fill_diagonal(Sigma, S)

    Sigma_plus = np.zeros((n, m))
    tolerance = 1e-10
    S_inv = np.array([1 / s if s > tolerance else 0 for s in S])
    np.fill_diagonal(Sigma_plus, S_inv)

    A_plus = Vt.T.dot(Sigma_plus).dot(U.T)

    A_plus_np = np.linalg.pinv(A)
    print(f"Совпадает ли наша A^+ с np.linalg.pinv? {np.allclose(A_plus, A_plus_np)}\n")

    print("ЧАСТЬ 3: Свойства псевдообратной матрицы Мура-Пенроуза")
    res1 = np.linalg.norm(A.dot(A_plus).dot(A) - A)
    res2 = np.linalg.norm(A_plus.dot(A).dot(A_plus) - A_plus)
    res3 = np.linalg.norm(A.dot(A_plus).T - A.dot(A_plus))
    res4 = np.linalg.norm(A_plus.dot(A).T - A_plus.dot(A))

    max_res = max(res1, res2, res3, res4)
    print(f"1. Невязка A A^+ A - A       : {res1:.3e}")
    print(f"2. Невязка A^+ A A^+ - A^+   : {res2:.3e}")
    print(f"3. Невязка (A A^+)^T - A A^+ : {res3:.3e}")
    print(f"4. Невязка (A^+ A)^T - A^+ A : {res4:.3e}")
    print(f"Максимальная невязка: {max_res:.3e}")
    print(f"Невязка не превышает 10^-12: {max_res <= 1e-12}\n")

    print("ЧАСТЬ 4: Проекторы на образ и ядро")
    P_Im = A.dot(A_plus)  #Проектор на образ A (размер m x m)
    P_Ker_orth = A_plus.dot(A)  #Проектор на ортогональное дополнение к ядру A (размер n x n)
    P_Ker = np.eye(n) - P_Ker_orth  #Проектор на ядро A


    #Свойства ортогональных проекторов: P^2 = P и P^T = P
    print(f"P_Im - проектор? (P^2=P, P^T=P): {np.allclose(P_Im.dot(P_Im), P_Im) and np.allclose(P_Im.T, P_Im)}")
    print(f"P_Ker_orth - проектор?: {np.allclose(P_Ker_orth.dot(P_Ker_orth), P_Ker_orth) and np.allclose(P_Ker_orth.T, P_Ker_orth)}\n")
    print(f"Свойство ядра: A * P_Ker = 0? : {np.allclose(A.dot(P_Ker), 0)}\n")
    #ЧАСТЬ 5: Восстановление сигнала, шум и обусловленность
    print("ЧАСТЬ 5: Восстановление сигнала, шум и обусловленность\n")
    #Зависимость ошибки от уровня шума
    A_exp = generate_matrix_with_cond(m, n, cond=50)
    x_true = np.array([random.gauss(0.0, 1.0) for _ in range(n)])
    x_true = np.linalg.pinv(A_exp).dot(A_exp).dot(x_true)

    norm_x_true = np.linalg.norm(x_true)

    print("Таблица 1: Относительная ошибка восстановления при различном уровне шума (κ = 50).")
    print("-" * 52)
    print(f"{'σ':<10} | {'||x_hat - x_true||':<20} | {'err':<10}")
    print("-" * 52)
    for sigma_tab in [0.01, 0.1, 0.5]:
        noise_tab = sigma_tab * np.array([random.gauss(0.0, 1.0) for _ in range(m)])
        y_noisy_tab = A_exp.dot(x_true) + noise_tab
        x_hat_tab = np.linalg.pinv(A_exp).dot(y_noisy_tab)

        abs_err = np.linalg.norm(x_hat_tab - x_true)
        rel_err = abs_err / norm_x_true
        print(f"{sigma_tab:<10.2f} | {abs_err:<20.3f} | {rel_err:<10.3f}")
    print("-" * 52 + "\n")
    noise_levels = np.logspace(-4, 0, 20)
    errors_noise = []

    fixed_sigma_tab2 = 0.1
    print(f"Таблица 2: Относительная ошибка восстановления при различной обусловленности (σ = {fixed_sigma_tab2}).")
    print("-" * 52)
    print(f"{'κ(A)':<10} | {'||x_hat - x_true||':<20} | {'err':<10}")
    print("-" * 52)
    for cond_tab in [10, 100, 1000]:
        A_cond_tab = generate_matrix_with_cond(m, n, cond_tab)
        x_true_c_tab = np.linalg.pinv(A_cond_tab).dot(A_cond_tab).dot(np.array([random.gauss(0.0, 1.0) for _ in range(n)]))
        norm_x_true_c = np.linalg.norm(x_true_c_tab)

        noise_tab2 = fixed_sigma_tab2 * np.array([random.gauss(0.0, 1.0) for _ in range(m)])
        y_noisy_tab2 = A_cond_tab.dot(x_true_c_tab) + noise_tab2

        x_hat_tab2 = np.linalg.pinv(A_cond_tab).dot(y_noisy_tab2)
        abs_err_c = np.linalg.norm(x_hat_tab2 - x_true_c_tab)
        rel_err_c = abs_err_c / norm_x_true_c
        print(f"{cond_tab:<10} | {abs_err_c:<20.3f} | {rel_err_c:<10.3f}")
    print("-" * 52 + "\n")


    for sigma in noise_levels:
        noise = sigma * np.array([random.gauss(0.0, 1.0) for _ in range(m)])
        y_noisy = A_exp.dot(x_true) + noise
        x_hat = np.linalg.pinv(A_exp).dot(y_noisy)
        errors_noise.append(np.linalg.norm(x_hat - x_true))

    #Зависимость ошибки от числа обусловленности
    cond_numbers = np.logspace(1, 4, 20)
    errors_cond = []
    fixed_noise_sigma = 0.05

    for cond in cond_numbers:
        A_cond = generate_matrix_with_cond(m, n, cond)
        x_true_c = np.linalg.pinv(A_cond).dot(A_cond).dot(np.array([random.gauss(0.0, 1.0) for _ in range(n)]))
        y_noisy = A_cond.dot(x_true_c) + fixed_noise_sigma * np.array([random.gauss(0.0, 1.0) for _ in range(m)])

        x_hat = np.linalg.pinv(A_cond).dot(y_noisy)
        errors_cond.append(np.linalg.norm(x_hat - x_true_c))

    #Отрисовка графиков экспериментов
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.canvas.manager.set_window_title('ЧАСТЬ 5: Восстановление сигнала, шум и обусловленность')
    ax1.plot(noise_levels, errors_noise, marker='o', color='blue')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Уровень шума (стандартное отклонение)')
    ax1.set_ylabel('Ошибка ||x_hat - x_true||')
    ax1.set_title('Зависимость ошибки от уровня шума\n(Фиксированная обусловленность = 10)')

    ax2.plot(cond_numbers, errors_cond, marker='s', color='red')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Число обусловленности (Cond)')
    ax2.set_ylabel('Ошибка ||x_hat - x_true||')
    ax2.set_title(f'Зависимость ошибки от обусловленности\n(Фиксированный шум = {fixed_noise_sigma})')

    plt.tight_layout()

    #ЧАСТЬ 6: 3D Визуализация геометрии псевдообратной матрицы
    #n=3 (пространство сигнала), m=2 (пространство измерений)
    A_vis = np.array([[random.gauss(0.0, 1.0) for _ in range(3)] for _ in range(2)])
    A_vis_plus = np.linalg.pinv(A_vis)

    #Проекторы
    P_Ker_orth_vis = A_vis_plus.dot(A_vis)
    P_Ker_vis = np.eye(3) - P_Ker_orth_vis

    #Генерируем истинный вектор
    x_true_vis = np.array([2.0, 1.5, 3.0])

    #Разлагаем истинный вектор на проекции
    x_ker = P_Ker_vis.dot(x_true_vis)  # Лежит в ядре (A * x_ker = 0)
    x_orth = P_Ker_orth_vis.dot(x_true_vis)  # Лежит в ортог. дополнении (решение мин. нормы)

    #Симулируем измерения с небольшим шумом
    y_vis = A_vis.dot(x_true_vis) + 0.5 * np.array([random.gauss(0.0, 1.0) for _ in range(2)])
    x_hat_vis = A_vis_plus.dot(y_vis)  # Восстановленный вектор

    #Настройка 3D графика
    fig1 = plt.figure(figsize=(10, 8))
    ax = fig1.add_subplot(111, projection='3d')
    fig1.canvas.manager.set_window_title('ЧАСТЬ 6: 3D Визуализация геометрии псевдообратной матрицы')
    plot_vector(ax, x_true_vis, 'black', 'Истинный x_true')
    plot_vector(ax, x_ker, 'blue', 'Проекция на ядро (x_ker)')
    plot_vector(ax, x_orth, 'green', 'Проекция на ортог. дополнение (x_orth, мин. норма)')
    plot_vector(ax, x_hat_vis, 'red', 'Восстановленный с шумом (x_hat)', linestyle='--')

    normal = A_vis[0, :]
    x2, y2 = np.meshgrid(np.linspace(-3, 3, 10), np.linspace(-3, 3, 10))
    z2 = (-normal[0] * x2 - normal[1] * y2) / normal[2]
    ax.plot_surface(x2, y2, z2, alpha=0.2, color='blue')

    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_zlim([-4, 4])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Геометрия решений недоопределенной системы в 3D')
    ax.legend()

    print("ЧАСТЬ 7: Самостоятельное углубление")
    #Здесь A^T - сопряженный оператор, (A * A^T) - матрица Грама
    print("1. Сопряженный оператор и псевдообращение:")
    A_A_T_inv = np.linalg.inv(A.dot(A.T))
    A_pinv_adjoint = A.T.dot(A_A_T_inv)
    print(f"Совпадает ли формула A^T(AA^T)^-1 с np.linalg.pinv(A)? {np.allclose(A_plus_np, A_pinv_adjoint)}")

    #Билинейная форма (A^+ y, x)
    print("\n2. Билинейная форма, задаваемая псевдообратной матрицей:")
    x_true_deep = np.array([random.gauss(0.0, 1.0) for _ in range(n)])
    y_ideal = A.dot(x_true_deep)
    x_hat_deep = A_plus_np.dot(y_ideal)

    bilinear_form_val = np.dot(x_hat_deep, x_true_deep)
    x_hat_norm_sq = np.dot(x_hat_deep, x_hat_deep)

    print(f"Значение формы <A^+ y, x_true>: {bilinear_form_val:.6f}")
    print(f"Квадрат нормы ||x_hat||^2:      {x_hat_norm_sq:.6f}")
    print(f"Совпадают ли они?               {np.isclose(bilinear_form_val, x_hat_norm_sq)}")

    #Обобщение на тензорные уравнения (Псевдообратный тензор)
    print("\n3. Обобщение на тензорные уравнения:")
    I, J, K_dim = 2, 3, 4
    #Генерируем случайный тензор
    T_op = np.array([[[random.gauss(0.0, 1.0) for _ in range(K_dim)] for _ in range(J)] for _ in range(I)])

    T_mat = T_op.reshape(I * J, K_dim)
    T_mat_pinv = np.linalg.pinv(T_mat)
    T_pinv = T_mat_pinv.reshape(K_dim, I, J)

    print(f"Размерность исходного тензора T: {T_op.shape}")
    print(f"Размерность псевдообратного T^+: {T_pinv.shape}")

    #Проверяем 1-е свойство Мура-Пенроуза для тензоров: T * T^+ * T = T
    T_plus_T = np.einsum('kmn, mnl -> kl', T_pinv, T_op)
    #T * (T^+ * T)
    T_reconstructed = np.einsum('ijk, kl -> ijl', T_op, T_plus_T)

    print(f"Свойство T * T^+ * T = T выполнено: {np.allclose(T_op, T_reconstructed)}")

    plt.show()


if __name__ == '__main__':
    main()