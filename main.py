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
    S = np.linspace(1, 1 / cond, min(m, n))
    Sigma = np.zeros((m, n))
    np.fill_diagonal(Sigma, S)
    return U.dot(Sigma).dot(V.T)


#Функция для отрисовки векторов
def plot_vector(ax, v, color, label, linestyle='-'):
    ax.quiver(0, 0, 0, v[0], v[1], v[2], color=color, label=label, arrow_length_ratio=0.1, linestyle=linestyle,
              linewidth=2)

def main():
    print("ЧАСТЬ 1 и 2: Генерация матрицы и вычисление псевдообратной через SVD")
    m, n = 3, 5  #Недоопределенная система (уравнений меньше, чем неизвестных)
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
    prop1 = np.allclose(A.dot(A_plus).dot(A), A)
    prop2 = np.allclose(A_plus.dot(A).dot(A_plus), A_plus)
    prop3 = np.allclose(A.dot(A_plus).T, A.dot(A_plus))
    prop4 = np.allclose(A_plus.dot(A).T, A_plus.dot(A))

    print(f"1. A A^+ A = A           : {prop1}")
    print(f"2. A^+ A A^+ = A^+       : {prop2}")
    print(f"3. (A A^+)^T = A A^+     : {prop3}")
    print(f"4. (A^+ A)^T = A^+ A     : {prop4}\n")

    print("ЧАСТЬ 4: Проекторы на образ и ядро")
    P_Im = A.dot(A_plus)  # Проектор на образ A (размер m x m)
    P_Ker_orth = A_plus.dot(A)  # Проектор на ортогональное дополнение к ядру A (размер n x n)
    P_Ker = np.eye(n) - P_Ker_orth  # Проектор на ядро A


    #Свойства ортогональных проекторов: P^2 = P и P^T = P
    print(f"P_Im - проектор? (P^2=P, P^T=P): {np.allclose(P_Im.dot(P_Im), P_Im) and np.allclose(P_Im.T, P_Im)}")
    print(f"P_Ker_orth - проектор?: {np.allclose(P_Ker_orth.dot(P_Ker_orth), P_Ker_orth) and np.allclose(P_Ker_orth.T, P_Ker_orth)}\n")
    print(f"Свойство ядра: A * P_Ker = 0? : {np.allclose(A.dot(P_Ker), 0)}\n")
    #ЧАСТЬ 5: Восстановление сигнала, шум и обусловленность
    #Зависимость ошибки от уровня шума
    m_exp, n_exp = 20, 50
    A_exp = generate_matrix_with_cond(m_exp, n_exp, cond=10)
    x_true = np.array([random.gauss(0.0, 1.0) for _ in range(n_exp)])
    x_true = np.linalg.pinv(A_exp).dot(A_exp).dot(x_true)

    noise_levels = np.logspace(-4, 0, 20)
    errors_noise = []

    for sigma in noise_levels:
        noise = sigma * np.array([random.gauss(0.0, 1.0) for _ in range(m_exp)])
        y_noisy = A_exp.dot(x_true) + noise
        x_hat = np.linalg.pinv(A_exp).dot(y_noisy)
        errors_noise.append(np.linalg.norm(x_hat - x_true))

    #Зависимость ошибки от числа обусловленности
    cond_numbers = np.logspace(1, 4, 20)
    errors_cond = []
    fixed_noise_sigma = 0.05

    for cond in cond_numbers:
        A_cond = generate_matrix_with_cond(m_exp, n_exp, cond)
        x_true_c = np.linalg.pinv(A_cond).dot(A_cond).dot(np.array([random.gauss(0.0, 1.0) for _ in range(n_exp)]))
        y_noisy = A_cond.dot(x_true_c) + fixed_noise_sigma * np.array([random.gauss(0.0, 1.0) for _ in range(m_exp)])

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

    plt.show()


if __name__ == '__main__':
    main()