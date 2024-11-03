import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from KalmanFilter import KalmanFilter

class KalmanApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Kalman Filter Application")
        
        # Основний контейнер для графіка і панелі керування
        main_frame = tk.Frame(root)
        main_frame.pack(expand=True, fill="both")

        # Контейнер для графіка
        fig, ax = plt.subplots(figsize=(5, 4))
        canvas = FigureCanvasTkAgg(fig, master=main_frame)
        canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Панель керування
        control_panel = tk.Frame(main_frame)
        control_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        # Параметри фільтра
        self.frequency = tk.DoubleVar(value=1.0)
        self.amplitude = tk.DoubleVar(value=5.0)
        self.offset = tk.DoubleVar(value=10.0)
        self.total_time = tk.DoubleVar(value=1.0)
        self.Q = tk.DoubleVar(value=1.0)
        self.R = tk.DoubleVar(value=10.0)
        self.P = tk.DoubleVar(value=1.0)
        self.initial_state = tk.DoubleVar(value=0.0)

        # Додавання полів введення для параметрів
        self.add_parameter_controls(control_panel, "Частота:", self.frequency)
        self.add_parameter_controls(control_panel, "Амплітуда:", self.amplitude)
        self.add_parameter_controls(control_panel, "Зсув:", self.offset)
        self.add_parameter_controls(control_panel, "Загальний час:", self.total_time)
        self.add_parameter_controls(control_panel, "Q (Матриця коваріації шуму процесу):", self.Q)
        self.add_parameter_controls(control_panel, "R (Матриця коваріації шуму вимірювання):", self.R)
        self.add_parameter_controls(control_panel, "P (Початкова матриця коваріації):", self.P)
        self.add_parameter_controls(control_panel, "Початкова оцінка стану:", self.initial_state)

        # Кнопки для керування графіком
        tk.Button(control_panel, text="Перестроїти графік", command=lambda: self.redraw_graph(ax, canvas)).pack(pady=5)
        tk.Button(control_panel, text="Видалити поточний графік", command=lambda: self.clear_graph(ax, canvas)).pack(pady=5)

        # Початковий графік
        self.redraw_graph(ax, canvas)

    def add_parameter_controls(self, frame, label_text, variable):
        tk.Label(frame, text=label_text).pack(anchor="w")
        tk.Entry(frame, textvariable=variable).pack(anchor="w", pady=2, fill=tk.X)

    def redraw_graph(self, ax, canvas):
        frequency = self.frequency.get()
        amplitude = self.amplitude.get()
        offset = self.offset.get()
        total_time = self.total_time.get()
        Q = np.array([[self.Q.get()]])
        R = np.array([[self.R.get()]])
        P = np.array([[self.P.get()]])
        initial_state = np.array([[self.initial_state.get()]])

        F = np.array([[1]])
        H = np.array([[1]])
        kf = KalmanFilter(F, H, Q, R, P, initial_state)

        sampling_interval = 0.001
        time_steps = np.arange(0, total_time, sampling_interval)
        true_signal = offset + amplitude * np.sin(2 * np.pi * frequency * time_steps)
        noisy_signal = [val + np.random.normal(0, np.sqrt(R[0][0])) for val in true_signal]

        kalman_estimates = []
        for measurement in noisy_signal:
            kf.predict()
            estimate = kf.update(measurement)
            kalman_estimates.append(estimate[0][0])

        variance_before = np.var(noisy_signal)
        variance_after = np.var(np.array(kalman_estimates) - true_signal)

        ax.clear()
        ax.plot(time_steps, noisy_signal, label='Шумовий сигнал', color='orange', linestyle='-', alpha=0.6)
        ax.plot(time_steps, true_signal, label='Справжній сигнал', linestyle='--', color='blue')
        ax.plot(time_steps, kalman_estimates, label='Оцінка фільтром Калмана', color='green')

        ax.set_xlabel('Час (с)')
        ax.set_ylabel('Значення')
        ax.set_title(f'Дисперсія до: {variance_before:.2f}, Після: {variance_after:.2f}')
        ax.legend()
        ax.grid()
        canvas.draw()

    def clear_graph(self, ax, canvas):
        ax.clear()
        canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = KalmanApp(root)
    root.mainloop()
