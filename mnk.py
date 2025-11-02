#pip install scipy
#pip install pandas matplotlib
#pip install sympy

import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from functools import partial
from scipy.optimize import curve_fit



class MNKApp:

    def __init__(self, root):
        self.root = root
        self.root.title("График по точкам")

        self.label = tk.Label(root, text="Выберите действие:", font=("Arial", 14))
        self.label.pack(pady=10)

        self.btn_graph = tk.Button(root, text="График по точкам и МНК", width=40, command=self.open_plot_only_window)
        self.btn_graph.pack(pady=5)

        self.btn_errors = tk.Button(root, text="Обработка и погрешности", width=40, command=self.open_error_menu)
        self.btn_errors.pack(pady=5)

    def open_plot_only_window(self):
        window = tk.Toplevel(self.root)
        window.title("График по точкам и МНК")

        tk.Label(window, text="Для использования 10^x, используйте e-формат или ^ / **: 10^(-3) = e-3").pack(pady=2)
        tk.Label(window, text="Введите заголовок для зависимости:").pack()
        label_entry = tk.Entry(window)
        label_entry.pack(pady=3)

        canvas = tk.Canvas(window)
        frame = tk.Frame(canvas)
        vsb = tk.Scrollbar(window, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)

        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.create_window((0, 0), window=frame, anchor='nw')
        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # Заголовки
        tk.Label(frame, text="Введите значения y, x, σy и σx", font=("Arial", 12)).grid(row=0, column=1, columnspan=5,
                                                                                        pady=5)

        self.axis_labels = []
        headers = ["y", "x", "σy", "σx"]
        #for j, header in enumerate(headers):
        for j, header in enumerate(headers):
            tk.Label(frame, text=header).grid(row=1, column=j + 1)

        tk.Label(frame, text="Название осей").grid(row=2, column=0)
        self.axis_labels = []
        for j in range(2):
            e = tk.Entry(frame, width=10)
            e.grid(row=2, column=j + 1)
            self.axis_labels.append(e)

        self.entries = []
        self.row_labels = []
        self.row_count = 10
        self.max_rows = 100
        self.no_sy = False
        self.no_sx = False

        def add_row(i):
            row_entries = []
            lbl = tk.Label(frame, text=str(i + 1))
            lbl.grid(row=i + 3, column=0)
            self.row_labels.append(lbl)
            for j in range(4):
                e = tk.Entry(frame, width=10)
                e.grid(row=i + 3, column=j + 1, padx=2, pady=1)
                e.bind("<Control-v>", lambda event, r=i, c=j: self.paste_from_clipboard(r, c))
                e.bind("<Command-v>", lambda event, r=i, c=j: self.paste_from_clipboard(r, c))  # для Mac
                e.bind("<Return>", lambda ev, row=i, col=j: handle_navigation(ev, row, col))
                e.bind("<Down>", lambda ev, row=i, col=j: handle_navigation(ev, row, col))
                e.bind("<Up>", lambda ev, row=i, col=j: handle_navigation(ev, row, col))
                e.bind("<Left>", lambda ev, row=i, col=j: handle_navigation(ev, row, col))
                e.bind("<Right>", lambda ev, row=i, col=j: handle_navigation(ev, row, col))
                row_entries.append(e)
            self.entries.append(row_entries)

        def handle_navigation(event, row, col):
            widget = event.widget
            if event.keysym == "Return":
                if row == len(self.entries) - 1 and len(self.entries) < self.max_rows:
                    add_row(len(self.entries))
                next_row = min(row + 1, len(self.entries) - 1)
                self.entries[next_row][col].focus_set()
            elif event.keysym == "Down" and row + 1 < len(self.entries):
                self.entries[row + 1][col].focus_set()
            elif event.keysym == "Up" and row - 1 >= 0:
                self.entries[row - 1][col].focus_set()
            elif event.keysym == "Right":
                if widget.index(tk.INSERT) == len(widget.get()):
                    if col + 1 < 4:
                        self.entries[row][col + 1].focus_set()
            elif event.keysym == "Left":
                if widget.index(tk.INSERT) == 0:
                    if col - 1 >= 0:
                        self.entries[row][col - 1].focus_set()

        for i in range(self.row_count):
            add_row(i)

        def toggle_sigma(which):
            if which == 'sy':
                self.no_sy = not self.no_sy
                for row in self.entries:
                    row[2].delete(0, tk.END)
                    row[2].insert(0, "0" if self.no_sy else "")
            elif which == 'sx':
                self.no_sx = not self.no_sx
                for row in self.entries:
                    row[3].delete(0, tk.END)
                    row[3].insert(0, "0" if self.no_sx else "")

        def process():
            x, y, sy, sx = [], [], [], []
            try:
                for row in self.entries:
                    if all(cell.get().strip() != "" for cell in row[:2]):
                        y.append(float(eval(row[0].get().replace("^", "**"))))
                        x.append(float(eval(row[1].get().replace("^", "**"))))
                        sy_val = float(eval(row[2].get().replace("^", "**"))) if not self.no_sy else 0.0
                        sx_val = float(eval(row[3].get().replace("^", "**"))) if not self.no_sx else 0.0
                        sy.append(sy_val)
                        sx.append(sx_val)

                if len(x) < 2:
                    raise ValueError("Мало данных для построения.")
                xlabel = self.axis_labels[1].get() or "x"
                ylabel = self.axis_labels[0].get() or "y"
                label = label_entry.get().strip()

                self.ask_plot_mode(x, y, sy, sx, xlabel, ylabel, label)

            except Exception as e:
                messagebox.showerror("Ошибка", str(e))

        # Кнопки справа
        btn_frame = tk.Frame(frame)
        btn_frame.grid(row=0, column=6, rowspan=4, padx=10, sticky="n")
        tk.Checkbutton(btn_frame, text="Нет σy", command=lambda: toggle_sigma('sy')).pack(pady=2)
        tk.Checkbutton(btn_frame, text="Нет σx", command=lambda: toggle_sigma('sx')).pack(pady=2)

        self.calc_area_var = tk.IntVar()
        tk.Checkbutton(btn_frame, text="Рассчитать площадь", variable=self.calc_area_var).pack(pady=4)

        # Кнопка далее
        tk.Button(frame, text="Далее", command=process).grid(row=4, column=6, pady=4)

    def ask_plot_mode(self, x, y, sigma_y, sigma_x, xlabel, ylabel, label):
        mode_win = tk.Toplevel(self.root)
        mode_win.title("Выбор режима графика")

        def plot_points():
            plt.figure(figsize=(10, 6))
            plt.errorbar(x, y, yerr=sigma_y, xerr=sigma_x, fmt='o', capsize=5, color='black', markersize=5)
            if label:
                plt.title(label)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            mode_win.destroy()

        def plot_connected():
            def calculate_area(xs, ys, method="trapezoid"):
                if method == "trapezoid":
                    return np.trapz(ys, xs)
                elif method == "rectangles":
                    dx = np.diff(xs)
                    return np.sum(ys[:-1] * dx)

            method_win = tk.Toplevel()
            method_win.title("Метод вычисления площади")
            tk.Label(method_win, text="Выберите метод:").pack(pady=5)

            for method in ["trapezoid", "rectangles"]:
                def use_method(m=method):
                    xs = np.array(x)
                    ys = np.array(y)
                    area = calculate_area(xs, ys, m)

                    plt.figure(figsize=(10, 6))
                    plt.errorbar(xs, ys, yerr=sigma_y, xerr=sigma_x, fmt='o-', color='black', capsize=5, markersize=5, linewidth=1)
                    plt.xlabel(xlabel)
                    plt.ylabel(ylabel)
                    plt.grid(True)
                    if hasattr(self, "calc_area_var") and self.calc_area_var.get():
                        area = calculate_area(np.array(x), np.array(y), m)
                        plt.figtext(0.15, 0.01, f"Площадь под графиком ({m}): {area:.6g}", fontsize=10)
                    plt.tight_layout()
                    plt.show()

                    method_win.destroy()
                    mode_win.destroy()
                tk.Button(method_win, text=method.capitalize(), command=use_method).pack(pady=3)

        def plot_mnk():
            select_win = tk.Toplevel(self.root)
            select_win.title("Выбор зависимости")

            tk.Label(select_win, text="Выберите зависимость:").pack(pady=5)

            def mode_ax():
                self.plot_mnk(x, y, sigma_y, sigma_x, xlabel, ylabel, label, mode="ax")
                select_win.destroy()
                mode_win.destroy()

            def mode_axb():
                self.plot_mnk(x, y, sigma_y, sigma_x, xlabel, ylabel, label, mode="axb")
                select_win.destroy()
                mode_win.destroy()

            def mode_poly2():
                self.plot_polynomial(x, y, sigma_y, sigma_x, xlabel, ylabel, label, degree=2)
                select_win.destroy()

            def mode_poly3():
                self.plot_polynomial(x, y, sigma_y, sigma_x, xlabel, ylabel, label, degree=3)
                select_win.destroy()

            def mode_sqrt():
                self.plot_custom_fit(x, y, sigma_y, sigma_x, xlabel, ylabel, label, model=lambda x, a, b: a * np.sqrt(x) + b,
                                     formula="a·√x + b")
                select_win.destroy()

            def mode_sin():
                self.plot_custom_fit(x, y, sigma_y, sigma_x, xlabel, ylabel, label, model=lambda x, a, b: a * np.sin(x) + b,
                                     formula="a·sin(x) + b")
                select_win.destroy()



            tk.Button(select_win, text="y = a·x", command=mode_ax).pack(pady=5)
            tk.Button(select_win, text="y = a·x + b", command=mode_axb).pack(pady=5)
            tk.Button(select_win, text="y = a·x² + b·x + c", command=mode_poly2).pack(pady=5)
            tk.Button(select_win, text="y = a·x³ + b·x² + c·x + d", command=mode_poly3).pack(pady=5)
            tk.Button(select_win, text="y = a·√x + b", command=mode_sqrt).pack(pady=5)
            tk.Button(select_win, text="y = a·sin(x) + b", command=mode_sin).pack(pady=5)

        tk.Label(mode_win, text="Как построить график?").pack(pady=5)
        tk.Button(mode_win, text="По точкам", command=plot_points).pack(pady=2)
        tk.Button(mode_win, text="Соединить точки", command=plot_connected).pack(pady=2)
        tk.Button(mode_win, text="Метод наименьших квадратов", command=plot_mnk).pack(pady=5)

    def plot_mnk(self, x, y, sigma_y, sigma_x, xlabel, ylabel, label, mode="axb"):
        x, y, sigma_y, sigma_x = map(np.array, (x, y, sigma_y, sigma_x))
        w = 1 / (sigma_y ** 2 + 1e-12)

        if mode == "axb":
            S = np.sum(w)
            Sx = np.sum(w * x)
            Sy = np.sum(w * y)
            Sxx = np.sum(w * x ** 2)
            Sxy = np.sum(w * x * y)
            Delta = S * Sxx - Sx ** 2

            a = (S * Sxy - Sx * Sy) / Delta
            b = (Sxx * Sy - Sx * Sxy) / Delta

            N = len(x)
            sigma_a = np.sqrt(np.sum((y - (a * x + b))**2) / ((N - 2) * np.sum((x - np.mean(x))**2)))
            sigma_b = sigma_a * np.sqrt(np.sum(x**2) / N)
            equation_text = f"a = {a:.4g} ± {sigma_a:.4g}, b = {b:.4g} ± {sigma_b:.4g}"
            y_line = a * x + b

        else:
            S = np.sum(x ** 2)
            a = np.sum(x * y) / S
            sigma_a = np.sqrt(np.sum((y - a * x)**2) / ((len(x) - 1) * np.sum((x - np.mean(x))**2)))
            b = 0
            sigma_b = 0
            equation_text = f"a = {a:.4g} ± {sigma_a:.4g}"
            y_line = a * x

        x_line = np.linspace(min(x), max(x), 500)
        y_line_full = a * x_line + b

        plt.figure(figsize=(10, 6))
        plt.errorbar(x, y, yerr=sigma_y, xerr=sigma_x, fmt='o', color='black', capsize=5, markersize=5)
        plt.plot(x_line, y_line_full, 'r-', linewidth=1)
        if hasattr(self, "calc_area_var") and self.calc_area_var.get():
            area = np.trapz(y_line_full, x_line)
            plt.figtext(0.1, 0.05, f"Площадь под графиком: {area:.6g}", fontsize=10)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        if label:
            plt.title(label)
        plt.figtext(0.1, 0.01, equation_text, fontsize=10, ha='left')
        plt.tight_layout()
        plt.show()


    def plot_polynomial(self, x, y, sigma_y, sigma_x, xlabel, ylabel, label, degree=2):
        x, y = map(np.array, (x, y))
        if len(x) <= degree:
            messagebox.showerror("Ошибка", f"Для полинома степени {degree} требуется минимум {degree + 1} точек.")
            return

        coeffs, cov = np.polyfit(x, y, degree, w=1 / (np.array(sigma_y) + 1e-10), cov=True)
        poly = np.poly1d(coeffs)
        sigma = np.sqrt(np.diag(cov))

        x_line = np.linspace(min(x), max(x), 500)
        y_line = poly(x_line)

        formula = " + ".join([f"{coeffs[i]:.4g}·x^{degree - i}" if degree - i > 1 else
                            f"{coeffs[i]:.4g}·x" if degree - i == 1 else
                            f"{coeffs[i]:.4g}" for i in range(len(coeffs))])
        formula_text = f"y = {formula}"
        coef_text = "\n".join([f"{chr(97 + i)} = {coeffs[i]:.4g} ± {sigma[i]:.2g}" for i in range(len(coeffs))])

        self.plot_with_errors(x, y, sigma_y, sigma_x, x_line, y_line, xlabel, ylabel, label, formula_text, coef_text)


    def plot_custom_fit(self, x, y, sigma_y, sigma_x, xlabel, ylabel, label, model, formula):
        x, y = map(np.array, (x, y))
        try:
            popt, pcov = curve_fit(model, x, y, sigma=np.array(sigma_y) + 1e-10, absolute_sigma=True)
            sigma = np.sqrt(np.diag(pcov))

            x_line = np.linspace(min(x), max(x), 500)
            y_line = model(x_line, *popt)

            formula_text = "y = " + formula
            coef_text = "\n".join([f"{chr(97 + i)} = {popt[i]:.4g} ± {sigma[i]:.2g}" for i in range(len(popt))])
            plt.errorbar(x, y, yerr=sigma_y, xerr=sigma_x, fmt='o', capsize=5, color='black')

            self.plot_with_errors(x, y, sigma_y, sigma_x, x_line, y_line, xlabel, ylabel, label, formula_text, coef_text)

        except Exception as e:
            messagebox.showerror("Ошибка подгонки", f"Не удалось аппроксимировать:\n{e}")

    def plot_with_errors(self, x, y, sigma_y, sigma_x, x_line, y_line, xlabel, ylabel, label, formula_text,
                         coef_text):
        plt.figure(figsize=(10, 6))
        plt.errorbar(x, y, yerr=sigma_y, xerr=sigma_x, fmt='o', capsize=5, color='black', label='Данные')
        plt.plot(x_line, y_line, 'r-', label=label or "Аппроксимация")
        if hasattr(self, "calc_area_var") and self.calc_area_var.get():
            area = np.trapz(y_line, x_line)
            plt.figtext(0.1, 0.05, f"Площадь под графиком: {area:.6g}", fontsize=10)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title("Аппроксимация")
        plt.legend()
        plt.grid(True)
        plt.figtext(0.98, 0.5, formula_text, fontsize=10, ha="right", va="top")
        plt.figtext(0.98, 0.3, coef_text, fontsize=9, ha="right", va="top")
        plt.tight_layout()
        plt.show()


    def open_error_menu(self):
        err_win = tk.Toplevel(self.root)
        err_win.title("Обработка и погрешности")

        tk.Button(err_win, text="Погрешность прямых измерений", command=self.direct_error_window).pack(padx=10, pady=10)
        tk.Button(err_win, text="Погрешность косвенных измерений", command=self.indirect_error_window).pack(padx=10, pady=10)

    def direct_error_window(self):
        win = tk.Toplevel(self.root)
        win.title("Погрешность прямых измерений")

        tk.Label(win, text="σ = √[1 / n(n - 1) * Σ(s - x_i)^2]", font=("Arial", 12)).pack(pady=5)
        tk.Label(win, text="Введите значения измерений по одному в столбик").pack(pady=5)

        canvas = tk.Canvas(win)
        frame = tk.Frame(canvas)
        vsb = tk.Scrollbar(win, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        canvas.create_window((0, 0), window=frame, anchor='nw')
        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        entries = []
        for i in range(30):
            e = tk.Entry(frame, width=10)
            e.grid(row=i, column=0, pady=1)
            e.bind("<Return>", lambda event, r=i: entries[r+1].focus_set() if r+1 < len(entries) else None)
            entries.append(e)

        def calc():
            try:
                values = [float(e.get()) for e in entries if e.get().strip() != ""]
                n = len(values)
                if n < 2:
                    raise ValueError("Введите минимум 2 значения")
                s = sum(values) / n
                variance = sum((s - val) ** 2 for val in values) / (n * (n - 1))
                sigma = np.sqrt(variance)
                messagebox.showinfo("Результат", f"Среднее: {s:.4f}\nПогрешность: {sigma:.4f}")
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))

        tk.Button(frame, text="Рассчитать", command=calc).grid(row=0, column=1, padx=10)

    def indirect_error_window(self):
        win = tk.Toplevel(self.root)
        win.title("Погрешность косвенных измерений")

        tk.Label(win, text="Введите формулу (пример: 2*A*(w*T0)/(b*p))", font=("Courier", 14)).pack(anchor="n", pady=20)
        #tk.Label(win, text="f =", font=("Arial", 12, "bold")).pack(anchor="nw")
        form_frame = tk.Frame(win)
        form_frame.pack(pady=5)
        tk.Label(form_frame, text="f =", font=("Arial", 11)).pack(side="left", padx=5)
        entry = tk.Entry(form_frame, width=40)
        entry.pack(side="left")

        tk.Label(win, text=("f — функция, значение и погрешность которой считаются,\n"
                            " знак '=' и 'f=' в уравнении не пишутся"
                            ), font=("Arial", 14), fg="red").pack()

        tk.Label(win, text=(
            "• * — умножение, / — деление\n"
            "• Скобки при делении: (a+b)/(c+d)\n"
            "• Степени: ^ или **\n"
            "• Логарифм: log(x, a), ln(x) — натуральный логарифм\n"
            "• exp(x), π = 3.14, e = 2.71\n"
            "• Формат: 2.5e-6 или 2.5*10**-6"
        ), justify="left", font=("Arial", 14)).pack(pady=10)

        def next_step():
            try:
                expr_str = entry.get().replace("^", "**").replace("ln", "log")
                formula = sp.sympify(expr_str, locals={"log": lambda x, a=sp.E: sp.log(x, a), "exp": sp.exp})
                vars_ = sorted(formula.free_symbols, key=lambda s: s.name)
                var_names = [str(v) for v in vars_]
                self.show_value_entry(var_names, formula)
            except Exception as e:
                messagebox.showerror("Ошибка", f"Проверьте формулу: {e}")

        tk.Button(win, text="Далее", command=next_step).pack(pady=10)

    def show_value_entry(self, var_names, formula):
        win = tk.Toplevel(self.root)
        win.title("Ввод значений переменных")

        tk.Label(win, text="Введите значения переменных и их погрешности:").pack(pady=3)

        partial_display = []
        subs_symbols = [sp.Symbol(name) for name in var_names]
        for var in subs_symbols:
            df_symbolic = sp.simplify(formula.diff(var))
            partial_display.append(f"∂f/∂{var} = {df_symbolic}")

        for text in partial_display:
            tk.Label(win, text=text, font=("Courier", 12)).pack(anchor="w")

        canvas = tk.Canvas(win)
        frame = tk.Frame(canvas)
        vsb = tk.Scrollbar(win, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)

        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        canvas.create_window((0, 0), window=frame, anchor='nw')
        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        tk.Label(frame, text="Переменная").grid(row=0, column=0)
        tk.Label(frame, text="Значение").grid(row=0, column=1)
        tk.Label(frame, text="Погрешность").grid(row=0, column=2)

        entries = []
        for i, name in enumerate(var_names):
            tk.Label(frame, text=name).grid(row=i+1, column=0)
            row_entries = []
            for j in range(2):
                e = tk.Entry(frame, width=12)
                e.grid(row=i + 1, column=j + 1)

                # Добавим навигацию по стрелкам и Enter
                e.bind("<Return>", lambda event, r=i, c=j: self.focus_entry_grid(event, entries, r, c))
                e.bind("<Down>", lambda event, r=i, c=j: self.focus_entry_grid(event, entries, r, c))
                e.bind("<Up>", lambda event, r=i, c=j: self.focus_entry_grid(event, entries, r, c))
                e.bind("<Left>", lambda event, r=i, c=j: self.focus_entry_grid(event, entries, r, c))
                e.bind("<Right>", lambda event, r=i, c=j: self.focus_entry_grid(event, entries, r, c))

                row_entries.append(e)

        def calculate():
            try:
                subs = {}
                sigmas = {}
                for i, name in enumerate(var_names):
                    val = float(eval(entries[i][0].get().replace("^", "**")))
                    sigma = float(eval(entries[i][1].get().replace("^", "**")))
                    subs[sp.Symbol(name)] = val
                    sigmas[name] = sigma

                f_val = float(formula.evalf(subs=subs))
                sigma_sq_sum = 0.0
                for var in formula.free_symbols:
                    if sigmas[str(var)] == 0:
                        continue
                    df = formula.diff(var).evalf(subs=subs)
                    sigma_sq_sum += (float(df) * sigmas[str(var)]) ** 2

                sigma_f = np.sqrt(sigma_sq_sum)

                messagebox.showinfo("Результат", f"f = {f_val:.6g} ± {sigma_f:.6g}")

            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка вычислений: {e}")

        tk.Button(frame, text="Рассчитать", command=calculate).grid(row=0, column=3, padx=10, pady=5)

    def focus_next_cell(self, event, row, col):
        widget = event.widget
        try:
            if event.keysym == "Down" and row + 1 < len(self.entries):
                self.entries[row + 1][col].focus_set()
            elif event.keysym == "Up" and row - 1 >= 0:
                self.entries[row - 1][col].focus_set()
            elif event.keysym == "Return":
                if row == len(self.entries) - 1 and len(self.entries) < 50:
                    self.add_row(row + 1)
                self.entries[min(row + 1, len(self.entries)-1)][col].focus_set()
            elif event.keysym == "Right":
                if widget.index(tk.INSERT) == len(widget.get()):
                    if col + 1 < len(self.entries[0]):
                        self.entries[row][col + 1].focus_set()
            elif event.keysym == "Left":
                if widget.index(tk.INSERT) == 0:
                    if col - 1 >= 0:
                        self.entries[row][col - 1].focus_set()
        except:
            pass

    def add_row(self, index):
        row_entries = []
        for j in range(4):
            e = tk.Entry(self.entries[0][j].master, width=10)
            e.grid(row=index+3, column=j+1, padx=2, pady=1)
            e.bind("<Return>", partial(self.focus_next_cell, row=index, col=j))
            e.bind("<Down>", partial(self.focus_next_cell, row=index, col=j))
            e.bind("<Up>", partial(self.focus_next_cell, row=index, col=j))
            e.bind("<Left>", partial(self.focus_next_cell, row=index, col=j))
            e.bind("<Right>", partial(self.focus_next_cell, row=index, col=j))
            row_entries.append(e)
        self.entries.append(row_entries)
        tk.Label(self.entries[0][0].master, text=str(index + 1)).grid(row=index+3, column=0)

        if self.no_sy:
            row_entries[2].insert(0, "0")
        if self.no_sx:
            row_entries[3].insert(0, "0")

    def show_value_entry(self, var_names, formula):
        win = tk.Toplevel(self.root)
        win.title("Ввод значений переменных")

        tk.Label(win, text="Введите значения величин и их погрешностей, представленных в формуле:").pack(pady=5)

        canvas = tk.Canvas(win)
        frame = tk.Frame(canvas)
        vsb = tk.Scrollbar(win, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.create_window((0, 0), window=frame, anchor='nw')
        frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        tk.Label(frame, text="Переменная").grid(row=0, column=0)
        tk.Label(frame, text="Значение").grid(row=0, column=1)
        tk.Label(frame, text="Погрешность").grid(row=0, column=2)

        entries = []
        for i, name in enumerate(var_names):
            tk.Label(frame, text=name).grid(row=i + 1, column=0)
            row_entries = []
            for j in range(2):
                e = tk.Entry(frame, width=12)
                e.grid(row=i + 1, column=j + 1)

                e.bind("<Return>", lambda event, r=i, c=j: self.focus_entry_grid(event, entries, r, c))
                e.bind("<Down>", lambda event, r=i, c=j: self.focus_entry_grid(event, entries, r, c))
                e.bind("<Up>", lambda event, r=i, c=j: self.focus_entry_grid(event, entries, r, c))
                e.bind("<Left>", lambda event, r=i, c=j: self.focus_entry_grid(event, entries, r, c))
                e.bind("<Right>", lambda event, r=i, c=j: self.focus_entry_grid(event, entries, r, c))

                row_entries.append(e)
            entries.append(row_entries)

        partials_text = tk.Label(frame, text="", justify="left", font=("Arial", 12))
        partials_text.grid(row=len(var_names) + 3, column=0, columnspan=3, pady=(10, 0), sticky="s")

        def update_partials():
            try:
                subs = {sp.Symbol(var_names[i]): float(entries[i][0].get()) for i in range(len(var_names))}
                text = ""
                for var in formula.free_symbols:
                    df = formula.diff(var)
                    text += f"∂f/∂{var} = {sp.simplify(df)}\n"
                partials_text.config(text=text)
            except:
                partials_text.config(text="")

        for row in entries:
            for entry in row:
                entry.bind("<KeyRelease>", lambda e: update_partials())

        def calc_result():
            try:
                subs = {}
                sigmas = {}
                for i, name in enumerate(var_names):
                    val_str = entries[i][0].get().replace("^", "**")
                    err_str = entries[i][1].get().replace("^", "**")
                    val = float(eval(val_str))
                    err = float(eval(err_str))
                    subs[sp.Symbol(name)] = val
                    sigmas[name] = err

                f_val = float(formula.evalf(subs=subs))
                partials = []
                for var in formula.free_symbols:
                    if sigmas[str(var)] == 0:
                        continue
                    df_numeric = float(formula.diff(var).evalf(subs=subs))
                    partials.append((df_numeric, sigmas[str(var)]))
                sigma_f = np.sqrt(sum((df * s) ** 2 for df, s in partials))

                messagebox.showinfo("Результат", f"f = {f_val:.6g} ± {sigma_f:.6g}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при расчёте: {e}")

        btn_next = tk.Button(frame, text="Далее", command=calc_result)
        btn_next.grid(row=0, column=3, padx=10, pady=10, sticky="ne")

    def focus_entry_grid(self, e, entries, r, c):
        try:
            if e.keysym == "Return" or e.keysym == "Down":
                entries[r + 1][c].focus_set()
            elif e.keysym == "Up" and r > 0:
                entries[r - 1][c].focus_set()
            elif e.keysym == "Right" and c < 1:
                entries[r][c + 1].focus_set()
            elif e.keysym == "Left" and c > 0:
                entries[r][c - 1].focus_set()
        except:
            pass

    def paste_from_clipboard(self, start_row, start_col):
        try:
            clipboard = self.root.clipboard_get()
            rows = clipboard.strip().split('\n')
            for i, line in enumerate(rows):
                values = line.strip().split('\t')  # табуляции из Excel
                for j, val in enumerate(values):
                    r, c = start_row + i, start_col + j
                    if r < len(self.entries) and c < len(self.entries[0]):
                        self.entries[r][c].delete(0, tk.END)
                        self.entries[r][c].insert(0, val.strip())
        except Exception as e:
            messagebox.showerror("Ошибка вставки", f"Не удалось вставить данные: {e}")

    def handle_paste(self, row, col, event=None):
        self.paste_from_clipboard(row, col)

    def paste_from_clipboard(self, start_row, start_col):
        try:
            content = self.root.clipboard_get()
            lines = content.strip().splitlines()

            for i, line in enumerate(lines):
                cells = line.strip().split('\t')
                for j, cell in enumerate(cells):
                    row_idx = start_row + i
                    col_idx = start_col + j
                    if row_idx < len(self.entries) and col_idx < len(self.entries[0]):
                        self.entries[row_idx][col_idx].delete(0, tk.END)
                        self.entries[row_idx][col_idx].insert(0, cell)
        except Exception as e:
            messagebox.showerror("Ошибка вставки", f"Не удалось вставить данные:\n{e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = MNKApp(root)
    root.mainloop()
