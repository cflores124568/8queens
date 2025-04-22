import random
import math
import time
import tkinter as tk
from tkinter import messagebox, scrolledtext
from tkinter import ttk

# 8 Queens Algorithms
def heuristic(P):
    count = 0
    for i in range(8):
        for j in range(i + 1, 8):
            if abs(P[i] - P[j]) == abs(i - j):
                count += 1
    return count

def random_permutation():
    P = list(range(8))
    random.shuffle(P)
    return P

def validate_state(state):
    if not isinstance(state, list) or len(state) != 8 or set(state) != set(range(8)):
        raise ValueError("State must be a permutation of [0,1,...,7]")

def hill_climbing(initial_state=None):
    P = initial_state if initial_state is not None else random_permutation()
    validate_state(P)
    current_h = heuristic(P)
    iterations = 0
    improved = True
    max_states = 2  # Current state and one swapped state
    while improved:
        iterations += 1
        improved = False
        best_P = P
        best_h = current_h
        for i in range(8):
            for j in range(i + 1, 8):
                P_swap = P.copy()
                P_swap[i], P_swap[j] = P_swap[j], P_swap[i]
                h_swap = heuristic(P_swap)
                if h_swap < best_h:
                    best_h = h_swap
                    best_P = P_swap
                    improved = True
        if improved:
            P = best_P
            current_h = best_h
    return P, current_h, iterations, max_states

def simulated_annealing(initial_state=None):
    P = initial_state if initial_state is not None else random_permutation()
    validate_state(P)
    T = 1000
    T_min = 1
    alpha = 0.99
    iterations = 0
    max_states = 2  # Current state and one swapped state
    while T > T_min:
        iterations += 1
        i, j = random.sample(range(8), 2)
        P_swap = P.copy()
        P_swap[i], P_swap[j] = P_swap[j], P_swap[i]
        delta_h = heuristic(P_swap) - heuristic(P)
        if delta_h < 0 or random.random() < math.exp(-delta_h / T):
            P = P_swap
        T *= alpha
        if heuristic(P) == 0:
            return P, 0, iterations, max_states
    return P, heuristic(P), iterations, max_states

def genetic_algorithm(initial_state=None, pop_size=100, generations=1000):
    def fitness(P):
        return 1 / (1 + heuristic(P))
    
    initial_P = initial_state if initial_state is not None else random_permutation()
    validate_state(initial_P)
    population = [initial_P] + [random_permutation() for _ in range(pop_size - 1)]
    iterations = 0
    max_states = pop_size * 2  # Population + new population
    for gen in range(generations):
        iterations += 1
        new_population = []
        for _ in range(pop_size):
            candidates = random.sample(population, 3)
            parent1 = max(candidates, key=fitness)
            candidates = random.sample(population, 3)
            parent2 = max(candidates, key=fitness)
            child = []
            for i in range(8):
                child.append(random.choice([parent1[i], parent2[i]]))
            if random.random() < 0.1:
                i, j = random.sample(range(8), 2)
                child[i], child[j] = child[j], child[i]
            new_population.append(child)
        best = max(population + new_population, key=fitness)
        population = new_population
        if fitness(best) == 1:
            return best, 0, iterations, max_states
    best = max(population, key=fitness)
    return best, heuristic(best), iterations, max_states

# Test Cases
TEST_CASES = [
    {
        "name": "Easy",
        "state": [0, 6, 4, 7, 1, 3, 5, 2],
        "heuristic": 1,
        "description": "Near-optimal state with one conflict",
        "expected": "Heuristic = 0 (solution expected)"
    },
    {
        "name": "Medium",
        "state": [0, 4, 7, 5, 2, 6, 1, 3],
        "heuristic": 4,
        "description": "Moderate conflicts requiring multiple moves",
        "expected": "Heuristic = 0 or low (depends on algorithm)"
    },
    {
        "name": "Hard",
        "state": [0, 1, 2, 3, 4, 5, 6, 7],
        "heuristic": 7,
        "description": "Worst-case state with queens on main diagonal",
        "expected": "Heuristic may remain high for hill-climbing"
    },
    {
        "name": "Perfect",
        "state": [1, 7, 5, 0, 2, 4, 6, 3],
        "heuristic": 0,
        "description": "Already a solution (no conflicts)",
        "expected": "Heuristic = 0 (no changes needed)"
    },
    {
        "name": "Edge-Invalid",
        "state": [0, 0, 0, 0, 0, 0, 0, 0],
        "heuristic": 28,
        "description": "Invalid state with all queens in row 0",
        "expected": "Should handle gracefully, may not find solution"
    }
]

# GUI Implementation
class EightQueensGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("8-Queens Problem Solver")
        self.cell_size = 50
        self.board_size = 8
        self.canvas_size = self.board_size * self.cell_size
        self.queens = []
        self.solutions = []
        self.current_solution_idx = -1
        self.last_algorithm = None

        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(padx=10, pady=10)

        self.canvas = tk.Canvas(self.main_frame, width=self.canvas_size, height=self.canvas_size)
        self.canvas.grid(row=0, column=0, columnspan=6)

        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.grid(row=1, column=0, columnspan=6, pady=5)
        tk.Button(self.button_frame, text="Run Hill-Climbing", command=self.run_hill_climbing).pack(side=tk.LEFT, padx=5)
        tk.Button(self.button_frame, text="Run Simulated Annealing", command=self.run_simulated_annealing).pack(side=tk.LEFT, padx=5)
        tk.Button(self.button_frame, text="Run Genetic Algorithm", command=self.run_genetic_algorithm).pack(side=tk.LEFT, padx=5)
        tk.Button(self.button_frame, text="Run Test Cases", command=self.run_test_cases).pack(side=tk.LEFT, padx=5)
        tk.Button(self.button_frame, text="Analyze Performance", command=self.analyze_performance).pack(side=tk.LEFT, padx=5)
        tk.Button(self.button_frame, text="Reset", command=self.reset).pack(side=tk.LEFT, padx=5)

        self.nav_frame = tk.Frame(self.main_frame)
        self.nav_frame.grid(row=2, column=0, columnspan=6, pady=5)
        tk.Button(self.nav_frame, text="Previous Solution", command=self.show_prev_solution).pack(side=tk.LEFT, padx=5)
        tk.Button(self.nav_frame, text="Next Solution", command=self.show_next_solution).pack(side=tk.LEFT, padx=5)

        self.input_frame = tk.Frame(self.main_frame)
        self.input_frame.grid(row=3, column=0, columnspan=6, pady=5)
        tk.Label(self.input_frame, text="Custom State (e.g., 0,6,4,7,1,3,5,2):").pack(side=tk.LEFT)
        self.state_entry = tk.Entry(self.input_frame, width=30)
        self.state_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(self.input_frame, text="Run with Custom State", command=self.run_custom_state).pack(side=tk.LEFT, padx=5)

        self.status_label = tk.Label(self.main_frame, text="Select an algorithm to solve", font=("Arial", 12))
        self.status_label.grid(row=4, column=0, columnspan=6, pady=5)

        self.solution_info = tk.Label(self.main_frame, text="No solutions available", font=("Arial", 10))
        self.solution_info.grid(row=5, column=0, columnspan=6)

        self.log_text = scrolledtext.ScrolledText(self.main_frame, height=10, width=60, font=("Arial", 10))
        self.log_text.grid(row=6, column=0, columnspan=6, pady=5)
        self.log_text.config(state='disabled')

        self.table_frame = None  # Initialize table frame for performance results

        self.draw_board()

    def draw_board(self):
        self.canvas.delete("all")
        for row in range(self.board_size):
            for col in range(self.board_size):
                color = "white" if (row + col) % 2 == 0 else "black"
                x1, y1 = col * self.cell_size, row * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color)

    def place_queens(self, P):
        self.canvas.delete("queen")
        self.queens = []
        for col in range(8):
            row = P[col]
            x = col * self.cell_size + self.cell_size // 2
            y = row * self.cell_size + self.cell_size // 2
            queen = self.canvas.create_text(x, y, text="♕", font=("Arial", 24), 
                                          fill="red" if heuristic(P) == 0 else "blue", tags="queen")
            self.queens.append(queen)

    def update_status(self, h, algorithm, iterations, runtime):
        status = f"{algorithm}: Heuristic = {h}, Iterations = {iterations}, Time = {runtime:.3f}s"
        if h == 0:
            status += " (Solution Found!)"
        else:
            status += " (Local Optimum)"
        self.status_label.config(text=status)

    def log_message(self, message):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def reset(self):
        self.canvas.delete("queen")
        self.queens = []
        self.solutions = self.solutions[-100:]  # Keep last 100 solutions
        self.current_solution_idx = min(self.current_solution_idx, len(self.solutions) - 1) if self.solutions else -1
        self.last_algorithm = None
        self.status_label.config(text="Select an algorithm to solve")
        self.solution_info.config(text="No solutions available")
        self.log_text.config(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state='disabled')
        if self.table_frame:
            self.table_frame.destroy()
            self.table_frame = None
        self.draw_board()

    def show_prev_solution(self):
        if not self.solutions:
            messagebox.showwarning("No Solutions", "Run an algorithm or test cases to find solutions.")
            self.log_message("No solutions available to cycle through.")
            return
        self.current_solution_idx = (self.current_solution_idx - 1) % len(self.solutions)
        self.display_current_solution()

    def show_next_solution(self):
        if not self.solutions:
            messagebox.showwarning("No Solutions", "Run an algorithm or test cases to find solutions.")
            self.log_message("No solutions available to cycle through.")
            return
        self.current_solution_idx = (self.current_solution_idx + 1) % len(self.solutions)
        self.display_current_solution()

    def display_current_solution(self):
        if not self.solutions or self.current_solution_idx < 0:
            return
        solution = self.solutions[self.current_solution_idx]
        state = solution["state"]
        h = solution["heuristic"]
        algorithm = solution["algorithm"]
        source = solution["source"]
        run = solution["run"]
        self.place_queens(state)
        solution_type = "Perfect Solution" if h == 0 else f"Local Optimum (h={h})"
        self.solution_info.config(
            text=f"{source} | Run {run} | State {self.current_solution_idx + 1}/{len(self.solutions)} | "
                 f"{solution_type} | Algorithm: {algorithm}"
        )
        self.log_message(f"Showing state {self.current_solution_idx + 1}/{len(self.solutions)} ({source}, Run {run})")
        self.log_message(f"State: {state}, Heuristic: {h}, Algorithm: {algorithm}")

    def run_algorithm(self, algorithm_func, name, initial_state=None, num_runs=5):
        self.log_message(f"\nRunning {name} ({num_runs} runs)...")
        source = "Custom" if initial_state is not None else "Random"
        
        run_results = []
        for run in range(1, num_runs + 1):
            start_time = time.time()
            current_state = initial_state if run == 1 and initial_state is not None else random_permutation()
            self.log_message(f"Run {run} - Initial state: {current_state}")
            
            try:
                P, h, iterations, max_states = algorithm_func(initial_state=current_state)
                runtime = time.time() - start_time
                self.solutions.append({
                    "state": P,
                    "heuristic": h,
                    "algorithm": name,
                    "source": source,
                    "run": run,
                    "iterations": iterations,
                    "runtime": runtime,
                    "max_states": max_states
                })
                run_results.append((P, h, iterations, runtime, max_states))
                self.log_message(f"Run {run} - Final state: {P} (Heuristic: {h}, Iterations: {iterations}, Time: {runtime:.3f}s, Max States: {max_states})")
            except ValueError as e:
                self.log_message(f"Run {run} - Error: {e}")
                continue
        
        if not run_results:
            messagebox.showerror("Error", f"{name} failed all runs due to invalid states.")
            return None, None, None
        
        best_result = min(run_results, key=lambda x: x[1])  # Best by heuristic
        P, h, iterations, runtime, max_states = best_result
        self.current_solution_idx = len(self.solutions) - 1
        self.last_algorithm = name
        
        self.place_queens(P)
        self.update_status(h, name, iterations, runtime)
        self.display_current_solution()
        
        perfect_solutions = sum(1 for _, h, _, _, _ in run_results if h == 0)
        if perfect_solutions > 0:
            messagebox.showinfo("Result", f"{name} found {perfect_solutions}/{num_runs} solutions. Navigate to see all results.")
        else:
            messagebox.showwarning("Result", f"{name} found no solutions. Best heuristic: {h}")
        
        return h, iterations, runtime

    def run_test_cases(self):
        self.reset()
        self.log_message("Running test cases...")
        with open("test_case_results.txt", "w") as f:
            f.write("8-Queens Test Case Results\n")
            f.write("=" * 50 + "\n")
            algorithms = [
                (hill_climbing, "Hill-Climbing"),
                (simulated_annealing, "Simulated Annealing"),
                (genetic_algorithm, "Genetic Algorithm")
            ]
            for test_case in TEST_CASES:
                self.log_message(f"\nTest Case: {test_case['name']} (Initial Heuristic = {test_case['heuristic']})")
                self.log_message(f"Description: {test_case['description']}")
                self.log_message(f"Initial state: {test_case['state']}")
                f.write(f"\nTest Case: {test_case['name']}\n")
                f.write(f"Initial State: {test_case['state']}\n")
                f.write(f"Initial Heuristic: {test_case['heuristic']}\n")
                f.write(f"Description: {test_case['description']}\n")
                f.write(f"Expected: {test_case['expected']}\n")
                for algo_func, algo_name in algorithms:
                    self.log_message(f"  Running {algo_name}...")
                    try:
                        h, iterations, runtime = self.run_algorithm(algo_func, algo_name, test_case['state'], num_runs=5)
                        if h is None:
                            continue
                        for sol in self.solutions[-5:]:
                            sol["source"] = f"Test Case: {test_case['name']}"
                        f.write(f"  {algo_name}:\n")
                        for sol in self.solutions[-5:]:
                            f.write(f"    Run {sol['run']} - State: {sol['state']}, Heuristic: {sol['heuristic']}, "
                                    f"Iterations: {sol['iterations']}, Runtime: {sol['runtime']:.3f}s, "
                                    f"Max States: {sol['max_states']}, Success: {'Yes' if sol['heuristic'] == 0 else 'No'}\n")
                        self.log_message(f"  Best Result: Heuristic = {h}, Iterations = {iterations}, Time = {runtime:.3f}s")
                    except ValueError as e:
                        self.log_message(f"  Error in {algo_name}: {e}")
                        f.write(f"  {algo_name}: Error - {e}\n")
                    self.root.update()
                    time.sleep(0.5)
                f.write("-" * 50 + "\n")
        self.log_message("Test cases completed. Results saved to test_case_results.txt")
        messagebox.showinfo("Test Cases", "Test cases completed. Check log and test_case_results.txt")

    def run_custom_state(self):
        input_str = self.state_entry.get().strip()
        try:
            state = [int(x) for x in input_str.split(",")]
            validate_state(state)
            self.log_message(f"Running algorithms with custom state: {state}")
            algorithms = [
                (hill_climbing, "Hill-Climbing"),
                (simulated_annealing, "Simulated Annealing"),
                (genetic_algorithm, "Genetic Algorithm")
            ]
            for algo_func, algo_name in algorithms:
                h, iterations, runtime = self.run_algorithm(algo_func, algo_name, state, num_runs=5)
                if h is None:
                    continue
                self.log_message(f"{algo_name} Best Result: Heuristic = {h}, Iterations = {iterations}, Time = {runtime:.3f}s")
                self.root.update()
                time.sleep(0.5)
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Error: {e}")
            self.log_message(f"Invalid input: {input_str}")

    def display_performance_table(self, metrics):
        if self.table_frame:
            self.table_frame.destroy()
        
        self.table_frame = tk.Frame(self.main_frame)
        self.table_frame.grid(row=7, column=0, columnspan=6, pady=5)
        tree = ttk.Treeview(self.table_frame, columns=(
            "Algorithm", "Success", "Heuristic", "Iterations", "Runtime", "States"
        ), show="headings")
        tree.heading("Algorithm", text="Algorithm")
        tree.heading("Success", text="Success Rate (%)")
        tree.heading("Heuristic", text="Avg Heuristic")
        tree.heading("Iterations", text="Avg Iterations (Success)")
        tree.heading("Runtime", text="Avg Runtime (s)")
        tree.heading("States", text="Avg Max States")
        tree.column("Algorithm", width=100)
        tree.column("Success", width=100)
        tree.column("Heuristic", width=100)
        tree.column("Iterations", width=120)
        tree.column("Runtime", width=100)
        tree.column("States", width=100)
        
        for algo, data in metrics.items():
            tree.insert("", "end", values=(
                algo,
                f"{data['success'] / data['total_runs'] * 100:.2f}" if data['total_runs'] > 0 else "0.00",
                f"{data['avg_heuristic']:.1f}" if data['total_runs'] > 0 else "0.0",
                f"{data['successful_iterations'] / data['successful_runs'] if data['successful_runs'] > 0 else 0:.1f}",
                f"{data['avg_time']:.3f}" if data['total_runs'] > 0 else "0.000",
                f"{data['avg_max_states']:.1f}" if data['total_runs'] > 0 else "0.0"
            ))
        tree.pack()

    def analyze_performance(self):
        self.log_message("\nPerformance Analysis:")
        with open("test_case_results.txt", "a") as f:
            f.write("\nPerformance Summary\n")
            f.write("=" * 50 + "\n")
            algorithms = ["Hill-Climbing", "Simulated Annealing", "Genetic Algorithm"]
            metrics = {
                algo: {
                    "success": 0,
                    "avg_time": 0,
                    "avg_iterations": 0,
                    "avg_heuristic": 0,
                    "avg_max_states": 0,
                    "total_runs": 0,
                    "successful_iterations": 0,
                    "successful_runs": 0
                } for algo in algorithms
            }
            
            for test_case in TEST_CASES:
                for algo_func, algo_name in [
                    (hill_climbing, "Hill-Climbing"),
                    (simulated_annealing, "Simulated Annealing"),
                    (genetic_algorithm, "Genetic Algorithm")
                ]:
                    for run in range(5):
                        try:
                            initial_state = test_case["state"] if run == 0 else random_permutation()
                            start_time = time.time()
                            P, h, iterations, max_states = algo_func(initial_state)
                            runtime = time.time() - start_time
                            metrics[algo_name]["total_runs"] += 1
                            if h == 0:
                                metrics[algo_name]["success"] += 1
                                metrics[algo_name]["successful_iterations"] += iterations
                                metrics[algo_name]["successful_runs"] += 1
                            metrics[algo_name]["avg_time"] += runtime
                            metrics[algo_name]["avg_iterations"] += iterations
                            metrics[algo_name]["avg_heuristic"] += h
                            metrics[algo_name]["avg_max_states"] += max_states
                        except ValueError:
                            pass
            
            for algo in algorithms:
                total = metrics[algo]["total_runs"]
                if total == 0:
                    continue
                success_rate = metrics[algo]["success"] / total * 100
                avg_time = metrics[algo]["avg_time"] / total
                avg_iterations = metrics[algo]["avg_iterations"] / total
                avg_heuristic = metrics[algo]["avg_heuristic"] / total
                avg_max_states = metrics[algo]["avg_max_states"] / total
                avg_successful_iterations = metrics[algo]["successful_iterations"] / metrics[algo]["successful_runs"] if metrics[algo]["successful_runs"] > 0 else 0
                
                self.log_message(f"{algo}:")
                self.log_message(f"  Completeness (Success Rate): {success_rate:.2f}%")
                self.log_message(f"  Cost Optimality (Avg Heuristic): {avg_heuristic:.1f}")
                self.log_message(f"  Cost Optimality (Avg Iterations for Success): {avg_successful_iterations:.1f}")
                self.log_message(f"  Time Complexity (Avg Runtime): {avg_time:.3f}s")
                self.log_message(f"  Space Complexity (Avg Max States): {avg_max_states:.1f}")
                f.write(f"{algo}:\n")
                f.write(f"  Completeness (Success Rate): {success_rate:.2f}%\n")
                f.write(f"  Cost Optimality (Avg Heuristic): {avg_heuristic:.1f}\n")
                f.write(f"  Cost Optimality (Avg Iterations for Success): {avg_successful_iterations:.1f}\n")
                f.write(f"  Time Complexity (Avg Runtime): {avg_time:.3f}s\n")
                f.write(f"  Space Complexity (Avg Max States): {avg_max_states:.1f}\n")
            
            self.display_performance_table(metrics)

    def run_hill_climbing(self):
        self.run_algorithm(hill_climbing, "Hill-Climbing")

    def run_simulated_annealing(self):
        self.run_algorithm(simulated_annealing, "Simulated Annealing")

    def run_genetic_algorithm(self):
        self.run_algorithm(genetic_algorithm, "Genetic Algorithm")

if __name__ == "__main__":
    root = tk.Tk()
    app = EightQueensGUI(root)
    root.mainloop()
