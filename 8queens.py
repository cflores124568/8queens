import random
import math
import time
import tkinter as tk
from tkinter import messagebox, scrolledtext
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# 8 Queens Algorithms
def heuristic(P):
    # Calculate the number of conflicts (attacking pairs) in a given 8-queens state
    count = 0
    for i in range(8):
        for j in range(i + 1, 8):
            # Check if queens in columns i and j are on the same diagonal
            if abs(P[i] - P[j]) == abs(i - j):
                count += 1
    return count

def random_permutation():
    # Generate a random permutation of [0,1,...,7] for initial queen placements
    P = list(range(8))
    random.shuffle(P)
    return P

def validate_state(state):
    # Validate that a state is a valid permutation for the 8-queens problem
    if not isinstance(state, list) or len(state) != 8 or set(state) != set(range(8)):
        raise ValueError("State must be a permutation of [0,1,...,7]")

def hill_climbing(initial_state=None):
    # Implement the hill-climbing algorithm to solve the 8-queens problem
    P = initial_state if initial_state is not None else random_permutation()
    validate_state(P)
    current_h = heuristic(P)  # Current number of conflicts
    iterations = 0  # Track number of iterations
    improved = True  # Flag to continue searching
    max_states = 2  # Current state and one swapped state
    while improved:
        iterations += 1
        improved = False
        best_P = P  # Best state found in this iteration
        best_h = current_h  # Best heuristic value
        for i in range(8):
            for j in range(i + 1, 8):
                P_swap = P.copy()
                P_swap[i], P_swap[j] = P_swap[j], P_swap[i]  # Swap two queens
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
    # Implement simulated annealing for the 8-queens problem
    P = initial_state if initial_state is not None else random_permutation()
    validate_state(P)
    T = 1000  # Initial temperature
    T_min = 1  # Minimum temperature
    alpha = 0.99  # Cooling rate
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
    # Implement a genetic algorithm for the 8-queens problem
    def fitness(P):
        # Calculate fitness as inverse of conflicts (higher is better)
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
            child = [random.choice([parent1[i], parent2[i]]) for i in range(8)]
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
    },
    {
        "name": "Invalid-OutOfRange",
        "state": [0, 1, 2, 3, 4, 5, 6, 8],
        "heuristic": None,
        "description": "State with value outside [0,7]",
        "expected": "Raises ValueError"
    },
    {
        "name": "Invalid-Missing",
        "state": [0, 1, 2, 3, 4, 5, 6, 6],
        "heuristic": None,
        "description": "State with duplicate values, missing 7",
        "expected": "Raises ValueError"
    },
    {
        "name": "Hard-Reverse",
        "state": [7, 6, 5, 4, 3, 2, 1, 0],
        "heuristic": 7,
        "description": "High-conflict state with queens on reverse diagonal",
        "expected": "Heuristic may remain high for hill-climbing"
    },
    {
        "name": "LocalOptima",
        "state": [4, 6, 0, 2, 7, 5, 3, 1],
        "heuristic": 2,
        "description": "State designed to trap hill-climbing in a local optimum",
        "expected": "Heuristic may remain > 0 for hill-climbing"
    }
]

# GUI Implementation
class EightQueensGUI:
    # GUI class for visualizing and solving the 8-queens problem
    def __init__(self, root):
        self.root = root
        self.root.title("8-Queens Problem Solver")
        self.cell_size = 50  # Pixel size of each chessboard cell
        self.board_size = 8  # Standard 8x8 chessboard
        self.canvas_size = self.board_size * self.cell_size  # Total canvas size
        self.queens = []  # Store canvas objects for queens
        self.solutions = []  # Store all solutions found
        self.current_solution_idx = -1  # Track current solution index
        self.last_algorithm = None  # Last algorithm executed
        self.performance_metrics = None  # Store performance data

        # Create main frame to hold all widgets
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(padx=10, pady=10)

        # Create canvas for the chessboard
        self.canvas = tk.Canvas(self.main_frame, width=self.canvas_size, height=self.canvas_size)
        self.canvas.grid(row=0, column=0, columnspan=6)

        # Create frame for algorithm and action buttons
        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.grid(row=1, column=0, columnspan=6, pady=5)
        tk.Button(self.button_frame, text="Run Hill-Climbing", command=self.run_hill_climbing).pack(side=tk.LEFT, padx=5)
        tk.Button(self.button_frame, text="Run Simulated Annealing", command=self.run_simulated_annealing).pack(side=tk.LEFT, padx=5)
        tk.Button(self.button_frame, text="Run Genetic Algorithm", command=self.run_genetic_algorithm).pack(side=tk.LEFT, padx=5)
        tk.Button(self.button_frame, text="Run Test Cases", command=self.run_test_cases).pack(side=tk.LEFT, padx=5)
        tk.Button(self.button_frame, text="Analyze Performance", command=self.analyze_performance).pack(side=tk.LEFT, padx=5)
        tk.Button(self.button_frame, text="Reset", command=self.reset).pack(side=tk.LEFT, padx=5)

        # Add visualization button
        self.add_visualizations_to_gui()

        # Create frame for navigation buttons
        self.nav_frame = tk.Frame(self.main_frame)
        self.nav_frame.grid(row=2, column=0, columnspan=6, pady=5)
        tk.Button(self.nav_frame, text="Previous Solution", command=self.show_prev_solution).pack(side=tk.LEFT, padx=5)
        tk.Button(self.nav_frame, text="Next Solution", command=self.show_next_solution).pack(side=tk.LEFT, padx=5)

        # Create frame for input fields
        self.input_frame = tk.Frame(self.main_frame)
        self.input_frame.grid(row=3, column=0, columnspan=6, pady=5)
        tk.Label(self.input_frame, text="Custom State (Ex: 0,6,4,7,1,3,5,2):").pack(side=tk.LEFT)
        self.state_entry = tk.Entry(self.input_frame, width=30)
        self.state_entry.pack(side=tk.LEFT, padx=5)
        tk.Label(self.input_frame, text="Number of Runs:").pack(side=tk.LEFT, padx=5)
        self.runs_spinbox = tk.Spinbox(self.input_frame, from_=1, to=100, width=5, validate="key")
        self.runs_spinbox.pack(side=tk.LEFT, padx=5)
        self.runs_spinbox.delete(0, tk.END)
        self.runs_spinbox.insert(0, "5")
        tk.Button(self.input_frame, text="Run with Custom State", command=self.run_custom_state).pack(side=tk.LEFT, padx=5)

        # Create labels for status and solution info
        self.status_label = tk.Label(self.main_frame, text="Select an algorithm to solve", font=("Arial", 12))
        self.status_label.grid(row=4, column=0, columnspan=6, pady=5)
        self.solution_info = tk.Label(self.main_frame, text="No solutions available", font=("Arial", 10))
        self.solution_info.grid(row=5, column=0, columnspan=6)

        # Create scrollable text area for logging
        self.log_text = scrolledtext.ScrolledText(self.main_frame, height=10, width=60, font=("Arial", 10))
        self.log_text.grid(row=6, column=0, columnspan=6, pady=5)
        self.log_text.config(state='disabled')

        self.table_frame = None  # Placeholder for performance table
        self.viz_frame = None  # Placeholder for visualization window

        # Draw the initial chessboard
        self.draw_board()

    def draw_board(self):
        # Draw an 8x8 chessboard with alternating black and white squares
        self.canvas.delete("all")
        for row in range(self.board_size):
            for col in range(self.board_size):
                # Alternate colors for checkerboard pattern
                color = "white" if (row + col) % 2 == 0 else "black"
                x1, y1 = col * self.cell_size, row * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color)

    def place_queens(self, P):
        # Place queens on the chessboard based on the given state
        self.canvas.delete("queen")  # Remove existing queens
        self.queens = []
        for col in range(8):
            row = P[col]
            # Calculate center of the cell for queen placement
            x = col * self.cell_size + self.cell_size // 2
            y = row * self.cell_size + self.cell_size // 2
            # Use red for solutions (h=0), blue for non-solutions
            queen = self.canvas.create_text(x, y, text="♕", font=("Arial", 24),
                                           fill="red" if heuristic(P) == 0 else "blue", tags="queen")
            self.queens.append(queen)

    def update_status(self, h, algorithm, iterations, runtime):
        # Update the status label with the results of the algorithm run
        status = f"{algorithm}: Heuristic = {h}, Iterations = {iterations}, Time = {runtime:.3f}s"
        # Indicate if a solution was found or a local optimum
        status += " (Solution Found!)" if h == 0 else " (Local Optimum)"
        self.status_label.config(text=status)

    def log_message(self, message):
        # Append a message to the log text area
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)  # Scroll to the end
        self.log_text.config(state='disabled')

    def reset(self):
        # Reset the GUI to its initial state
        self.canvas.delete("queen")  # Clear queens
        self.queens = []
        self.solutions = self.solutions[-100:]  # Keep last 100 solutions
        # Update solution index, or set to -1 if no solutions
        self.current_solution_idx = min(self.current_solution_idx, len(self.solutions) - 1) if self.solutions else -1
        self.last_algorithm = None
        self.status_label.config(text="Select an algorithm to solve")
        self.solution_info.config(text="No solutions available")
        # Clear log text
        self.log_text.config(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state='disabled')
        # Destroy performance table if it exists
        if self.table_frame:
            self.table_frame.destroy()
            self.table_frame = None
        # Destroy visualization window if it exists
        if self.viz_frame:
            self.viz_frame.destroy()
            self.viz_frame = None
        self.draw_board()  # Redraw empty board

    def show_prev_solution(self):
        # Display the previous solution in the solutions list
        if not self.solutions:
            # Show warning if no solutions are available
            messagebox.showwarning("No Solutions", "Run an algorithm or test cases to find solutions.")
            self.log_message("No solutions available to cycle through.")
            return
        # Cycle to previous solution
        self.current_solution_idx = (self.current_solution_idx - 1) % len(self.solutions)
        self.display_current_solution()

    def show_next_solution(self):
        # Display the next solution in the solutions list
        if not self.solutions:
            # Show warning if no solutions are available
            messagebox.showwarning("No Solutions", "Run an algorithm or test cases to find solutions.")
            self.log_message("No solutions available to cycle through.")
            return
        # Cycle to next solution
        self.current_solution_idx = (self.current_solution_idx + 1) % len(self.solutions)
        self.display_current_solution()

    def display_current_solution(self):
        # Display the currently selected solution on the board and update info
        if not self.solutions or self.current_solution_idx < 0:
            return
        solution = self.solutions[self.current_solution_idx]
        state = solution["state"]
        h = solution["heuristic"]
        algorithm = solution["algorithm"]
        source = solution["source"]
        run = solution["run"]
        self.place_queens(state)  # Update board with queens
        # Determine if it's a perfect solution or local optimum
        solution_type = "Perfect Solution" if h == 0 else f"Local Optimum (h={h})"
        # Update solution info label
        self.solution_info.config(
            text=f"{source} | Run {run} | State {self.current_solution_idx + 1}/{len(self.solutions)} | "
                 f"{solution_type} | Algorithm: {algorithm}"
        )
        # Log the displayed solution details
        self.log_message(f"Showing state {self.current_solution_idx + 1}/{len(self.solutions)} ({source}, Run {run})")
        self.log_message(f"State: {state}, Heuristic: {h}, Algorithm: {algorithm}")

    def get_num_runs(self):
        # Retrieve the number of runs from the spinbox
        try:
            num_runs = int(self.runs_spinbox.get())
            if num_runs < 1:
                raise ValueError("Number of runs must be at least 1.")
            return num_runs
        except ValueError:
            # Show error message for invalid input
            messagebox.showerror("Invalid Input", "Please enter a valid positive integer for the number of runs.")
            return None

    def run_algorithm(self, algorithm_func, name, initial_state=None):
        # Run the specified algorithm and display results
        num_runs = self.get_num_runs()
        if num_runs is None:
            return None, None, None

        self.log_message(f"\nRunning {name} ({num_runs} runs)...")
        source = "Custom" if initial_state is not None else "Random"
        
        run_results = []
        for run in range(1, num_runs + 1):
            start_time = time.time()
            # Use initial state for first run if provided, else random
            current_state = initial_state if run == 1 and initial_state is not None else random_permutation()
            self.log_message(f"Run {run} - Initial state: {current_state}")
            
            try:
                P, h, iterations, max_states = algorithm_func(initial_state=current_state)
                runtime = time.time() - start_time
                # Store solution details
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
            # Show error if all runs failed
            messagebox.showerror("Error", f"{name} failed all runs due to invalid states.")
            return None, None, None
        
        # Select best result based on lowest heuristic
        best_result = min(run_results, key=lambda x: x[1])
        P, h, iterations, runtime, max_states = best_result
        self.current_solution_idx = len(self.solutions) - 1
        self.last_algorithm = name
        
        self.place_queens(P)
        self.update_status(h, name, iterations, runtime)
        self.display_current_solution()
        
        # Count perfect solutions
        perfect_solutions = sum(1 for _, h, _, _, _ in run_results if h == 0)
        if perfect_solutions > 0:
            messagebox.showinfo("Result", f"{name} found {perfect_solutions}/{num_runs} solutions. Navigate to see all results.")
        else:
            messagebox.showwarning("Result", f"{name} found no solutions. Best heuristic: {h}")
        
        return h, iterations, runtime

    def run_test_cases(self):
        # Run all test cases with all algorithms and save results to a file
        num_runs = self.get_num_runs()
        if num_runs is None:
            return
        
        self.reset()  # Clear current state
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
                # Log test case details
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
                        h, iterations, runtime = self.run_algorithm(algo_func, algo_name, test_case['state'])
                        if h is None:
                            continue
                        # Update source for recent solutions
                        for sol in self.solutions[-num_runs:]:
                            sol["source"] = f"Test Case: {test_case['name']}"
                        f.write(f"  {algo_name}:\n")
                        for sol in self.solutions[-num_runs:]:
                            f.write(f"    Run {sol['run']} - State: {sol['state']}, Heuristic: {sol['heuristic']}, "
                                    f"Iterations: {sol['iterations']}, Runtime: {sol['runtime']:.3f}s, "
                                    f"Max States: {sol['max_states']}, Success: {'Yes' if sol['heuristic'] == 0 else 'No'}\n")
                        self.log_message(f"  Best Result: Heuristic = {h}, Iterations = { iterations}, Time = {runtime:.3f}s")
                    except ValueError as e:
                        self.log_message(f"  Error in {algo_name}: {e}")
                        f.write(f"  {algo_name}: Error - {e}\n")
                    self.root.update()
                    time.sleep(0.5)  # Brief pause for GUI responsiveness
                f.write("-" * 50 + "\n")
        self.log_message("Test cases completed. Results saved to test_case_results.txt")
        messagebox.showinfo("Test Cases", "Test cases completed. Check log and test_case_results.txt")

    def run_custom_state(self):
        # Run algorithms with a user-provided custom state
        num_runs = self.get_num_runs()
        if num_runs is None:
            return
        
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
                h, iterations, runtime = self.run_algorithm(algo_func, algo_name, state)
                if h is None:
                    continue
                self.log_message(f"{algo_name} Best Result: Heuristic = {h}, Iterations = {iterations}, Time = {runtime:.3f}s")
                self.root.update()
                time.sleep(0.5)  # Brief pause for GUI responsiveness
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Error: {e}")
            self.log_message(f"Invalid input: {input_str}")

    def display_performance_table(self, metrics):
        # Display a table of performance metrics for all algorithms
        if self.table_frame:
            self.table_frame.destroy()
        
        self.table_frame = tk.Frame(self.main_frame)
        self.table_frame.grid(row=7, column=0, columnspan=6, pady=5)
        tree = ttk.Treeview(self.table_frame, columns=(
            "Algorithm", "Success", "Heuristic", "Iterations", "Runtime", "States"
        ), show="headings")
        # Set column headings
        tree.heading("Algorithm", text="Algorithm")
        tree.heading("Success", text="Success Rate (%)")
        tree.heading("Heuristic", text="Avg Heuristic")
        tree.heading("Iterations", text="Avg Iterations (Success)")
        tree.heading("Runtime", text="Avg Runtime (s)")
        tree.heading("States", text="Avg Max States")
        # Set column widths
        tree.column("Algorithm", width=100)
        tree.column("Success", width=100)
        tree.column("Heuristic", width=100)
        tree.column("Iterations", width=120)
        tree.column("Runtime", width=100)
        tree.column("States", width=100)
        
        # Populate table with metrics
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

    def add_visualizations_to_gui(self):
        # Add a button to trigger performance visualizations
        tk.Button(self.button_frame, text="Show Visualizations", command=self.show_visualizations).pack(side=tk.LEFT, padx=5)
        self.viz_frame = None

    def show_visualizations(self):
        # Display performance visualizations in a new window with tabs
        if not hasattr(self, 'performance_metrics') or not self.performance_metrics:
            messagebox.showwarning("No Data", "Please run performance analysis first.")
            return
        
        if self.viz_frame:
            self.viz_frame.destroy()
        
        # Create new top-level window for visualizations
        self.viz_frame = tk.Toplevel(self.root)
        self.viz_frame.title("Algorithm Performance Visualizations")
        self.viz_frame.geometry("1000x800")
        
        # Create notebook for tabbed visualizations
        notebook = ttk.Notebook(self.viz_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        success_tab = ttk.Frame(notebook)
        runtime_tab = ttk.Frame(notebook)
        complexity_tab = ttk.Frame(notebook)
        
        notebook.add(success_tab, text="Success Rate")
        notebook.add(runtime_tab, text="Runtime vs Iterations")
        notebook.add(complexity_tab, text="Time and Space Complexity")
        
        # Create charts for each tab
        self.create_success_rate_chart(success_tab)
        self.create_runtime_iterations_chart(runtime_tab)
        self.create_time_space_chart(complexity_tab)

        # Add caption for time and space chart
        caption = tk.Label(
            complexity_tab,
            text="Left y-axis (blue bars): Average runtime in seconds (time complexity). "
                 "Right y-axis (red bars): Average max states (space complexity).",
            font=("Arial", 10)
        )
        caption.pack(side=tk.BOTTOM, pady=5)

    def create_success_rate_chart(self, parent_frame):
        # Create a bar chart showing success rates for each algorithm
        metrics = self.performance_metrics
        algorithms = list(metrics.keys())
        success_rates = [
            metrics[algo]['success'] / metrics[algo]['total_runs'] * 100 
            if metrics[algo]['total_runs'] > 0 else 0 
            for algo in algorithms
        ]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(algorithms, success_rates, color=['#3498db', '#e74c3c', '#2ecc71'])
        
        # Add percentage labels above bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1,
                f'{height:.1f}%',
                ha='center',
                fontweight='bold'
            )
        
        ax.set_ylim(0, 105)
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Algorithm Success Rates')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        canvas = FigureCanvasTkAgg(fig, parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_runtime_iterations_chart(self, parent_frame):
        # Create a dual-axis bar chart comparing runtime and iterations
        metrics = self.performance_metrics
        algorithms = list(metrics.keys())
        avg_runtimes = [
            metrics[algo]['avg_time'] 
            if metrics[algo]['total_runs'] > 0 else 0 
            for algo in algorithms
        ]
        
        avg_iterations = [
            metrics[algo]['successful_iterations'] / metrics[algo]['successful_runs'] 
            if metrics[algo]['successful_runs'] > 0 else 0 
            for algo in algorithms
        ]
        
        fig, ax1 = plt.subplots(figsize=(8, 6))
        
        x = np.arange(len(algorithms))
        width = 0.35
        runtime_bars = ax1.bar(x - width/2, avg_runtimes, width, label='Avg Runtime (s)', color='#3498db')
        ax1.set_xlabel('Algorithm')
        ax1.set_ylabel('Runtime (seconds)', color='#3498db')
        ax1.tick_params(axis='y', labelcolor='#3498db')
        
        ax2 = ax1.twinx()
        iter_bars = ax2.bar(x + width/2, avg_iterations, width, label='Avg Iterations (Success)', color='#e74c3c')
        ax2.set_ylabel('Iterations', color='#e74c3c')
        ax2.tick_params(axis='y', labelcolor='#e74c3c')
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(algorithms)
        ax1.set_title('Runtime vs Iterations for Successful Runs')
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_time_space_chart(self, parent_frame):
        # Create a dual-axis bar chart comparing time and space complexity
        metrics = self.performance_metrics
        algorithms = list(metrics.keys())
        
        avg_runtimes = [
            metrics[algo]['avg_time'] if metrics[algo]['total_runs'] > 0 else 0 
            for algo in algorithms
        ]
        avg_max_states = [
            metrics[algo]['avg_max_states'] if metrics[algo]['total_runs'] > 0 else 0 
            for algo in algorithms
        ]
        
        fig, ax1 = plt.subplots(figsize=(8, 6))
        
        x = np.arange(len(algorithms))
        width = 0.35
        
        runtime_bars = ax1.bar(x - width/2, avg_runtimes, width, label='Avg Runtime (s)', color='#3498db')
        ax1.set_xlabel('Algorithm')
        ax1.set_ylabel('Runtime (seconds)', color='#3498db')
        ax1.tick_params(axis='y', labelcolor='#3498db')

        ax2 = ax1.twinx()
        states_bars = ax2.bar(x + width/2, avg_max_states, width, label='Avg Max States', color='#e74c3c')
        ax2.set_ylabel('Max States', color='#e74c3c')
        ax2.tick_params(axis='y', labelcolor='#e74c3c')

        ax1.set_xticks(x)
        ax1.set_xticklabels(algorithms)
        ax1.set_title('Time and Space Complexity')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Add value labels above bars
        for bars in [runtime_bars, states_bars]:
            for bar in bars:
                height = bar.get_height()
                ax = ax1 if bars == runtime_bars else ax2
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.01 * max(max(avg_runtimes), max(avg_max_states)),
                    f'{height:.2f}' if bars == runtime_bars else f'{int(height)}',
                    ha='center',
                    fontweight='bold'
                )
        
        ax1.spines['top'].set_visible(False)
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def analyze_performance(self):
        # Analyze performance across test cases and display results
        num_runs = self.get_num_runs()
        if num_runs is None:
            return
        
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
            
            # Run each algorithm on each test case
            for test_case in TEST_CASES:
                for algo_func, algo_name in [
                    (hill_climbing, "Hill-Climbing"),
                    (simulated_annealing, "Simulated Annealing"),
                    (genetic_algorithm, "Genetic Algorithm")
                ]:
                    for run in range(num_runs):
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
            
            self.performance_metrics = metrics
            
            # Log and save performance summary
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
            self.show_visualizations()

    def run_hill_climbing(self):
        # Run the hill-climbing algorithm
        self.run_algorithm(hill_climbing, "Hill-Climbing")

    def run_simulated_annealing(self):
        # Run the simulated annealing algorithm
        self.run_algorithm(simulated_annealing, "Simulated Annealing")

    def run_genetic_algorithm(self):
        # Run the genetic algorithm
        self.run_algorithm(genetic_algorithm, "Genetic Algorithm")

if __name__ == "__main__":
    # Main entry point to launch the GUI
    root = tk.Tk()
    app = EightQueensGUI(root)
    root.mainloop()
