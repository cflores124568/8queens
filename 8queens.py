import random
import math
import time
import tkinter as tk
from tkinter import messagebox, scrolledtext

#8 Queens Algorithms
def heuristic(P):
    #Compute the number of diagonal conflicts in a state P
    count = 0
    for i in range(8):
        for j in range(i + 1, 8):
            if abs(P[i] - P[j]) == abs(i - j):
                count += 1
    return count

def random_permutation():
    #Generate a random permutation of [0, 1, ..., 7]
    P = list(range(8))
    random.shuffle(P)
    return P

def hill_climbing(initial_state=None):
    P = initial_state if initial_state is not None else random_permutation()
    current_h = heuristic(P)
    iterations = 0
    improved = True
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
    return P, current_h, iterations

def simulated_annealing(initial_state=None):
    P = initial_state if initial_state is not None else random_permutation()
    T = 1000
    T_min = 1
    alpha = 0.99
    iterations = 0
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
            return P, 0, iterations
    return P, heuristic(P), iterations

def genetic_algorithm(initial_state=None, pop_size=100, generations=1000):
    def fitness(P):
        return 1 / (1 + heuristic(P))
    
    initial_P = initial_state if initial_state is not None else random_permutation()
    population = [initial_P] + [random_permutation() for _ in range(pop_size - 1)]
    iterations = 0
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
            return best, 0, iterations
    best = max(population, key=fitness)
    return best, heuristic(best), iterations

#Test Cases
TEST_CASES = [
    {"name": "Easy", "state": [0, 6, 4, 7, 1, 3, 5, 2], "heuristic": 1},
    {"name": "Medium", "state": [0, 4, 7, 5, 2, 6, 1, 3], "heuristic": 4},
    {"name": "Hard", "state": [0, 1, 2, 3, 4, 5, 6, 7], "heuristic": 7},
    {"name": "Edge", "state": [1, 7, 5, 0, 2, 4, 6, 3], "heuristic": 0}
]

#GUI Implementation
class EightQueensGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("8-Queens Problem Solver")
        self.cell_size = 50
        self.board_size = 8
        self.canvas_size = self.board_size * self.cell_size
        self.queens = []
        self.solutions = []
        self.current_solution_idx = -1  # Start at -1 so first click shows index 0
        self.last_algorithm = None

        # Create main frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(padx=10, pady=10)

        # Create canvas for chessboard
        self.canvas = tk.Canvas(self.main_frame, width=self.canvas_size, height=self.canvas_size)
        self.canvas.grid(row=0, column=0, columnspan=6)

        # Create buttons for algorithms and actions
        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.grid(row=1, column=0, columnspan=6, pady=5)
        tk.Button(self.button_frame, text="Run Hill-Climbing", command=self.run_hill_climbing).pack(side=tk.LEFT, padx=5)
        tk.Button(self.button_frame, text="Run Simulated Annealing", command=self.run_simulated_annealing).pack(side=tk.LEFT, padx=5)
        tk.Button(self.button_frame, text="Run Genetic Algorithm", command=self.run_genetic_algorithm).pack(side=tk.LEFT, padx=5)
        tk.Button(self.button_frame, text="Run Test Cases", command=self.run_test_cases).pack(side=tk.LEFT, padx=5)
        tk.Button(self.button_frame, text="Reset", command=self.reset).pack(side=tk.LEFT, padx=5)
        
        # Add navigation buttons
        self.nav_frame = tk.Frame(self.main_frame)
        self.nav_frame.grid(row=2, column=0, columnspan=6, pady=5)
        tk.Button(self.nav_frame, text="Previous Solution", command=self.show_prev_solution).pack(side=tk.LEFT, padx=5)
        tk.Button(self.nav_frame, text="Next Solution", command=self.show_next_solution).pack(side=tk.LEFT, padx=5)

        # Create status label
        self.status_label = tk.Label(self.main_frame, text="Select an algorithm to solve", font=("Arial", 12))
        self.status_label.grid(row=3, column=0, columnspan=6, pady=5)

        # Create solution info label
        self.solution_info = tk.Label(self.main_frame, text="No solutions available", font=("Arial", 10))
        self.solution_info.grid(row=4, column=0, columnspan=6)

        # Create text area for logs
        self.log_text = scrolledtext.ScrolledText(self.main_frame, height=10, width=60, font=("Arial", 10))
        self.log_text.grid(row=5, column=0, columnspan=6, pady=5)
        self.log_text.config(state='disabled')

        # Draw initial chessboard
        self.draw_board()

    def draw_board(self):
        #Draw an 8x8 chessboard with alternating colors
        self.canvas.delete("all")
        for row in range(self.board_size):
            for col in range(self.board_size):
                color = "white" if (row + col) % 2 == 0 else "black"
                x1, y1 = col * self.cell_size, row * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color)

    def place_queens(self, P):
        #Place queens on the board based on permutation P
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
        #Update the status label with heuristic value and solution status
        status = f"{algorithm}: Heuristic = {h}, Iterations = {iterations}, Time = {runtime:.3f}s"
        if h == 0:
            status += " (Solution Found!)"
        else:
            status += " (Local Optimum)"
        self.status_label.config(text=status)

    def log_message(self, message):
        #Add a message to the log text area
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def reset(self):
        #Reset the board and status
        self.canvas.delete("queen")
        self.queens = []
        self.solutions = []
        self.current_solution_idx = -1
        self.last_algorithm = None
        self.status_label.config(text="Select an algorithm to solve")
        self.solution_info.config(text="No solutions available")
        self.log_text.config(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state='disabled')
        self.draw_board()

    def show_prev_solution(self):
        #Show the previous solution in the solutions list
        if not self.solutions:
            messagebox.showwarning("No Solutions", "Run an algorithm or test cases to find solutions.")
            self.log_message("No solutions available to cycle through.")
            return
        
        self.current_solution_idx = (self.current_solution_idx - 1) % len(self.solutions)
        self.display_current_solution()

    def show_next_solution(self):
        #Show the next solution in the solutions list
        if not self.solutions:
            messagebox.showwarning("No Solutions", "Run an algorithm or test cases to find solutions.")
            self.log_message("No solutions available to cycle through.")
            return
        
        self.current_solution_idx = (self.current_solution_idx + 1) % len(self.solutions)
        self.display_current_solution()

    def display_current_solution(self):
        #Display the current solution from the solutions list
        if not self.solutions or self.current_solution_idx < 0:
            return
    
        current_solution = self.solutions[self.current_solution_idx]
        current_h = heuristic(current_solution)
    
        self.place_queens(current_solution)
    
        # Update solution info
        solution_type = "Perfect Solution" if current_h == 0 else f"Local Optimum (h={current_h})"
        self.solution_info.config(
            text=f"State {self.current_solution_idx + 1}/{len(self.solutions)} | {solution_type} | Algorithm: {self.last_algorithm}"
        )
    
        self.log_message(f"Showing state {self.current_solution_idx + 1}/{len(self.solutions)}")
        self.log_message(f"State: {current_solution}, Heuristic: {current_h}")

    def run_algorithm(self, algorithm_func, name, initial_state=None):
        #Run the specified algorithm and update GUI
        start_time = time.time()
        self.log_message(f"\nRunning {name}...")
        if initial_state is not None:
            self.log_message(f"Initial state: {initial_state}")
        else:
            initial_state = random_permutation()
            self.log_message(f"Initial state: {initial_state}")
    
        P, h, iterations = algorithm_func(initial_state=initial_state)
        runtime = time.time() - start_time
        self.last_algorithm = name
    
        # Store this run's result (allow duplicates from different runs)
        self.solutions.append(P)
        self.current_solution_idx = len(self.solutions) - 1
    
        self.place_queens(P)
        self.update_status(h, name, iterations, runtime)
        self.log_message(f"Final state: {P} (Heuristic: {h})")
    
        # Update solution navigation info
        self.display_current_solution()
    
        if h == 0:
            messagebox.showinfo("Result", f"{name} found a solution!")
        else:
            messagebox.showwarning("Result", f"{name} stuck at local optimum (h={h}).")
        return h, iterations, runtime

    def run_test_cases(self):
        #Run all test cases for each algorithm and log results
        self.reset()
        self.log_message("Running test cases...")
        algorithms = [
            (hill_climbing, "Hill-Climbing"),
            (simulated_annealing, "Simulated Annealing"),
            (genetic_algorithm, "Genetic Algorithm")
        ]
        for test_case in TEST_CASES:
            self.log_message(f"\nTest Case: {test_case['name']} (Initial Heuristic = {test_case['heuristic']})")
            self.log_message(f"Initial state: {test_case['state']}")
            for algo_func, algo_name in algorithms:
                self.log_message(f"  Running {algo_name}...")
                h, iterations, runtime = self.run_algorithm(algo_func, algo_name, test_case['state'])
                self.log_message(f"  Result: Heuristic = {h}, Iterations = {iterations}, Time = {runtime:.3f}s")
                self.root.update()  # Update GUI to show progress
                time.sleep(0.5)  # Brief pause to make progress visible

    def run_hill_climbing(self):
        #Run Hill-climbing and update GUI
        self.run_algorithm(hill_climbing, "Hill-Climbing")

    def run_simulated_annealing(self):
        #Run Simulated Annealing and update GUI
        self.run_algorithm(simulated_annealing, "Simulated Annealing")

    def run_genetic_algorithm(self):
        #Run Genetic Algorithm and update GUI
        self.run_algorithm(genetic_algorithm, "Genetic Algorithm")

#Main Program 
if __name__ == "__main__":
    root = tk.Tk()
    app = EightQueensGUI(root)
    root.mainloop()