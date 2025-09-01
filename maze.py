import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random
from collections import deque
import heapq
import time

# --- CSS Glow Styling ---
st.markdown("""
<style>
html, body, [class*="css"]  {
    font-family: 'Segoe UI', sans-serif;
    background-color: #0d0d0d;
    color: #00ccff;
}
h1, h2, h3, h4 {
    font-size: 32px !important;
    text-align: center;
    color: #00ccff !important;
    text-shadow: 0 0 10px #00ccff, 0 0 20px #00ccff;
}
div[data-testid="metric-container"] {
    background-color: #111 !important;
    padding: 10px;
    border-radius: 10px;
    box-shadow: 0 0 10px #00ccff;
    margin-bottom: 15px;
}
div[data-testid="stMetricValue"] {
    font-size: 26px !important;
    color: #00ffea !important;
    text-shadow: 0 0 8px #00e6ff;
}
button {
    background-color: #00ccff !important;
    color: black !important;
    font-weight: bold;
    border-radius: 10px;
    box-shadow: 0 0 10px #00ccff;
}
button:hover {
    background-color: #0099cc !important;
    box-shadow: 0 0 20px #00ccff;
}
</style>
""", unsafe_allow_html=True)

# --- Maze Generator ---
def generate_maze(rows, cols):
    rows, cols = (rows if rows % 2 == 1 else rows - 1, cols if cols % 2 == 1 else cols - 1)
    maze = [[1] * cols for _ in range(rows)]
    def carve(x, y):
        dirs = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        random.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if 1 <= nx < rows - 1 and 1 <= ny < cols - 1 and maze[nx][ny] == 1:
                maze[nx][ny] = 0
                maze[x + dx // 2][y + dy // 2] = 0
                carve(nx, ny)
    maze[1][1] = 0
    carve(1, 1)
    for _ in range(int(rows * cols * 0.02)):
        r, c = random.randint(1, rows - 2), random.randint(1, cols - 2)
        maze[r][c] = 0
    empty_cells = [(r, c) for r in range(rows) for c in range(cols) if maze[r][c] == 0]
    start, goal = random.sample(empty_cells, 2)
    while abs(start[0] - goal[0]) + abs(start[1] - goal[1]) < (rows + cols) // 3:
        goal = random.choice(empty_cells)
    return maze, start, goal

# --- Path Reconstruction ---
def reconstruct_path(parent, start, goal):
    path = []
    node = goal
    while node in parent:
        path.append(node)
        node = parent[node]
    if node == start:
        path.append(start)
        path.reverse()
        return path
    return []

# --- Algorithms ---
def bfs(maze, start, goal):
    queue = deque([start])
    visited = {start}
    parent = {}
    while queue:
        x, y = queue.popleft()
        if (x, y) == goal: break
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < len(maze) and 0 <= ny < len(maze[0]) and maze[nx][ny] == 0 and (nx, ny) not in visited:
                visited.add((nx, ny))
                parent[(nx, ny)] = (x, y)
                queue.append((nx, ny))
    return reconstruct_path(parent, start, goal)

def dfs(maze, start, goal):
    stack = [start]
    visited = {start}
    parent = {}
    while stack:
        x, y = stack.pop()
        if (x, y) == goal: break
        dirs = [(-1,0),(1,0),(0,-1),(0,1)]
        random.shuffle(dirs)
        for dx, dy in dirs:
            nx, ny = x+dx, y+dy
            if 0 <= nx < len(maze) and 0 <= ny < len(maze[0]) and maze[nx][ny] == 0 and (nx, ny) not in visited:
                visited.add((nx, ny))
                parent[(nx, ny)] = (x, y)
                stack.append((nx, ny))
    return reconstruct_path(parent, start, goal)

def astar(maze, start, goal):
    def heuristic(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start))
    parent = {}
    g_score = {start: 0}
    while open_set:
        _, cost, current = heapq.heappop(open_set)
        if current == goal: break
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nx, ny = current[0]+dx, current[1]+dy
            if 0 <= nx < len(maze) and 0 <= ny < len(maze[0]) and maze[nx][ny] == 0:
                tentative_g = cost + 1
                if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = tentative_g
                    heapq.heappush(open_set, (tentative_g + heuristic((nx, ny), goal), tentative_g, (nx, ny)))
                    parent[(nx, ny)] = current
    return reconstruct_path(parent, start, goal)

# --- Draw Maze ---
def draw_maze(maze, path=None, start=None, goal=None, path_color=2, title=""):
    rows, cols = len(maze), len(maze[0])
    data = [[1 if maze[r][c] == 1 else 0 for c in range(cols)] for r in range(rows)]
    if path:
        for r, c in path:
            data[r][c] = path_color
    if start:
        data[start[0]][start[1]] = 3
    if goal:
        data[goal[0]][goal[1]] = 4
    colors = ["#ffffff", "#000000", "#00ffcc", "#0066ff", "#ff3366", "#ff00ff", "#ffff00"]
    cmap = mcolors.ListedColormap(colors)
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(data, cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=22, color="#00ccff")
    return fig

# --- UI Layout ---
st.markdown("<h1>🌌 Maze Solving Comparison</h1>", unsafe_allow_html=True)
rows = st.slider("Rows", 11, 101, 51, step=2)
cols = st.slider("Columns", 11, 101, 51, step=2)

if st.button("⚡ Generate & Solve Maze"):
    maze, start, goal = generate_maze(rows, cols)

    for name, algo, color in [
        ("BFS", bfs, 2),
        ("DFS", dfs, 5),
        ("A*", astar, 6)
    ]:
        st.markdown(f"<h2>{name} Algorithm</h2>", unsafe_allow_html=True)
        start_time = time.time()
        path = algo(maze, start, goal)
        duration = time.time() - start_time
        fig = draw_maze(maze, path=path, start=start, goal=goal, path_color=color, title=f"{name} Path")
        st.pyplot(fig)
        if path:
            st.success(f"✅ Found in {duration:.3f} seconds")
            st.metric("📏 Path Length", len(path))
        else:
            st.error("❌ No path found")

with st.expander("🎨 Color Legend"):
    st.markdown("""
    - ⚪️ **White** – Open path  
    - ⚫️ **Black** – Wall  
    - 🟦 **Blue** – Start point  
    - 🔴 **Red** – Goal point  
    - 🟢 **Neon Green** – BFS Path  
    - 💖 **Pink** – DFS Path  
    - 💛 **Yellow** – A* Path  
    """)
