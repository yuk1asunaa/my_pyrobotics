import heapq
import math
import matplotlib.pyplot as plt


def heuristic(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def a_star(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    directions = [
        (0, 1),
        (0, -1),
        (1, 0),
        (-1, 0),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    ]

    open_heap = []
    heapq.heappush(open_heap, (0, start))

    came_from = {}
    g_score = {start: 0.0}
    f_score = {start: heuristic(start, goal)}
    closed = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current in closed:
            continue
        closed.add(current)

        if current == goal:
            return reconstruct_path(came_from, current)

        cx, cy = current
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy

            if not (0 <= nx < rows and 0 <= ny < cols):
                continue
            if grid[nx][ny] != 0:
                continue

            neighbor = (nx, ny)
            step_cost = math.hypot(dx, dy)
            tentative_g = g_score[current] + step_cost

            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                score = tentative_g + heuristic(neighbor, goal)
                f_score[neighbor] = score
                heapq.heappush(open_heap, (score, neighbor))

    return None


def visualize_with_matplotlib(grid, path, start, goal):
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(grid, cmap="Greys", origin="upper")

    if path:
        xs = [p[1] for p in path]
        ys = [p[0] for p in path]
        ax.plot(xs, ys, "b-", linewidth=2, label="Path")

    ax.scatter(start[1], start[0], c="green", s=100, marker="o", label="Start")
    ax.scatter(goal[1], goal[0], c="red", s=100, marker="x", label="Goal")

    ax.set_title("A* Path Planning")
    ax.set_xticks(range(len(grid[0])))
    ax.set_yticks(range(len(grid)))
    ax.grid(color="lightgray", linestyle="-", linewidth=0.5)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    grid = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]

    start = (0, 0)
    goal = (6, 6)

    path = a_star(grid, start, goal)

    if path:
        print("找到路径：", path)
    else:
        print("未找到路径。")

    visualize_with_matplotlib(grid, path, start, goal)
                