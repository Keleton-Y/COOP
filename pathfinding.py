import heapq
import math
import random
import numpy as np
from collections import defaultdict

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

class AStarNode:
    def __init__(self, position, time_step, cost, heuristic, parent=None):
        self.position = position
        self.time_step = time_step
        self.cost = cost  
        self.heuristic = heuristic  
        self.parent = parent

    def __lt__(self, other):
        return (self.cost + self.heuristic * 2) < (other.cost + other.heuristic * 2)


class CollisionIndex:
    def __init__(self, paths=None):
        self.collision_index = defaultdict(dict)
        self.move_index = defaultdict(set)
        self.min_time_step = 0  

        if paths:
            self.update(paths)

    def update(self, paths):
        for path in paths:
            for i in range(1, len(path)):
                prev_position, prev_time_step = path[i - 1]
                current_position, current_time_step = path[i]
                self.collision_index[prev_time_step][prev_position] = True
                self.move_index[prev_time_step].add((prev_position, current_position))
                self.min_time_step = min(self.min_time_step, prev_time_step)

    def is_collision(self, current_position, previous_position, time_step):
        if (current_position, previous_position) in self.move_index[time_step - 1]:
            return True
        if self.collision_index[time_step].get(current_position, False):
            return True
        return False

    def delete(self, max_time_step):
        time_steps_to_delete = [t for t in range(self.min_time_step, max_time_step)]
        for t in time_steps_to_delete:
            if t in self.collision_index:
                del self.collision_index[t]
            if t in self.move_index:
                del self.move_index[t]

        self.min_time_step = max_time_step


def manhattan_distance(start, goal):
    return abs(start[0] - goal[0]) + abs(start[1] - goal[1])


def reconstruct_path(node):
    path = []
    while node:
        path.append((node.position, node.time_step))
        node = node.parent
    return path[::-1]  


def a_star(grid, start, goal, collision_index=None, start_time=0):
    open_set = []
    nodes_expanded = 0
    index_dict = set()
    start_node = AStarNode(position=start, time_step=start_time, cost=0, heuristic=manhattan_distance(start, goal))
    heapq.heappush(open_set, start_node)
    index_dict.add((start, start_time))
    closed_set = set()
    change_flag = False
    if grid[start[0]][start[1]] == 1:
        grid[start[0]][start[1]] = 0
        change_flag = True
    
    while open_set:
        current_node = heapq.heappop(open_set)
        current_position, current_time_step = current_node.position, current_node.time_step
        nodes_expanded += 1

        if current_position == goal:
            return reconstruct_path(current_node)

        if (current_position, current_time_step) in closed_set:
            continue
        closed_set.add((current_position, current_time_step))

        for direction in DIRECTIONS + [(0, 0)]:  
            next_position = (current_position[0] + direction[0], current_position[1] + direction[1])
            next_time_step = current_time_step + 1

            if not (0 <= next_position[0] < len(grid) and 0 <= next_position[1] < len(grid[0])):
                continue

            if grid[next_position[0]][next_position[1]] == 1:
                continue

            if collision_index and collision_index.is_collision(next_position, current_position, next_time_step):
                continue

            if (next_position, next_time_step) in index_dict:
                continue

            index_dict.add((next_position, next_time_step))
            new_cost = current_node.cost + 1
            heuristic = manhattan_distance(next_position, goal)
            next_node = AStarNode(position=next_position, time_step=next_time_step, cost=new_cost, heuristic=heuristic,
                                  parent=current_node)

            heapq.heappush(open_set, next_node)

    if change_flag:
        grid[start[0]][start[1]] = 1
    return []



def a_star_progressive(grid, start, goal, collision_index=None, start_time=0, d=0):
    open_set = []
    nodes_expanded = 0
    start_node = AStarNode(position=start, time_step=start_time, cost=0, heuristic=manhattan_distance(start, goal))
    heapq.heappush(open_set, start_node)
    closed_set = set()
    warehouse_size = len(grid) * len(grid[0])  
    rect_area = (abs(goal[0] - start[0]) + 1) * (abs(goal[1] - start[1]) + 1)  
    node_limit = rect_area + math.sqrt(warehouse_size)  

    min_x = min(start[0], goal[0]) - 4
    max_x = max(start[0], goal[0]) + 4
    min_y = min(start[1], goal[1]) - 4
    max_y = max(start[1], goal[1]) + 4

    min_x = max(0, min_x)
    max_x = min(len(grid) - 1, max_x)
    min_y = max(0, min_y)
    max_y = min(len(grid[0]) - 1, max_y)

    while open_set:
        current_node = heapq.heappop(open_set)
        current_position, current_time_step = current_node.position, current_node.time_step

        nodes_expanded += 1

        if nodes_expanded > node_limit:
            return []

        distance_to_goal = manhattan_distance(current_position, goal)

        if distance_to_goal <= d:
            return reconstruct_path(current_node)

        if (current_position, current_time_step) in closed_set:
            continue
        closed_set.add((current_position, current_time_step))

        for direction in DIRECTIONS + [(0, 0)]:  
            next_position = (current_position[0] + direction[0], current_position[1] + direction[1])
            next_time_step = current_time_step + 1

            if not (min_x <= next_position[0] <= max_x and min_y <= next_position[1] <= max_y):
                continue

            if not (0 <= next_position[0] < len(grid) and 0 <= next_position[1] < len(grid[0])):
                continue

            if grid[next_position[0]][next_position[1]] == 1:
                continue

            if collision_index and collision_index.is_collision(next_position, current_position, next_time_step):
                continue

            new_cost = current_node.cost + 1
            heuristic = manhattan_distance(next_position, goal)
            next_node = AStarNode(position=next_position, time_step=next_time_step, cost=new_cost, heuristic=heuristic,
                                  parent=current_node)

            heapq.heappush(open_set, next_node)
    return []


def calculate_d_max(starts, goals):
    distances = [manhattan_distance(start, goal) for start, goal in zip(starts, goals)]
    return max(distances)


def calculate_next_d(d):
    new_d = math.floor(d / math.sqrt(2))
    return new_d if new_d > 4 else 0


def generate_random_priority_sequence(num_robots):
    return random.sample(range(num_robots), num_robots)


def mapf(grid, robot_starts, robot_goals, current_time, collision_index):
    num_robots = len(robot_starts)
    paths = [[] for _ in range(num_robots)]  
    current_positions = robot_starts[:]  
    current_times = [current_time] * num_robots  

    d_max = calculate_d_max(robot_starts, robot_goals)
    d = math.floor(d_max / math.sqrt(2))

    while d > 0:
        priority_sequence = generate_random_priority_sequence(num_robots)

        for robot_index in priority_sequence:
            if current_positions[robot_index] == robot_goals[robot_index]:
                continue

            start = current_positions[robot_index]
            goal = robot_goals[robot_index]
            current_time = current_times[robot_index]

            partial_path = a_star_progressive(grid, start, goal, collision_index, start_time=current_time, d=d)

            if partial_path:
                current_positions[robot_index], last_time = partial_path[-1]
                current_times[robot_index] = last_time

                paths[robot_index].extend(partial_path)
                collision_index.update([partial_path])
        d = calculate_next_d(d)

    priority_sequence = generate_random_priority_sequence(num_robots)
    for robot_index in priority_sequence:
        if current_positions[robot_index] != robot_goals[robot_index]:
            start = current_positions[robot_index]
            goal = robot_goals[robot_index]
            current_time = current_times[robot_index]

            final_path = a_star(grid, start, goal, collision_index, start_time=current_time)
            if final_path:
                paths[robot_index].extend(final_path)
                collision_index.update([final_path])
    return paths


def generate_random_positions(grid, num_robots):
    positions = set()
    robots = []
    size = len(grid)

    while len(robots) < num_robots * 2:
        x = random.randint(0, size - 1)
        y = random.randint(0, size - 1)
        if grid[x][y] == 0 and (x, y) not in positions:  
            positions.add((x, y))
            robots.append((x, y))

    robot_starts = robots[:num_robots]
    robot_goals = robots[num_robots:]

    return robot_starts, robot_goals


def online_task_planning(grid, start, goal, collision_index, current_time_step):
    path = a_star(grid, start, goal, collision_index, start_time=current_time_step)
    if path:
        collision_index.update([path])  
        return path
    else:
        return []


def plan_paths(grid, robot_starts, robot_goals):
    paths = []
    collision_index = CollisionIndex()  

    for i in range(len(robot_starts)):
        start = robot_starts[i]
        goal = robot_goals[i]
        path = a_star(grid, start, goal, collision_index)
        if path:
            paths.append(path)
            collision_index.update([path])  
        else:
            paths.append([])
            continue

    return paths


def a_star_segment(grid, start, goal, target_distance, collision_index=None, start_time=0):
    open_set = []
    nodes_expanded = 0
    index_dict = set()

    start_node = AStarNode(position=start, time_step=start_time, cost=0, heuristic=manhattan_distance(start, goal))
    heapq.heappush(open_set, start_node)
    index_dict.add((start, start_time))
    closed_set = set()

    while open_set:
        current_node = heapq.heappop(open_set)
        current_position, current_time_step = current_node.position, current_node.time_step
        nodes_expanded += 1

        if manhattan_distance(current_position, goal) <= target_distance:
            return reconstruct_path(current_node), current_position, current_time_step

        if (current_position, current_time_step) in closed_set:
            continue
        closed_set.add((current_position, current_time_step))

        for direction in DIRECTIONS + [(0, 0)]:
            next_position = (current_position[0] + direction[0], current_position[1] + direction[1])
            next_time_step = current_time_step + 1

            if not (0 <= next_position[0] < len(grid) and 0 <= next_position[1] < len(grid[0])):
                continue
            if grid[next_position[0]][next_position[1]] == 1:
                continue
            if collision_index and collision_index.is_collision(next_position, current_position, next_time_step):
                continue
            if (next_position, next_time_step) in index_dict:
                continue

            index_dict.add((next_position, next_time_step))
            new_cost = current_node.cost + 1
            heuristic = manhattan_distance(next_position, goal)
            next_node = AStarNode(position=next_position, time_step=next_time_step, cost=new_cost, heuristic=heuristic, parent=current_node)
            heapq.heappush(open_set, next_node)

    return [], start, start_time  


def progressive_a_star(grid, start, goal, collision_index=None, start_time=0):
    D = manhattan_distance(start, goal)
    relay_distances = [D / 4, D / 2]  
    total_path = []
    current_start = start
    current_time = start_time

    for target_distance in relay_distances + [0]:  
        path_segment, current_start, current_time = a_star_segment(
            grid, current_start, goal, target_distance, collision_index, current_time
        )
        if not path_segment:
            return []
        total_path.extend(path_segment[:-1])  

    total_path.append((goal, current_time))  
    collision_index.update([total_path])
    return total_path
