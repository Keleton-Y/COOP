
import copy
import tracemalloc
import time
from enum import Enum
from math import inf

from entities import *
from pathfinding import *
from rack_matching import (greedy_select_racks, calculate_processing_time,
                           build_weighted_table, calculate_weighted_shelves_from_table,
                           WeightedRackSearchHNSW, greedy_select_racks_top_k)
import random

import numpy as np
import matplotlib.pyplot as plt
from dqn_brain import *


class BundlingAlgorithm(Enum):
    DISABLE = "[Disable]"
    REVERSE = "[Reverse]"
    HYBRID = "[Hybrid_index]"


class MatchingAlgorithm(Enum):
    GREEDY = "[Greedy]"
    GREEDY_TOP_K = "[GreedyTopK]"
    HNSW_TOP_K = "[HnswTopK]"
    SIMPLE = "[Simple]"
    DQN = "[DQN]"


class RackGroupAlgorithm(Enum):
    RANDOM = "[Random]"
    GREEDY = "[Greedy]"
    DRL = "[DRL]"


class PathFindingAlgorithm(Enum):
    A_STAR = "[AStar]"
    P3F = "[P3F]"


class Warehouse:
    def __init__(self, layout_file_path, order_file_path, robot_num, workstation_num, algorithm=None, print_event=0,
                 dqn_agent=None):
        self.print_event = print_event
        if algorithm is None:
            algorithm = [BundlingAlgorithm.REVERSE, MatchingAlgorithm.GREEDY_TOP_K,
                         RackGroupAlgorithm.GREEDY, PathFindingAlgorithm.A_STAR]
        if not (isinstance(algorithm[0], BundlingAlgorithm) and
                isinstance(algorithm[1], MatchingAlgorithm) and
                isinstance(algorithm[2], RackGroupAlgorithm) and
                isinstance(algorithm[3], PathFindingAlgorithm)):
            raise ValueError("algorithm 参数的各项必须是相应的枚举类成员")

        self.picking_time = 0.75
        self.finished_orders = 0
        self.finished_orders_in_steps = []
        self.item_types = set()

        self.layout = np.zeros((0, 0))
        self.racks = []
        self.load_layout_and_racks(layout_file_path)  
        self.clear_roads = np.zeros_like(self.layout)  
        self.order_pool = []  
        self.read_orders_from_file(order_file_path, 20)  

        self.total_orders = len(self.order_pool)
        self.robots = []
        self.workstations = []
        self.initialize_robots(robot_num)  
        self.initialize_workstations(workstation_num)  

        self.robots_map = {robot.id: robot for robot in self.robots}  
        self.racks_map = {rack.id: rack for rack in self.racks}  
        self.workstations_map = {ws.id: ws for ws in self.workstations}  
        self.available_orders = []  
        self.simple_orders = []  
        self.complex_orders = []  
        self.current_time_step = 0  

        self.weighted_table = build_weighted_table(random.sample(self.order_pool, min(50000, len(self.order_pool))))
        self.weighted_racks = calculate_weighted_shelves_from_table(self.racks, self.weighted_table)
        
        self.mips_index = WeightedRackSearchHNSW(self.weighted_racks, self.racks, ef=200, M=16)
        
        self.order_index = OrderIndex()
        self.complex_order_index = None

        self.collision_index = CollisionIndex()  

        self.status_event_table = EventTable(self)  
        self.order_event_table = EventTable(self)  
        self.path_planning_event_table = EventTable(self)  

        self.initialize_order_event_table(20)

        self.matching_total_time = 0
        self.pathfinding_total_time = 0
        self.total_deliveries = 0
        self.total_path_length = 0

        self.bundling_method = algorithm[0]
        self.matching_method = algorithm[1]
        self.select_method = algorithm[2]
        self.pathfind_method = algorithm[3]

        self.dqn_agent = None
        self.last_state = None
        self.last_action = None
        
        if self.matching_method == MatchingAlgorithm.DQN:
            self.dqn_agent = dqn_agent

    def run(self):
        start_time = time.time()  

        while len(self.order_pool) or len(self.available_orders) or self.status_event_table.not_empty() \
                or self.order_event_table.not_empty() or self.path_planning_event_table.not_empty():
            self.increment_time()
            if not self.print_event and self.current_time_step % 100 == 0:
                print(
                    f"\r当前时间步: {self.current_time_step} | 完成进度: {self.finished_orders / self.total_orders * 100:.2f}%"
                    , end="")
            if self.current_time_step % 10 == 0:
                self.finished_orders_in_steps.append(self.finished_orders)
            if self.finished_orders == self.total_orders:
                break
            
        end_time = time.time()  
        elapsed_time = end_time - start_time  
        if not self.print_event:
            print(
                f"\r当前时间步: {self.current_time_step} | 完成进度: {self.finished_orders / self.total_orders * 100:.2f}%"
                , end="")
        print(f"\n运行结束，总时间步: {self.current_time_step}")
        print(f"运行时长: {elapsed_time:.2f} 秒")

    def increment_time(self):
        self.current_time_step += 1
        
        self.status_event_table.execute_events(self.current_time_step)

        self.order_event_table.execute_events(self.current_time_step)

        self.assign_orders_to_workstations()

        start_time = time.time()
        self.path_planning_event_table.execute_events(self.current_time_step)
        end_time = time.time()
        self.pathfinding_total_time += (end_time - start_time)

        self.status_event_table.delete_event(self.current_time_step)
        self.order_event_table.delete_event(self.current_time_step)
        self.path_planning_event_table.delete_event(self.current_time_step)

        if self.current_time_step % 10 == 0:
            self.collision_index.delete(self.current_time_step)

    def assign_orders_to_workstations(self):
        idle_robots = [robot for robot in self.robots if robot.status == 0]
        num_idle_robots = len(idle_robots)
        if num_idle_robots == 0:
            return

        while self.available_orders and num_idle_robots > 0:
            min_time_steps_workstation = min(
                self.workstations_map.values(),
                key=lambda ws: ws.time_step_to_free
            )

            if (min_time_steps_workstation.time_step_to_free - self.current_time_step >=
                    self.layout.shape[0] + self.layout.shape[1]):
                break
            start_time = time.time()
            if self.matching_method == MatchingAlgorithm.SIMPLE:
                if len(self.simple_orders):
                    new_matching = self.match_orders_with_racks(self.simple_orders[0], min_time_steps_workstation)
                else:
                    new_matching = self.match_orders_with_racks(self.complex_orders[0], min_time_steps_workstation)
                if len(new_matching.rack_list) == 0:
                    absence_flag = False
                    for item_type in new_matching.order_list[0].items_needed:
                        if item_type not in self.item_types:
                            absence_flag = True
                    if not absence_flag:
                        break
            else:
                new_matching = self.match_orders_with_racks(self.available_orders[0], min_time_steps_workstation)
            matched_racks = new_matching.rack_list
            end_time = time.time()
            self.matching_total_time += (end_time - start_time)

            num_racks_needed = len(matched_racks)

            if num_idle_robots < num_racks_needed:
                break

            self.finish_match(new_matching)
            matched_orders = new_matching.order_list

            for order in matched_orders:
                try:
                    if len(order.items_needed) > 2:
                        self.complex_orders.remove(order)
                    else:
                        self.simple_orders.remove(order)
                except ValueError as e:
                    print("ValueError: Order not found in available_orders.")
                    print("Order to remove:", order)
                    print("Current complex_orders:", self.complex_orders)
                    print(matched_orders)
                    raise  
                try:
                    self.available_orders.remove(order)
                except ValueError as e:
                    print("ValueError: Order not found in available_orders.")
                    print("Order to remove:", order)
                    print("Current available_orders:", self.available_orders)
                    raise  

            num_idle_robots -= num_racks_needed
            self.total_deliveries += num_racks_needed

    def match_orders_with_racks(self, order, workstation):
        selected_racks = []
        contributions = []
        if self.matching_method == MatchingAlgorithm.GREEDY or self.matching_method == MatchingAlgorithm.SIMPLE:
            selected_racks, contributions = greedy_select_racks(self.racks, order)
        else:
            solutions = []
            if self.matching_method == MatchingAlgorithm.GREEDY_TOP_K:
                solutions = greedy_select_racks_top_k(self.racks, order)
            elif self.matching_method == MatchingAlgorithm.HNSW_TOP_K:
                solutions = self.mips_index.kmips_select_racks_top_k(order)
            elif self.matching_method == MatchingAlgorithm.DQN:
                solutions = self.mips_index.kmips_select_racks_top_k_with_dqn(order, workstation=workstation,
                                                                              agent=dqn_agent)
            if self.select_method == RackGroupAlgorithm.RANDOM and len(solutions):
                selected_racks, contributions = solutions[0]
            elif self.select_method == RackGroupAlgorithm.GREEDY:
                min_max_distance = inf
                for solution in solutions:
                    max_distance = -inf
                    for rack in solution[0]:
                        distance = manhattan_distance(rack.position, workstation.position)
                        max_distance = max(max_distance, distance)
                    if min_max_distance > max_distance:
                        min_max_distance = max_distance
                        selected_racks, contributions = solution

        processing_times = calculate_processing_time(selected_racks, contributions, self.picking_time)
        new_matching = Matching(workstation, selected_racks, None, [order], processing_times)
        return new_matching

    def finish_match(self, matching):
        rack_list = matching.rack_list  

        idle_robots = {robot for robot in self.robots if robot.status == 0}
        matched_robots = []  

        for rack in rack_list:
            closest_robot = None
            min_distance = float('inf')

            for robot in idle_robots:
                distance = abs(robot.position[0] - rack.position[0]) + abs(robot.position[1] - rack.position[1])
                if distance < min_distance:
                    closest_robot = robot
                    min_distance = distance

            if closest_robot:
                matched_robots.append(closest_robot)
                closest_robot.set_status(1)
                idle_robots.remove(closest_robot)

            rack.set_status(1)

        matching.robot_list = matched_robots

        if self.bundling_method == BundlingAlgorithm.REVERSE:
            main_order = matching.order_list[0]
            self.order_index.remove_orders([main_order])
            remaining_inv = calculate_remaining_inventory(rack_list, main_order)
            matched_orders = self.order_index.find_satisfiable_orders(remaining_inv)
            matching.order_list.extend(matched_orders)
            matching.recalculate_picking_times(self.picking_time)
        elif self.bundling_method == BundlingAlgorithm.HYBRID:
            main_order = matching.order_list[0]
            self.order_index.remove_orders([main_order])
            remaining_inv = calculate_remaining_inventory(rack_list, main_order)
            can_fill = True
            for order in self.complex_orders:
                if order is main_order:
                    continue
                for needed_item, needed_quantity in order.items_needed.items():
                    if remaining_inv.get(needed_item, 0) < needed_quantity:
                        can_fill = False
                        break
                if not can_fill:
                    continue
                matching.order_list.append(order)
                for needed_item, needed_quantity in order.items_needed.items():
                    remaining_inv[needed_item] -= needed_quantity
            matched_orders = self.order_index.find_satisfiable_orders(remaining_inv)
            matching.order_list.extend(matched_orders)
            matching.recalculate_picking_times(self.picking_time)

        new_event = ('1', matching)
        self.path_planning_event_table.add_event(self.current_time_step, new_event)

    def read_orders_from_file(self, file_path, time_multiplier):
        orders_list = []

        with open(file_path, 'r') as file:
            for idx, line in enumerate(file):
                time_str, items_str = line.strip().split(': ', 1)
                arrival_time_step = int(time_str) * time_multiplier
                items_needed = eval(items_str)
                order_id = f"order_{idx + 1}"
                order = Order(order_id, items_needed, arrival_time_step)
                orders_list.append(order)

        self.order_pool = orders_list

    def load_layout_and_racks(self, file_path):
        with open(file_path, 'r') as f:
            H, W, C = map(int, f.readline().strip().split())
            warehouse = []
            for _ in range(H):
                row = list(map(int, f.readline().strip().split()))
                warehouse.append(row)
            warehouse = np.array(warehouse)
            racks = []
            for line in f:
                parts = line.strip().split()
                rack_id = parts[0]
                position = tuple(map(int, parts[1].split(',')))  
                items = {int(k): int(v) for k, v in (item.split(':') for item in parts[2:])}
                self.item_types.update(items.keys())
                racks.append(Rack(rack_id, position, items))

        self.layout = warehouse
        self.racks = racks

    def initialize_robots(self, robot_num):
        for i in range(1, robot_num + 1):
            while True:
                position = (random.randint(0, self.layout.shape[0] - 1), random.randint(0, self.layout.shape[1] - 1))
                if self.clear_roads[position] == 0 and position not in [r.position for r in self.robots]:
                    break
            robot = Robot(f"robot_{i}", position)
            self.robots.append(robot)

    def initialize_workstations(self, workstation_num):
        H, W = self.layout.shape
        positions = []

        for i in range(1, H - 1):
            positions.append((i, W - 3))

        for j in range(1, W - 1):
            positions.append((H - 3, j))

        for i in range(1, workstation_num + 1):
            position = positions[i % len(positions)]
            ws = Workstation(f"ws_{i}", position)
            self.workstations.append(ws)

    def initialize_order_event_table(self, time_interval=20):
        if not self.order_pool:
            return  

        latest_arrival_time = self.order_pool[-1].arrival_time_step

        time_step = 0
        while time_step <= latest_arrival_time:
            time_step += time_interval
            self.order_event_table.add_event(time_step, ('A',))  

    def print_info(self):
        print("当前使用的算法配置:")
        print(f" - Bundling Algorithm        : {self.bundling_method.value}")
        print(f" - Matching Algorithm        : {self.matching_method.value}")
        print(f" - Selecting Algorithm       : {self.select_method.value}")
        print(f" - Path Finding Algorithm    : {self.pathfind_method.value}")
        print("其它参数：")
        print(f" - 总订单数: {self.total_orders}")
        print(f" - 仓库布局: {self.layout.shape[0]}X{self.layout.shape[1]}")
        print(f" - 机器人数/工作站数: {len(self.robots)}/{len(self.workstations)}")


class EventTable:
    def __init__(self, belong_to: Warehouse):
        self.events = {}  
        self.warehouse = belong_to

    def add_event(self, time_step, event):
        if time_step not in self.events:
            self.events[time_step] = []
        self.events[time_step].append(event)

    def get_events(self, time_step):
        return self.events.pop(time_step, [])

    def delete_event(self, time_step):
        if time_step in self.events:
            del self.events[time_step]
            
    def not_empty(self):
        return bool(self.events)

    def execute_events(self, time_step):
        events_to_execute = self.get_events(time_step)
        pathfinding_batch = []
        if len(events_to_execute) and self.warehouse.print_event:
            print(f"--- Time step {time_step} ---")
        for event in events_to_execute:
            if self.warehouse.print_event:
                print(event)

            event_type = event[0]  
            if event_type == 'R':  
                EventTable.flip_object_status(event[1])
            elif event_type == 'A':  
                self.process_order_arrival(time_step)
            elif event_type == 'F':  
                self.complete_order(event[1])
            elif event_type in ['1', '3']:  
                self.handle_path_planning(event_type, *event[1:])
            elif event_type == '2':
                if self.warehouse.pathfind_method == PathFindingAlgorithm.A_STAR:
                    self.handle_path_planning(event_type, *event[1:])
                elif self.warehouse.pathfind_method == PathFindingAlgorithm.P3F:
                    pathfinding_batch.append(event[1])
        if self.warehouse.pathfind_method == PathFindingAlgorithm.P3F and len(pathfinding_batch):
            self.path_planning_phase2_in_batch(pathfinding_batch)

    def show_all_events(self):
        if not self.events:
            print("No events in the event table.")
        else:
            for time_step, event_list in sorted(self.events.items()):
                print(f"Time Step {time_step}:")
                for event in event_list:
                    print(f"  Event: {event}")

    
    @staticmethod
    def flip_object_status(object):
        object.set_status(0)
        
    def process_order_arrival(self, time_step):
        order_pool = self.warehouse.order_pool  
        available_orders = self.warehouse.available_orders  

        idx = 0
        while idx < len(order_pool) and order_pool[idx].arrival_time_step <= time_step:
            idx += 1

        available_orders.extend(order_pool[:idx])
        self.warehouse.simple_orders.extend(order for order in order_pool[:idx] if len(order.items_needed) <= 2)
        self.warehouse.complex_orders.extend(order for order in order_pool[:idx] if len(order.items_needed) > 2)

        self.warehouse.order_index.add_orders(order_pool[:idx])
        del order_pool[:idx]

    def complete_order(self, orders):
        for order in orders:
            self.warehouse.finished_orders += 1
            if self.warehouse.print_event:
                print(f"Order {order.id} completed")
    
    def handle_path_planning(self, phase, *args):
        if phase == '1':
            self.path_planning_phase1(*args)
        elif phase == '2':
            self.path_planning_phase2(*args)
        elif phase == '3':
            self.path_planning_phase3(*args)

    def path_planning_phase1(self, matching):
        time_step = self.warehouse.current_time_step
        collision_index = self.warehouse.collision_index
        origins = [robot.position for robot in matching.robot_list]
        destinations = [rack.position for rack in matching.rack_list]

        for origin, destination in zip(origins, destinations):
            path = a_star(self.warehouse.clear_roads, origin, destination, collision_index, time_step)
            matching.robot_arriving_times.append(time_step + len(path) - 1)
            if path:
                collision_index.update([path])  
                self.warehouse.total_path_length += max(0, len(path) - 1)

        for robot, rack in zip(matching.robot_list, matching.rack_list):
            robot.position = rack.position

        self.warehouse.path_planning_event_table.add_event(time_step + 1, ('2', matching))

    def path_planning_phase2(self, matching):
        collision_index = self.warehouse.collision_index
        time_step_to_free = matching.workstation.time_step_to_free  
        origins = [rack.position for rack in matching.rack_list]
        destinations = [matching.workstation.position] * len(matching.rack_list)
        robot_arriving_times = matching.robot_arriving_times  
        picking_times = matching.picking_times  
        robots = matching.robot_list  
        racks = matching.rack_list  

        arrival_at_workstation_times = []  
        for origin, destination, arrival_time in zip(origins, destinations, robot_arriving_times):
            
            path = a_star(self.warehouse.layout, origin, destination, collision_index, arrival_time)
            arrival_at_workstation = arrival_time + len(path) - 1
            arrival_at_workstation_times.append(arrival_at_workstation)
            if path:
                collision_index.update([path])  
                self.warehouse.total_path_length += max(0, len(path) - 1)

        sorted_info = sorted(
            zip(arrival_at_workstation_times, picking_times, robots, racks),
            key=lambda x: x[0]
        )

        for arrival_at_workstation, picking_time, robot, rack in sorted_info:
            start_processing_time = max(arrival_at_workstation, time_step_to_free)
            completion_time = start_processing_time + picking_time

            time_step_to_free = completion_time
            matching.workstation.time_step_to_free = completion_time

            self.warehouse.path_planning_event_table.add_event(
                completion_time,
                ('3', matching.workstation.position, robot, rack)
            )

        self.warehouse.order_event_table.add_event(
            max(matching.workstation.time_step_to_free, self.warehouse.current_time_step + 1)
            , ('F', matching.order_list))

    def path_planning_phase2_in_batch(self, matching_list):
        collision_index = self.warehouse.collision_index
        robot_info_list = []

        for matching in matching_list:
            origins = [rack.position for rack in matching.rack_list]
            destinations = [matching.workstation.position] * len(matching.rack_list)
            robot_arriving_times = matching.robot_arriving_times
            picking_times = matching.picking_times
            robots = matching.robot_list
            racks = matching.rack_list

            temp_robot_info = []

            for origin, destination, arrival_time, picking_time, robot, rack in zip(
                    origins, destinations, robot_arriving_times, picking_times, robots, racks
            ):
                path = a_star(self.warehouse.layout, origin, destination, collision_index, arrival_time)
                arrival_at_workstation = arrival_time + len(path) - 1
                matching.rack_arriving_times_temp.append(arrival_at_workstation)

                temp_robot_info.append(
                    [0, picking_time, robot, rack, matching, origin, destination, arrival_time])

            idle_times = matching.calculate_idle_times()

            for i, idle_time in enumerate(idle_times):
                temp_robot_info[i][0] = idle_time

            robot_info_list.extend(temp_robot_info)

        robot_info_list.sort(key=lambda x: x[0])

        for idle_time, picking_time, robot, rack, matching, origin, destination, arrival_time in robot_info_list:
            path = a_star(self.warehouse.layout, origin, destination, collision_index, arrival_time)
            self.warehouse.total_path_length += max(0, len(path) - 1)
            arrival_at_workstation = arrival_time + len(path) - 1
            matching.sorted_robot_info.append((arrival_at_workstation, picking_time, robot, rack))

        for matching in matching_list:
            sorted_info = sorted(matching.sorted_robot_info, key=lambda x: x[0])
            time_step_to_free = matching.workstation.time_step_to_free

            for arrival_at_workstation, picking_time, robot, rack in sorted_info:
                start_processing_time = max(arrival_at_workstation, time_step_to_free)
                completion_time = start_processing_time + picking_time

                time_step_to_free = completion_time
                matching.workstation.time_step_to_free = completion_time

                self.warehouse.path_planning_event_table.add_event(
                    completion_time,
                    ('3', matching.workstation.position, robot, rack)
                )

            self.warehouse.order_event_table.add_event(
                max(matching.workstation.time_step_to_free, self.warehouse.current_time_step + 1),
                ('F', matching.order_list)
            )

    def path_planning_phase3(self, origin, robot, rack):
        time_step = self.warehouse.current_time_step
        collision_index = self.warehouse.collision_index
        destination = rack.position
        self.warehouse.layout[destination[0]][destination[1]] = 0
        path = a_star(self.warehouse.layout, origin, destination, collision_index, time_step)
        self.warehouse.layout[destination[0]][destination[1]] = 1
        if path:
            collision_index.update([path])  
            self.warehouse.total_path_length += max(0, len(path) - 1)

        event_time_step = time_step + len(path) - 1
        self.warehouse.status_event_table.add_event(max(event_time_step, time_step), ('R', robot))
        self.warehouse.status_event_table.add_event(max(event_time_step, time_step), ('R', rack))


algorithm_combination_COOP = [BundlingAlgorithm.HYBRID, MatchingAlgorithm.DQN,
                              RackGroupAlgorithm.GREEDY, PathFindingAlgorithm.P3F]
algorithm_combinations = [algorithm_combination_COOP]
order_files_A = ["orders_A100K.txt","orders_A200K.txt",
                 "orders_A400K.txt", "orders_A700K.txt", "orders_A1000K.txt"]
experimental_setting_A = ("warehouse_layout_RealA.txt", order_files_A, 503, 101)
experimental_settings = [experimental_setting_A]

file_path = "DQN_Model/dqn_trained.model"
k = 5
lr = 1e-3
gamma = 0.995
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
buffer_capacity = 10000
batch_size = 64
dqn_agent = create_agent_from_checkpoint(file_path, k, lr, gamma, epsilon, epsilon_decay, epsilon_min, buffer_capacity, batch_size)

for layout_file, order_files, robot_num, workstations_num in experimental_settings:
    length_list_list = []
    for order_file in order_files:
        length_list = []
        for ac in algorithm_combinations:
            length = []
            for i in range(1):
                env = Warehouse(layout_file, order_file, robot_num, workstations_num, ac, print_event=False,
                                dqn_agent=dqn_agent)
                print("\n" + "-" * 40)
                env.print_info()
                env.run()
                print(f"Order Throughput = {(env.finished_orders / env.current_time_step * 3600):.4f} orders per hour")
                print(f"total_deliveries = {env.total_deliveries}")
                print(f"total_path_length = {env.total_path_length}")
                print(f"pathfinding cost = {env.pathfinding_total_time}s")
                print(f"matching cost = {env.matching_total_time}s")
                print("-" * 40)
            length_list.append(length)
        length_list_list.append(length_list)
    print(length_list_list)
    