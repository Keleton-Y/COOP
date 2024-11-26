import math
from collections import deque
from sortedcontainers import SortedList
from collections import defaultdict


class Robot:
    def __init__(self, robot_id, position):
        self.id = robot_id  
        self.position = position  
        self.status = 0  

    def set_status(self, status):
        self.status = status

    def move_to(self, new_position):
        self.position = new_position

    def __repr__(self):
        return f"Robot(id={self.id}, position={self.position}, status={self.status})"


class Order:
    def __init__(self, order_id, items_needed, arrival_time_step):
        self.id = order_id  
        self.items_needed = items_needed  
        self.workstation = None  
        self.matched_racks = []  
        self.arrival_time_step = arrival_time_step  

    def set_workstation(self, workstation):
        self.workstation = workstation

    def set_matched_racks(self, racks):
        self.matched_racks = racks

    def show_items_needed(self):
        if not self.items_needed:
            print("This order needs no item.")
        else:
            print(f"Order {self.id} needs items:")
            for item, quantity in self.items_needed.items():
                print(f"  - {item}: {quantity}")

    def __lt__(self, other):
        
        return self.id < other.id

    def __repr__(self):
        return f"Order(id={self.id}, items_needed={self.items_needed})"


class Rack:
    def __init__(self, rack_id, position, items):
        self.id = rack_id  
        self.position = position  
        self.items = items  
        self.status = 0  

    def set_status(self, status):
        self.status = status

    def __repr__(self):
        return f"Rack(id={self.id}, position={self.position}, status={self.status}, items={self.items})"


class Workstation:
    def __init__(self, ws_id, position):
        self.id = ws_id  
        self.position = position  
        self.rack_queue = deque()  
        self.time_step_to_free = 0  
        self.last_update_time = 0  

    def add_rack(self, rack):
        self.rack_queue.append(rack)

    def process_racks(self, current_time):
        if self.rack_queue:
            self.time_step_to_free = len(self.rack_queue) * 10  
            self.last_update_time = current_time

    def update(self, current_time):
        if current_time - self.last_update_time >= self.time_step_to_free:
            
            self.rack_queue.clear()
            self.time_step_to_free = 0

    def __repr__(self):
        return f"Workstation(id={self.id}, position={self.position})"


class Matching:
    def __init__(self, workstation, rack_list, robot_list, order_list, picking_times):
        self.workstation = workstation
        self.rack_list = rack_list
        self.robot_list = robot_list
        self.order_list = order_list
        self.picking_times = picking_times
        self.robot_arriving_times = []
        
        self.rack_arriving_times_temp = []
        self.sorted_robot_info = []

    def calculate_idle_times(self):
        
        racks_info = sorted(
            enumerate(zip(self.rack_arriving_times_temp, self.picking_times)),
            key=lambda x: x[1][0]  
        )

        queue_time = self.workstation.time_step_to_free  
        start_times = [0] * len(self.rack_arriving_times_temp)  
        idle_times = [0] * len(self.rack_arriving_times_temp)  
        wait_times = [0] * len(self.rack_arriving_times_temp)  
        
        for original_index, (arrival_time, processing_time) in racks_info:
            
            start_time = max(arrival_time, queue_time)
            start_times[original_index] = start_time
            queue_time = start_time + processing_time  
        
        for i in range(len(racks_info) - 1, -1, -1):
            original_index = racks_info[i][0]
            arrival_time = self.rack_arriving_times_temp[original_index]
            start_time = start_times[original_index]

            if i == len(racks_info) - 1:
                wait_time = start_time - arrival_time
            else:
                next_index = racks_info[i + 1][0]
                next_arrival_time = self.rack_arriving_times_temp[next_index]
                next_wait_time = wait_times[next_index]
                delta_pt = self.picking_times[next_index] - self.picking_times[original_index]
                delta_at = next_arrival_time - arrival_time

                if start_time <= next_arrival_time:
                    wait_time = start_time - arrival_time  
                else:
                    wait_time = next_wait_time + delta_pt + delta_at  

            wait_times[original_index] = wait_time  

        for i, (original_index, _) in enumerate(racks_info):
            wait_time = wait_times[original_index]
            idle_time = wait_time  

            idle_time += sum(
                start_times[racks_info[j + 1][0]] - start_times[racks_info[j][0]] - self.picking_times[racks_info[j][0]]
                for j in range(i, len(racks_info) - 1)
            )

            idle_times[original_index] = idle_time  

        return idle_times

    def recalculate_picking_times(self, unit_picking_time):
        self.picking_times = [0] * len(self.rack_list)

        rack_distances = [
            (rack, abs(rack.position[0] - self.workstation.position[0]) + abs(
                rack.position[1] - self.workstation.position[1]))
            for rack in self.rack_list
        ]

        rack_distances.sort(key=lambda x: x[1])

        total_items_needed = {}
        for order in self.order_list:
            for item, quantity in order.items_needed.items():
                total_items_needed[item] = total_items_needed.get(item, 0) + quantity

        for rack, _ in rack_distances:
            total_contribution = 0  
            for item, available_quantity in rack.items.items():
                if item in total_items_needed:
                    contribution = min(total_items_needed[item], available_quantity)
                    total_contribution += contribution
                    total_items_needed[item] -= contribution
                    if total_items_needed[item] == 0:
                        del total_items_needed[item]  

            rack_index = self.rack_list.index(rack)
            self.picking_times[rack_index] = math.ceil(total_contribution * unit_picking_time)


class OrderIndex:
    def __init__(self):
        self.order_index = defaultdict(SortedList)  

    def add_orders(self, new_orders):
        for order in new_orders:
            if 1 <= len(order.items_needed) <= 2:
                for item, quantity in order.items_needed.items():
                    self.order_index[item].add((quantity, order))

    def remove_orders(self, orders_to_remove):
        for order in orders_to_remove:
            for item, quantity in order.items_needed.items():
                
                self.order_index[item].discard((quantity, order))

    def find_satisfiable_orders(self, current_inventory):
        satisfied_orders = []
        remaining_inventory = current_inventory.copy()  

        for item, available_quantity in remaining_inventory.items():
            added_orders = []
            
            if item not in self.order_index:
                continue  

            for quantity, order in list(self.order_index[item]):  
                if quantity > remaining_inventory[item]:
                    break

                can_fill = True
                for needed_item, needed_quantity in order.items_needed.items():
                    if remaining_inventory.get(needed_item, 0) < needed_quantity:
                        can_fill = False
                        break

                if not can_fill:
                    continue

                added_orders.append(order)

                for needed_item, needed_quantity in order.items_needed.items():
                    try:
                        remaining_inventory[needed_item] -= needed_quantity
                    except KeyError as e:
                        
                        print(f"KeyError: '{needed_item}' not found in remaining_inventory.")
                        print(f"Current remaining inventory keys: {list(remaining_inventory.keys())}")
                        print(f"Order items needed: {order.items_needed}")
                        print(f"Current item being checked: {needed_item}, needed quantity: {needed_quantity}")
                        raise  

            self.remove_orders(added_orders)
            satisfied_orders.extend(added_orders)

        return satisfied_orders


def calculate_remaining_inventory(racks, order):
    total_inventory = defaultdict(int)

    for rack in racks:
        for item, quantity in rack.items.items():
            total_inventory[item] += quantity

    for item, quantity in order.items_needed.items():
        total_inventory[item] -= quantity

    total_inventory = {item: qty for item, qty in total_inventory.items() if qty > 0}

    return total_inventory
