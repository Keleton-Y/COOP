import math

import numpy as np
from collections import defaultdict
import random
from sklearn.random_projection import SparseRandomProjection
from sklearn.neighbors import NearestNeighbors
import faiss
import hnswlib
import heapq



def greedy_select_racks(racks, order):
    remaining_order = order.items_needed.copy()

    selected_racks = []  
    rack_contributions = {}  

    while remaining_order:
        best_rack = None
        best_contribution = {}
        max_contribution_value = 0 

        for rack in racks:
            if rack.status == 1:
                continue

            contribution = {}
            contribution_value = 0 

            for product, quantity_needed in remaining_order.items():
                if product in rack.items and rack.items[product] > 0:
                    contribution_amount = min(quantity_needed, rack.items[product])
                    contribution[product] = contribution_amount
                    contribution_value += contribution_amount

            if contribution_value > max_contribution_value:
                best_rack = rack
                best_contribution = contribution
                max_contribution_value = contribution_value

        if not best_rack:
            break

        selected_racks.append(best_rack)
        rack_contributions[best_rack.id] = best_contribution

        best_rack.set_status(1)

        for product, quantity_provided in best_contribution.items():
            remaining_order[product] -= quantity_provided
            if remaining_order[product] == 0:
                del remaining_order[product]

    for rack in selected_racks:
        rack.set_status(0)

    return selected_racks, rack_contributions


def greedy_select_racks_top_k(racks, order, k=5):
    initial_remaining_order = order.items_needed.copy()
    candidates = [([], initial_remaining_order, {})]
    solutions = []

    while candidates:
        complete_solutions = [
            (selected_racks, contributions)
            for selected_racks, remaining_order, contributions in candidates
            if not remaining_order
        ]
        
        if complete_solutions:
            return complete_solutions

        new_candidates = []  
        
        for selected_racks, remaining_order, current_contributions in candidates:
            heap = []
            for rack in racks:
                if rack.status == 1 or rack in selected_racks:
                    continue

                contribution = {}
                contribution_value = 0
                for product, quantity_needed in remaining_order.items():
                    if product in rack.items and rack.items[product] > 0:
                        contribution_amount = min(quantity_needed, rack.items[product])
                        contribution[product] = contribution_amount
                        contribution_value += contribution_amount

                if contribution_value > 0:
                    heapq.heappush(heap, (contribution_value, rack.id, rack, contribution))

                if len(heap) > k:
                    heapq.heappop(heap)

            top_k_racks = [(rack, contribution) for _, _, rack, contribution in heap]
            
            for rack, contribution in top_k_racks:
                
                new_selected_racks = selected_racks + [rack]
                
                new_contributions = current_contributions.copy()
                new_contributions[rack.id] = contribution
                
                new_remaining_order = remaining_order.copy()
                for product, quantity_provided in contribution.items():
                    new_remaining_order[product] -= quantity_provided
                    if new_remaining_order[product] == 0:
                        del new_remaining_order[product]  
                
                new_candidates.append((new_selected_racks, new_remaining_order, new_contributions))
                if len(new_candidates) >= 200:
                    break
        
        candidates = new_candidates
    return solutions


def calculate_processing_time(selected_racks, rack_contributions, picking_time):
    processing_times = []

    for rack in selected_racks:
        rack_id = rack.id
        contribution = rack_contributions.get(rack_id, {})
        total_items = sum(contribution.values())  
        processing_time = math.ceil(total_items * picking_time)  
        processing_times.append(processing_time)

    return processing_times


def build_weighted_table(sampled_orders, n_max=21):
    weighted_table = {}
    
    all_products = set()
    for order in sampled_orders:
        all_products.update(order.items_needed.keys())
    
    for product in all_products:
        weighted_table[product] = []
        
        for G_r_i in range(n_max + 1):
            expected_value = 0
            count = 0
            for order in sampled_orders:
                if product in order.items_needed and order.items_needed[product] > 0:
                    G_t_i = order.items_needed[product]
                    expected_value += min(G_t_i, G_r_i) / G_t_i
                    count += 1
            
            if count > 0:
                expected_value /= count
            weighted_table[product].append(expected_value)

    return weighted_table


def calculate_weighted_shelf_from_table(shelf, weighted_table):
    weighted_shelf = {}
    
    for product, G_r_i in shelf.items():
        if product in weighted_table:
            if G_r_i < len(weighted_table[product]):
                weighted_shelf[product] = weighted_table[product][G_r_i]
            else:
                
                weighted_shelf[product] = 1
    return weighted_shelf


def calculate_weighted_shelves_from_table(rack_list, weighted_table):
    weighted_shelves = []
    
    for rack in rack_list:
        weighted_shelf = {}
        for product, G_r_i in rack.items.items():
            if product in weighted_table:
                if G_r_i < len(weighted_table[product]):
                    weighted_shelf[product] = weighted_table[product][G_r_i]
                else:
                    weighted_shelf[product] = 1
        weighted_shelves.append(weighted_shelf)
    return weighted_shelves



def generate_test_environment():
    products = [f"product_{i}" for i in range(1, 11)]
    shelves = []
    for _ in range(50):
        num_products = random.randint(2, 6)  
        selected_products = random.sample(products, num_products)  
        shelf = {}
        remaining_capacity = 30
        for i, product in enumerate(selected_products):
            if i == len(selected_products) - 1:
                quantity = remaining_capacity
            else:
                max_quantity = remaining_capacity - (len(selected_products) - i - 1)
                quantity = random.randint(1, max_quantity)
            shelf[product] = quantity
            remaining_capacity -= quantity
        shelves.append(shelf)
    
    orders = []
    for _ in range(50):
        order = {}
        num_order_products = random.randint(2, 5)  
        selected_order_products = random.sample(products, num_order_products)
        remaining_order_quantity = 12
        for i, product in enumerate(selected_order_products):
            if i == len(selected_order_products) - 1:
                quantity = remaining_order_quantity
            else:
                max_quantity = remaining_order_quantity - (len(selected_order_products) - i - 1)
                quantity = random.randint(1, max_quantity)
            order[product] = quantity
            remaining_order_quantity -= quantity
        orders.append(order)
    
    N_max = 12
    weighted_table = build_weighted_table(orders, N_max)
    
    for i, shelf in enumerate(shelves):
        weighted_shelf = calculate_weighted_shelf_from_table(shelf, weighted_table)
        print(f"货架 {i} 加权后的货架向量: {weighted_shelf}")
    
    query_order = random.choice(orders)
    
    print("\n随机抽取的订单需求:")
    for product, quantity in query_order.items():
        print(f"{product}: {quantity}")
    
    selected_shelves, shelf_contributions = greedy_select_racks(shelves, query_order)
    
    print("\n使用到的货架组合:")
    for shelf_idx in sorted(selected_shelves):
        print(f"货架 {shelf_idx}: {shelves[shelf_idx]}")
    
    print("\n每个货架的贡献:")
    for shelf_idx, contribution in shelf_contributions.items():
        print(f"货架 {shelf_idx} 贡献: {contribution}")


def generate_random_vectors(n_vectors, dimension):
    return np.random.randint(2, size=(n_vectors, dimension))


def generate_query_vector(dimension):
    return np.random.randint(10, size=dimension)


class LSH:
    def __init__(self, n_hashes, dimension):
        self.n_hashes = n_hashes
        self.dimension = dimension
        
        self.projections = np.random.randn(n_hashes, dimension)
        self.hash_tables = defaultdict(list)

    def hash_vector(self, vector):
        return tuple((np.dot(self.projections, vector) > 0).astype(int))

    def add_vector(self, vector, index):
        hash_value = self.hash_vector(vector)
        self.hash_tables[hash_value].append(index)

    def search(self, query_vector):
        hash_value = self.hash_vector(query_vector)
        return self.hash_tables.get(hash_value, [])


def inner_product(v1, v2):
    return np.dot(v1, v2)


class WeightedRackSearch:
    def __init__(self, weighted_racks, rack_list):
        
        assert len(weighted_racks) == len(rack_list), "weighted_racks和rack_list的长度必须相同"
        
        self.all_products = set()
        for rack in weighted_racks:
            self.all_products.update(rack.keys())
        self.all_products = list(self.all_products)  

        self.rack_matrix = np.array([
            [rack.get(product, 0) for product in self.all_products]  
            for rack in weighted_racks
        ]).astype('float32')

        self.dim = self.rack_matrix.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(self.rack_matrix)

        self.rack_list = rack_list

    def find_top_k_amips(self, required_items, k=5):
        query_vector = np.array([
            required_items.get(product, 0) for product in self.all_products
        ]).reshape(1, -1).astype('float32')

        distances, indices = self.index.search(query_vector, k)

        top_k_racks = [(self.rack_list[int(indices[0][i])], distances[0][i]) for i in range(k)]
        return top_k_racks


class WeightedRackSearchHNSW:
    def __init__(self, weighted_racks, rack_list, ef=200, M=16):
        assert len(weighted_racks) == len(rack_list), "weighted_racks 和 rack_list 的长度必须相同"

        self.all_products = set()
        for rack in weighted_racks:
            self.all_products.update(rack.keys())
        self.all_products = list(self.all_products)  

        self.rack_matrix = np.array([
            [rack.get(product, 0) for product in self.all_products]  
            for rack in weighted_racks
        ]).astype('float32')

        self.dim = self.rack_matrix.shape[1]
        self.index = hnswlib.Index(space='ip', dim=self.dim)  

        num_elements = len(self.rack_matrix)
        self.index.init_index(max_elements=num_elements, ef_construction=ef, M=M)
        self.index.add_items(self.rack_matrix)  

        self.index.set_ef(200)  
        self.rack_list = rack_list

    def find_top_k_amips(self, required_items, k=5, ef_search=None):
        if isinstance(required_items, dict):
            query_vector = np.array([
                required_items.get(product, 0) for product in self.all_products
            ]).reshape(1, -1).astype('float32')
        else:
            query_vector = required_items.reshape(1, -1).astype('float32')

        if ef_search:
            self.index.set_ef(ef_search)

        labels, distances = self.index.knn_query(query_vector, k=k)

        top_k_racks = [(self.rack_list[int(labels[0][i])], distances[0][i]) for i in range(k)]
        return top_k_racks

    def find_top_k_amips_in_batch(self, required_items_list, k=5, ef_search=None):
        query_vectors = []
        for required_items in required_items_list:
            if isinstance(required_items, dict):
                query_vector = np.array([
                    required_items.get(product, 0) for product in self.all_products
                ]).astype('float32')
            else:
                query_vector = required_items.astype('float32')

            query_vectors.append(query_vector)

        query_vectors = np.stack(query_vectors)

        if ef_search:
            self.index.set_ef(ef_search)

        labels, distances = self.index.knn_query(query_vectors, k=k)

        results = []
        for i in range(len(required_items_list)):
            top_k_racks = [
                (self.rack_list[int(labels[i][j])], distances[i][j]) for j in range(k)
            ]
            results.append(top_k_racks)

        return results

    def kmips_select_racks_top_k_with_dqn(self, order, k=5, workstation=None, ef_search=None, agent=None):
        if workstation is None:
            raise ValueError("workstation parameter must be provided.")

        remaining_order = order.items_needed.copy()
        selected_racks = []
        rack_contributions = {}
        state = None

        while remaining_order:
            top_k_racks = self.find_top_k_amips(remaining_order, k=k, ef_search=ef_search)
            state = get_state(top_k_racks, workstation)
            selected_rack = top_k_racks[agent.select_action(state)]
            contribution = {}
            
            for rack, _ in [selected_rack]:
                if rack.status == 1 or rack in selected_racks:
                    continue

                for product, quantity_needed in remaining_order.items():
                    if product in rack.items and rack.items[product] > 0:
                        contribution[product] = min(quantity_needed, rack.items[product])

            selected_racks.append(selected_rack)
            rack_contributions[selected_rack.id] = contribution

            for product, quantity_provided in contribution.items():
                remaining_order[product] -= quantity_provided
                if remaining_order[product] == 0:
                    del remaining_order[product]

        return [(selected_racks, rack_contributions)]

    def kmips_select_racks_top_k(self, order, k=5, ef_search=None):
        initial_remaining_order = order.items_needed.copy()
        candidates = [([], initial_remaining_order, {})]
        solutions = []

        while candidates:
            complete_solutions = [
                (selected_racks, contributions)
                for selected_racks, remaining_order, contributions in candidates
                if not remaining_order
            ]

            if complete_solutions:
                solutions.extend(complete_solutions)
                return solutions
            new_candidates = []  

            remaining_orders_batch = [remaining_order for _, remaining_order, _ in candidates]

            top_k_racks_batch = self.find_top_k_amips_in_batch(
                remaining_orders_batch, k=k, ef_search=ef_search
            )

            for i, (selected_racks, remaining_order, current_contributions) in enumerate(candidates):
                top_k_racks = top_k_racks_batch[i]  
                for rack, _ in top_k_racks:
                    
                    if rack.status == 1 or rack in selected_racks:
                        continue

                    contribution = {}
                    for product, quantity_needed in remaining_order.items():
                        if product in rack.items and rack.items[product] > 0:
                            contribution[product] = min(quantity_needed, rack.items[product])

                    new_selected_racks = selected_racks + [rack]
                    new_contributions = current_contributions.copy()
                    new_contributions[rack.id] = contribution

                    new_remaining_order = remaining_order.copy()
                    for product, quantity_provided in contribution.items():
                        new_remaining_order[product] -= quantity_provided
                        if new_remaining_order[product] == 0:
                            del new_remaining_order[product]

                    new_candidates.append((new_selected_racks, new_remaining_order, new_contributions))
                    if len(new_candidates) >= 200:
                        break
            candidates = new_candidates
        return solutions
        