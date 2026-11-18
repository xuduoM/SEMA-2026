"""Observation graph construction."""

import math
import time

import networkx as nx
import numpy as np
from esma.utils.structural_information import caculate_1d_se, trans_to_tree

class GraphBuilder:
    def __init__(self):

        self.edge_weights = {}
    def _init_phases_graphs(self, phase_name):

        if phase_name == "early":
            self.nodes = [

                "mineral",
                "gas",
                "supply_left",
                "supply_cap",
                "worker_supply",

                "worker_idle",
                "mineral_slots_left",
                "gas_slots_left",

                "vespene_geyser_count",
                "pylon_count",
                "nexus_count",
                "gas_buildings_count",

                "gateway_count",
                "cybernetics_core_count",
                "robotics_facility_count",

                "stargate_count",
                "forge_count",
                "warpgate_research_status",

                "probe_count",
                "zealot_count",
                "stalker_count",
                "adept_count",
                "immortal_count",
                "observer_count",

                "voidray_count",
                "phoenix_count",
                "blink_research_status",

                "charge_research_status",
                "protoss_ground_weapons_level_1",
                "protoss_air_weapons_level_1",

                "resonating_glaives_research_status",
                "enemy_ground_units",
                "enemy_air_units",
                "enemy_worker_count"
            ]

            self.connections = [

                (0, 4), (0, 18),
                (0, 9), (0, 10), (0, 12),

                (1, 4), (1, 11),
                (1, 22), (1, 24),
                (1, 28), (1, 29),

                (4, 0), (4, 1),
                (4, 10),
                (4, 5), (4, 6), (4, 7),

                (5, 4),
                (5, 6),
                (5, 7),

                (6, 4),
                (6, 5),
                (6, 10),
                (6, 18),

                (7, 4),
                (7, 5),
                (7, 8),
                (7, 11),

                (2, 3), (2, 9), (2, 4),

                (3, 9), (3, 19), (3, 20), (3, 24),

                (9, 0), (9, 3), (9, 10), (9, 2),

                (10, 0), (10, 4), (10, 9), (10, 18), (10, 6),

                (11, 1), (11, 4), (11, 8), (11, 7),

                (8, 11), (8, 7),

                (12, 0),
                (12, 19), (12, 20), (12, 21),
                (12, 17),

                (13, 20), (13, 21),
                (13, 14), (13, 15),
                (13, 26),

                (14, 22), (14, 23),
                (14, 13),

                (15, 24), (15, 25),
                (15, 13),

                (16, 28), (16, 29),

                (17, 12), (17, 13),

                (18, 0), (18, 1), (18, 10), (18, 6),

                (19, 12), (19, 27), (19, 3),

                (20, 12), (20, 13), (20, 26), (20, 3),

                (21, 12), (21, 13), (21, 30),

                (22, 0), (22, 1), (22, 14),

                (23, 14), (23, 31),

                (24, 1), (24, 15), (24, 32), (24, 3),

                (25, 15), (25, 32),

                (26, 13), (26, 20),

                (27, 19),
                (27, 13),

                (28, 16), (28, 19), (28, 22),

                (29, 16), (29, 24), (29, 25),

                (30, 21), (30, 13),

                (31, 22), (31, 19), (31, 23),

                (32, 24), (32, 25), (32, 20),

                (33, 18), (33, 4),
            ]

        elif phase_name == "mid":

            self.nodes = [

                "mineral",

                "gas",

                "supply_cap",

                "supply_left",

                "supply_used",

                "army_supply",

                "worker_supply",

                "nexus_count",

                "planning_nexus_count",

                "pylon_count",

                "gas_buildings_count",

                "planning_gas_buildings_count",

                "gateway_count",

                "warp_gate_count",

                "cybernetics_core_count",

                "twilight_council_count",

                "robotics_facility_count",

                "stargate_count",

                "robotics_bay_count",

                "fleet_beacon_count",

                "probe_count",

                "zealot_count",

                "stalker_count",

                "immortal_count",

                "colossus_count",

                "voidray_count",

                "observer_count",

                "warp_prism_count",

                "oracle_count",

                "worker_idle",

                "mineral_slots_left",

                "gas_slots_left",

                "warpgate_research_status",

                "charge_research_status",

            ]

            self.connections = [

                (0, 9), (0, 12), (0, 16), (0, 17),

                (0, 20), (0, 21), (0, 22), (0, 24),

                (1, 10), (1, 15), (1, 19),

                (1, 23), (1, 25), (1, 28),

                (2, 20), (2, 21), (2, 22), (2, 23), (2, 24), (2, 25),

                (3, 2), (3, 4),

                (4, 2), (4, 5),

                (6, 0), (6, 1), (6, 7),

                (6, 29), (6, 30), (6, 31),

                (7, 0), (7, 6), (7, 8), (7, 20),

                (20, 7), (20, 29),

                (9, 12), (9, 14), (9, 16), (9, 17),

                (12, 21), (12, 22), (12, 32),

                (14, 22), (14, 15), (14, 16), (14, 17),

                (15, 33), (15, 32),

                (16, 23), (16, 26), (16, 27),

                (16, 18),

                (17, 25), (17, 28),

                (21, 12), (21, 33),

                (22, 12), (22, 14), (22, 32),

                (23, 16), (23, 1),

                (24, 16), (24, 18), (24, 0),

                (25, 17), (25, 1), (25, 19),

                (26, 16), (26, 14),

                (27, 16), (27, 12),

                (28, 17), (28, 1),

                (29, 6), (29, 30), (29, 31),

                (30, 6), (30, 7), (30, 9),

                (31, 6), (31, 10), (31, 11),

                (32, 12), (32, 13), (32, 14),

                (33, 15), (33, 21),

                (8, 7), (8, 0),

                (11, 10), (11, 1),

                (18, 16), (18, 24),

                (19, 17), (19, 25),

                (10, 1), (10, 8), (10, 31),

                (5, 2), (5, 21), (5, 22), (5, 23), (5, 24), (5, 25),

                (13, 12), (13, 32),

                (7, 30),

                (10, 31),

            ]

        elif phase_name == "late":

            self.nodes = [

                "mineral",
                "gas",
                "supply_cap",

                "worker_supply",
                "nexus_count",
                "gas_buildings_count",

                "worker_idle",
                "mineral_slots_left",
                "gas_slots_left",

                "supply_used",
                "supply_left",
                "army_supply",

                "warp_gate_count",
                "robotics_facility_count",
                "stargate_count",
                "robotics_bay_count",
                "fleet_beacon_count",

                "carrier_count",
                "tempest_count",
                "colossus_count",
                "mothership_count",
                "archon_count",
                "voidray_count",
                "immortal_count",
                "stalker_count",

                "observer_count",
                "warp_prism_count",
                "high_templar_count",

                "warpgate_research_status",
                "charge_research_status",
            ]

            self.connections = [

                (0, 4), (0, 17), (0, 19), (0, 20),
                (0, 12), (0, 24),

                (1, 5), (1, 13), (1, 14),
                (1, 17), (1, 18), (1, 20),
                (1, 22), (1, 23),

                (2, 9), (2, 10), (2, 11),
                (2, 17), (2, 19), (2, 20),

                (3, 0), (3, 1), (3, 4), (3, 5),
                (3, 6), (3, 7), (3, 8),

                (4, 0), (4, 3), (4, 12),
                (4, 17),

                (5, 1), (5, 3), (5, 8),
                (5, 13), (5, 14),

                (6, 3), (6, 7), (6, 8),
                (6, 4),

                (7, 3), (7, 4), (7, 0),
                (7, 6),

                (8, 3), (8, 5), (8, 1),
                (8, 6),

                (9, 2), (9, 11), (9, 17),

                (10, 2), (10, 9), (10, 12),

                (11, 2), (11, 17), (11, 19), (11, 24),

                (12, 0), (12, 4), (12, 24),
                (12, 28),

                (13, 1), (13, 19), (13, 23), (13, 25),
                (13, 15),

                (14, 1), (14, 17), (14, 18), (14, 22),
                (14, 16),

                (15, 13), (15, 19),

                (16, 14), (16, 17), (16, 18),

                (17, 0), (17, 1), (17, 2), (17, 14), (17, 16),

                (18, 1), (18, 14), (18, 16),

                (19, 0), (19, 13), (19, 15),

                (20, 0), (20, 1), (20, 14),

                (21, 27), (21, 11),

                (22, 1), (22, 14),

                (23, 1), (23, 13),

                (24, 0), (24, 12),

                (25, 13), (25, 1),

                (26, 13), (26, 12),

                (27, 12), (27, 21),

                (28, 12),

                (29, 12),
            ]

        n = len(self.nodes)
        self.adj_matrix = np.zeros((n, n), dtype=float)

        for connection in self.connections:
            i, j = connection[0], connection[1]
            self.adj_matrix[i][j] = self.adj_matrix[j][i] = 1.0

        self.G = nx.Graph()
        for node in self.nodes:
            self.G.add_node(node)

        for i in range(n):
            for j in range(i + 1, n):
                if self.adj_matrix[i][j] > 0:
                    self.G.add_edge(self.nodes[i], self.nodes[j], weight=self.adj_matrix[i][j])

        edges = []
        for edge in self.G.edges():
            edges.append([edge[0], edge[1], 1])
        self.H_G = caculate_1d_se(edges, len(self.G.nodes))

        self.tree = trans_to_tree(self.adj_matrix, 2)
        self.H_T = self.tree.entropy()

    def _get_base_importance(self, u, v):
        """Return the static base importance of an edge."""

        economy_core = {("worker_supply", "mineral"), ("worker_supply", "gas"),
                        ("probe_count", "mineral"), ("probe_count", "gas")}
        if (u, v) in economy_core or (v, u) in economy_core:
            return 0.3

        production_limit = {("pylon_count", "supply_cap"), ("supply_cap", "zealot_count"),
                            ("supply_cap", "stalker_count")}
        if (u, v) in production_limit or (v, u) in production_limit:
            return 0.25

        production = {("nexus_count", "probe_count"), ("gateway_count", "zealot_count"),
                      ("cybernetics_core_count", "stalker_count"), ("stargate_count", "voidray_count")}
        if (u, v) in production or (v, u) in production:
            return 0.2

        tech_upgrade = {("cybernetics_core_count", "blink_research_status"),
                        ("forge_count", "protoss_ground_weapons_level_1"),
                        ("protoss_ground_weapons_level_1", "zealot_count")}
        if (u, v) in tech_upgrade or (v, u) in tech_upgrade:
            return 0.15

        enemy_response = {("enemy_ground_units", "immortal_count"),
                          ("enemy_air_units", "voidray_count")}
        if (u, v) in enemy_response or (v, u) in enemy_response:
            return 0.1

        resource_gathering = {("vespene_geyser_count", "gas_buildings_count"),
                              ("gas_buildings_count", "gas")}
        if (u, v) in resource_gathering or (v, u) in resource_gathering:
            return 0.1

        return 0.05

    def _get_game_phase(self, obs):
        """Determine the current game phase."""
        game_time = obs.get("resource", {}).get("game_time", "00:00")
        try:
            minutes = int(game_time.split(':')[0])
        except:
            minutes = 0

        if minutes < 3:
            return "early_economy"
        elif minutes < 6:
            return "mid_transition"
        elif minutes < 10:
            return "mid_military"
        else:
            return "late_full"

    def _get_emergency_state(self, obs):
        """Detect urgent game states."""
        emergency_state = {
            "supply_block": False,
            "under_attack": False,
            "resource_starved": False,
            "tech_behind": False
        }

        supply_left = obs.get("resource", {}).get("supply_left", 10)
        if supply_left <= 1:
            emergency_state["supply_block"] = True

        enemy_units_near = self._detect_enemy_units_near_base(obs)
        if enemy_units_near > 0:
            emergency_state["under_attack"] = True

        mineral = obs.get("resource", {}).get("mineral", 0)
        gas = obs.get("resource", {}).get("gas", 0)
        if mineral < 100 and gas < 50:
            emergency_state["resource_starved"] = True

        cyber_core_count = obs.get("building", {}).get("cybernetics_core_count", 0)
        game_time = obs.get("resource", {}).get("game_time", "00:00")
        minutes = int(game_time.split(':')[0]) if ':' in game_time else 0
        if minutes > 4 and cyber_core_count == 0:
            emergency_state["tech_behind"] = True

        return emergency_state

    def _get_phase_adjustment(self, u, v, phase, obs):
        """Adjust edge weight for the game phase."""
        adjustment = 0.0

        if phase == "early_economy":

            if self._is_economy_related(u, v):
                adjustment += 0.4
            if self._is_basic_building_related(u, v):
                adjustment += 0.3

            if self._is_military_related(u, v):
                adjustment -= 0.3
            if self._is_advanced_tech_related(u, v):
                adjustment -= 0.4

        elif phase == "mid_transition":

            if self._is_economy_related(u, v):
                adjustment += 0.2
            if self._is_tech_related(u, v):
                adjustment += 0.3
            if self._is_basic_military_related(u, v):
                adjustment += 0.2

        elif phase == "mid_military":

            if self._is_military_related(u, v):
                adjustment += 0.4
            if self._is_tech_upgrade_related(u, v):
                adjustment += 0.3

            if self._is_expansion_related(u, v):
                adjustment += 0.2

        elif phase == "late_full":

            if self._is_advanced_military_related(u, v):
                adjustment += 0.3
            if self._is_advanced_tech_related(u, v):
                adjustment += 0.3
            if self._is_economy_related(u, v):
                adjustment += 0.1

        return adjustment

    def _get_emergency_adjustment(self, u, v, emergency_state, obs):
        """Adjust edge weight for urgent states."""
        adjustment = 0.0

        if emergency_state["supply_block"]:
            if self._is_supply_related(u, v):
                adjustment += 0.5

        if emergency_state["under_attack"]:
            if self._is_defense_related(u, v):
                adjustment += 0.6
            if self._is_combat_unit_related(u, v):
                adjustment += 0.4

            if self._is_economy_related(u, v):
                adjustment -= 0.3
            if self._is_tech_related(u, v):
                adjustment -= 0.3

        if emergency_state["resource_starved"]:
            if self._is_economy_related(u, v):
                adjustment += 0.4
            if self._is_worker_related(u, v):
                adjustment += 0.3

        if emergency_state["tech_behind"]:
            if self._is_tech_building_related(u, v):
                adjustment += 0.4
            if self._is_tech_related(u, v):
                adjustment += 0.3

        return adjustment

    def _is_economy_related(self, u, v):
        """Check whether an edge is economy-related."""
        economy_nodes = {"mineral", "gas", "worker_supply", "probe_count", "nexus_count"}
        return u in economy_nodes or v in economy_nodes

    def _is_basic_building_related(self, u, v):
        """Check whether an edge involves basic structures."""
        basic_building_nodes = {"pylon_count", "gateway_count", "cybernetics_core_count", "gas_buildings_count"}
        return u in basic_building_nodes or v in basic_building_nodes

    def _is_military_related(self, u, v):
        """Check whether an edge is military-related."""
        military_nodes = {"zealot_count", "stalker_count", "adept_count", "immortal_count",
                          "voidray_count", "phoenix_count"}
        return u in military_nodes or v in military_nodes

    def _is_basic_military_related(self, u, v):
        """Check whether an edge involves basic military units."""
        basic_military_nodes = {"zealot_count", "stalker_count", "adept_count"}
        return u in basic_military_nodes or v in basic_military_nodes

    def _is_advanced_military_related(self, u, v):
        """Check whether an edge involves advanced military units."""
        advanced_military_nodes = {"immortal_count", "voidray_count", "phoenix_count", "observer_count"}
        return u in advanced_military_nodes or v in advanced_military_nodes

    def _is_tech_related(self, u, v):
        """Check whether an edge is technology-related."""
        tech_nodes = {"cybernetics_core_count", "warpgate_research_status", "blink_research_status",
                      "charge_research_status"}
        return u in tech_nodes or v in tech_nodes

    def _is_tech_upgrade_related(self, u, v):
        """Check whether an edge involves technology upgrades."""
        upgrade_nodes = {"protoss_ground_weapons_level_1", "protoss_air_weapons_level_1",
                         "resonating_glaives_research_status"}
        return u in upgrade_nodes or v in upgrade_nodes

    def _is_advanced_tech_related(self, u, v):
        """Check whether an edge involves advanced technology."""
        advanced_tech_nodes = {"robotics_facility_count", "stargate_count", "forge_count"}
        return u in advanced_tech_nodes or v in advanced_tech_nodes

    def _is_supply_related(self, u, v):
        """Check whether an edge is supply-related."""
        supply_nodes = {"supply_left", "supply_cap", "pylon_count"}
        return u in supply_nodes or v in supply_nodes

    def _is_defense_related(self, u, v):
        """Check whether an edge is defense-related."""
        defense_nodes = {"enemy_ground_units", "enemy_air_units", "zealot_count", "stalker_count"}
        return u in defense_nodes or v in defense_nodes

    def _is_combat_unit_related(self, u, v):
        """Check whether an edge involves combat units."""
        combat_nodes = {"zealot_count", "stalker_count", "immortal_count", "voidray_count"}
        return u in combat_nodes or v in combat_nodes

    def _is_worker_related(self, u, v):
        """Check whether an edge involves workers."""
        worker_nodes = {"worker_supply", "probe_count"}
        return u in worker_nodes or v in worker_nodes

    def _is_tech_building_related(self, u, v):
        """Check whether an edge involves technology structures."""
        tech_building_nodes = {"cybernetics_core_count", "robotics_facility_count", "stargate_count"}
        return u in tech_building_nodes or v in tech_building_nodes

    def _is_expansion_related(self, u, v):
        """Check whether an edge is expansion-related."""
        expansion_nodes = {"nexus_count", "mineral", "probe_count"}
        return u in expansion_nodes or v in expansion_nodes

    def _detect_enemy_units_near_base(self, obs):
        """Estimate the number of enemies near the base."""

        enemy_ground = obs.get("enemy", {}).get("unit", {}).get("enemy_ground_units", 0)
        enemy_air = obs.get("enemy", {}).get("unit", {}).get("enemy_air_units", 0)
        return enemy_ground + enemy_air

    def calculate_information_delta(self, current_info, previous_info, K):
        """Calculate normalized information change by percentage."""
        delta_info = {}

        if previous_info is None:
            return delta_info

        for key in list(current_info.keys())[:K + 1]:
            if key == 'game_time':
                continue

            current_value = current_info.get(key)
            previous_value = previous_info.get(key)

            if isinstance(current_value, (int, float)) and isinstance(previous_value, (int, float)):
                raw_delta = current_value - previous_value

                if previous_value != 0:

                    norm_delta = raw_delta / max(1.0, abs(previous_value))

                    delta_info[key] = max(-1.0, min(1.0, norm_delta))
                else:

                    delta_info[key] = raw_delta / 100.0

        return delta_info

    def assign_weights(self, delta_info):
        """Assign and normalize edge weights from information changes."""
        adj = self.adj_matrix.copy()
        current_weights = []

        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):

                if self.adj_matrix[i][j] > 0:
                    node1, node2 = self.nodes[i], self.nodes[j]

                    delta1 = float(delta_info.get(node1, 0)) if not isinstance(delta_info.get(node1, 0), bool) else (
                        1.0 if delta_info.get(node1, 0) else 0.0)
                    delta2 = float(delta_info.get(node2, 0)) if not isinstance(delta_info.get(node2, 0), bool) else (
                        1.0 if delta_info.get(node2, 0) else 0.0)

                    abs_product = abs(delta1 * delta2)
                    if abs_product > 0:
                        weight = abs_product * math.log(1 + abs_product)
                    else:
                        weight = 0.0

                    self.edge_weights[(node1, node2)] = weight
                    current_weights.append(weight)

        if current_weights:
            max_w = max(current_weights)
            min_w = min(current_weights)

            denom = max_w if max_w > 0 else 1.0

            for i in range(len(self.nodes)):
                for j in range(i + 1, len(self.nodes)):
                    if self.adj_matrix[i][j] > 0:
                        n1, n2 = self.nodes[i], self.nodes[j]

                        norm_w = self.edge_weights[(n1, n2)] / denom

                        self.edge_weights[(n1, n2)] = norm_w
                        adj[i][j] = norm_w
                        adj[j][i] = norm_w
        return adj

def print_encoding_tree(tree, node_names=None, root_id=None):
    """
    Print the encoding tree.
    tree: PartitionTree instance
    node_names: optional node names
    root_id: optional root node ID; detected automatically when omitted
    """
    if not hasattr(tree, 'tree_node'):
        print("Error: invalid encoding tree object")
        return

    node_dict = tree.tree_node

    if root_id is None:
        for node_id, node in node_dict.items():
            if node.parent is None:
                root_id = node_id
                break

    if root_id is None:
        print("Error: root node not found")
        return

    def _print_node(node_id, indent="", last=True):
        """Print a node recursively."""
        if node_id not in node_dict:
            return

        node = node_dict[node_id]
        print(indent, end="")
        if last:
            print("└── ", end="")
            new_indent = indent + "    "
        else:
            print("├── ", end="")
            new_indent = indent + "│   "

        node_info = f"ID:{node_id}"

        if node_names and hasattr(node, 'partition') and len(node.partition) == 1:
            leaf_index = node.partition[0]
            if 0 <= leaf_index < len(node_names):
                node_info += f" [{node_names[leaf_index]}]"

        node_info += f" (vol:{node.vol}, g:{node.g})"
        if hasattr(node, 'child_h'):
            node_info += f" depth:{node.child_h}"

        print(node_info)

        if node.children:
            children_list = list(node.children)
            for i, child_id in enumerate(children_list):
                _print_node(child_id, new_indent, i == len(children_list) - 1)

    print("Encoding tree:")
    _print_node(root_id)

if __name__ == "__main__":
    t1 = time.time()
    gw = GraphBuilder()
    gw._init_phases_graphs("early")
    t2 = time.time()
    print(t2 - t1)
