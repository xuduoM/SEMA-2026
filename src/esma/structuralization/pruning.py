"""Structural-information pruning operations."""

from esma.utils.structural_information import caculate_1d_se

def remove_graph_node(G, remove_id):
    """Remove a node from a copy of the graph."""
    G_copy = G.copy()
    remove_node = sorted(list(G_copy.nodes()))[remove_id]
    G_copy.remove_node(remove_node)

    edges = []
    for edge in G_copy.edges():
        edges.append([edge[0], edge[1], 1])
    return caculate_1d_se(edges, len(G_copy.nodes))

def remove_tree_node(tree, adj, remove_id):
    new_adj = adj.copy()

    new_adj[remove_id, :] = 0.00001
    new_adj[:, remove_id] = 0.00001

    return tree.temp_entropy(new_adj)

def cal_deltaH(G, tree, adj, original_G, original_T):
    delta_graph = []
    delta_tree = []
    for i in range(len(G.nodes)):
        se_G = remove_graph_node(G, i)
        se_T = remove_tree_node(tree, adj, i)
        delta_graph.append(abs(se_G - original_G))
        delta_tree.append(abs(se_T - original_T))

    return delta_graph, delta_tree

def combine_delta(alpha, beta, G, tree, adj, original_G, original_T):
    delta_graph, delta_tree = cal_deltaH(G, tree, adj, original_G, original_T)
    combine_delta_H = []
    for i in range(len(delta_tree)):
        cdH = alpha * delta_graph[i] + beta * delta_tree[i]
        combine_delta_H.append(cdH)
    return combine_delta_H

def get_nested_value(data, key_path):
    """Read a nested value using a dot-separated path."""
    keys = key_path.split('.')
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif isinstance(current, list):
            try:
                idx = int(key)
                if 0 <= idx < len(current):
                    current = current[idx]
                else:
                    return None
            except ValueError:

                return current
        else:
            return None
    return current

def prune_obs(G, u, combine_delta, obs):
    nodes = list(G.nodes())
    low_delta_indices = [i for i, d in enumerate(combine_delta) if d < u]
    low_delta_nodes = [nodes[i] for i in low_delta_indices]

    if len(low_delta_nodes) <= 5:
        nodes_to_remove = set(low_delta_nodes)
    else:
        low_delta_with_values = [(nodes[i], combine_delta[i]) for i in low_delta_indices]
        low_delta_with_values.sort(key=lambda x: x[1])
        nodes_to_remove = set([node for node, _ in low_delta_with_values[:5]])

    remaining_nodes = [node for node in nodes if node not in nodes_to_remove]

    pruned_obs = {}

    node_to_path = {

        'mineral': 'resource.mineral',
        'gas': 'resource.gas',
        'supply_left': 'resource.supply_left',
        'supply_cap': 'resource.supply_cap',
        'worker_supply': 'resource.worker_supply',
        'vespene_geyser_count': 'resource.vespene_geyser_count',

        'pylon_count': 'building.pylon_count',
        'nexus_count': 'building.nexus_count',
        'gateway_count': 'building.gateway_count',
        'cybernetics_core_count': 'building.cybernetics_core_count',
        'gas_buildings_count': 'building.gas_buildings_count',
        'robotics_facility_count': 'building.robotics_facility_count',
        'stargate_count': 'building.stargate_count',
        'forge_count': 'building.forge_count',

        'probe_count': 'unit.probe_count',
        'zealot_count': 'unit.zealot_count',
        'stalker_count': 'unit.stalker_count',
        'adept_count': 'unit.adept_count',
        'immortal_count': 'unit.immortal_count',
        'observer_count': 'unit.observer_count',
        'voidray_count': 'unit.voidray_count',
        'phoenix_count': 'unit.phoenix_count',

        'warpgate_research_status': 'research.cybernetics_core.warpgate_research_status',
        'blink_research_status': 'research.twilight_council.blink_research_status',
        'charge_research_status': 'research.twilight_council.charge_research_status',
        'protoss_ground_weapons_level_1': 'research.forge.protoss_ground_weapon_level_1_research_status',
        'protoss_air_weapons_level_1': 'research.cybernetics_core.protoss_air_weapon_level_1_research_status',
        'resonating_glaives_research_status': 'research.twilight_council.resonating_glaives_research_status',

        'enemy_ground_units': 'enemy.unit.enemy_ground_units',
        'enemy_air_units': 'enemy.unit.enemy_air_units',
        'enemy_worker_count': 'enemy.unit.enemy_worker_count'
    }

    for node in remaining_nodes:
        if node in obs:
            pruned_obs[node] = obs[node]

    for node in remaining_nodes:
        if node not in pruned_obs and node in node_to_path:
            path = node_to_path[node]
            value = get_nested_value(obs, path)

            if value is not None:

                path_parts = path.split('.')
                current = pruned_obs

                for i, part in enumerate(path_parts[:-1]):
                    if part not in current:

                        if i + 1 < len(path_parts) and path_parts[i + 1].isdigit():
                            current[part] = []
                        else:
                            current[part] = {}

                    if isinstance(current[part], list):

                        idx = int(path_parts[i + 1])
                        while len(current[part]) <= idx:
                            current[part].append({})
                        current = current[part][idx]
                    else:
                        current = current[part]

                last_part = path_parts[-1]
                if isinstance(current, list):

                    idx = int(last_part)
                    if idx < len(current):
                        current[idx] = value
                else:
                    current[last_part] = value

    return pruned_obs
