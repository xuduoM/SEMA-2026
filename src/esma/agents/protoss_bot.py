"""Core ESMA Protoss agent."""

import json
import math
import os
import random
import time
from collections import Counter, deque

from typing import Set, List

import tiktoken
from numpy.matlib import randn
from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Race, Difficulty, Result
from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2.position import Point2

from esma.agents.evaluation_expert import EvaluationExpert
from esma.structuralization.building_graph import GraphBuilder
from esma.agents.decision_agent import DecisionAgent
from esma.agents.analysis_expert import AnalyzationExpert
from esma.structuralization.pruning import combine_delta, prune_obs

MIN_DISTANCE = 3
BUILDING_DISTANCE = 7
CRITICAL_BUILDING_DISTANCE = 3
SCOUTING_INTERVAL = 300
from esma.envs.sc2_env import ActionDictionary

summary = ""
ep_history = deque(maxlen=10)

class ProtossBot(BotAI):
    def __init__(self, llmdm, llmsm, gw):
        super().__init__()
        self.prun_times = []
        self.input_tokens = []
        self.llm_decision = llmdm
        self.llm_summary = llmsm
        self.iteration = 0
        self.temp_failure_list = []
        self.inference_times = []
        self.ep_time = 0
        self.ep_result = 0
        self.rally_defend = False
        self.last_scouting_iteration = 0
        self.gw = gw
        self.episode_result = 0
        self.sc2dict = ActionDictionary('Protoss')
        self.action_dict = []
        self.own_race = Race.Protoss
        self.history = deque(maxlen=5)
        self.current_action_index = 0
        self.action_queue = None

    def get_information(self):
        self.worker_supply = self.workers.amount
        self.army_supply = self.supply_army

        self.base_count = self.structures(UnitTypeId.NEXUS).amount
        self.base_pending = self.already_pending(UnitTypeId.NEXUS)
        self.pylon_count = self.structures(UnitTypeId.PYLON).amount
        self.gas_buildings_count = self.structures(UnitTypeId.ASSIMILATOR).amount
        self.gateway_count = self.structures(UnitTypeId.GATEWAY).amount
        self.forge_count = self.structures(UnitTypeId.FORGE).amount
        self.photon_cannon_count = self.structures(UnitTypeId.PHOTONCANNON).amount
        self.shield_battery_count = self.structures(UnitTypeId.SHIELDBATTERY).amount
        self.warp_gate_count = self.structures(UnitTypeId.WARPGATE).amount
        self.cybernetics_core_count = self.structures(UnitTypeId.CYBERNETICSCORE).amount
        self.twilight_council_count = self.structures(UnitTypeId.TWILIGHTCOUNCIL).amount
        self.robotics_facility_count = self.structures(UnitTypeId.ROBOTICSFACILITY).amount
        self.stargate_count = self.structures(UnitTypeId.STARGATE).amount
        self.templar_archives_count = self.structures(UnitTypeId.TEMPLARARCHIVE).amount
        self.dark_shrine_count = self.structures(UnitTypeId.DARKSHRINE).amount
        self.robotics_bay_count = self.structures(UnitTypeId.ROBOTICSBAY).amount
        self.fleet_beacon_count = self.structures(UnitTypeId.FLEETBEACON).amount

        information = {

            'game_time': self.time_formatted,
            'worker_supply': self.workers.amount,
            'mineral': self.minerals,
            'gas': self.vespene,
            'supply_left': self.supply_left,
            'supply_cap': self.supply_cap,
            'supply_used': self.supply_used,
            'army_supply': self.supply_army,

            'nexus_count': self.base_count,
            'pylon_count': self.pylon_count,
            'gas_buildings_count': self.gas_buildings_count,
            'gateway_count': self.gateway_count,
            'forge_count': self.forge_count,
            'photon_cannon_count': self.photon_cannon_count,
            'shield_battery_count': self.shield_battery_count,
            'warp_gate_count': self.warp_gate_count,
            'cybernetics_core_count': self.cybernetics_core_count,
            'twilight_council_count': self.twilight_council_count,
            'robotics_facility_count': self.robotics_facility_count,
            'statgate_count': self.stargate_count,
            'templar_archives_count': self.templar_archives_count,
            'dark_shrine_count': self.dark_shrine_count,
            'robotics_bay_count': self.robotics_bay_count,
            'fleet_beacon_count': self.fleet_beacon_count,

            "probe_count": self.units(UnitTypeId.PROBE).amount,
            'Zealot_count': self.units(UnitTypeId.ZEALOT).amount,
            'stalker_count': self.units(UnitTypeId.STALKER).amount,
            'sentry_count': self.units(UnitTypeId.SENTRY).amount,
            'adept_count': self.units(UnitTypeId.ADEPT).amount,
            'high_templar_count': self.units(UnitTypeId.HIGHTEMPLAR).amount,
            'dark_templar_count': self.units(UnitTypeId.DARKTEMPLAR).amount,
            'immortal_count': self.units(UnitTypeId.IMMORTAL).amount,
            'colossus_count': self.units(UnitTypeId.COLOSSUS).amount,
            'disruptor_count': self.units(UnitTypeId.DISRUPTOR).amount,
            'archon_count': self.units(UnitTypeId.ARCHON).amount,
            'observer_count': self.units(UnitTypeId.OBSERVER).amount,
            'warp_prism_count': self.units(UnitTypeId.WARPPRISM).amount,
            'phoenix_count': self.units(UnitTypeId.PHOENIX).amount,
            'voidray_count': self.units(UnitTypeId.VOIDRAY).amount,
            'Oracle_count': self.units(UnitTypeId.ORACLE).amount,
            'Carrier_count': self.units(UnitTypeId.CARRIER).amount,
            'tempest_count': self.units(UnitTypeId.TEMPEST).amount,
            'mothership_count': self.units(UnitTypeId.MOTHERSHIP).amount,

            'planning_nexus_count': self.already_pending(UnitTypeId.NEXUS),
            'planning_pylon_count': self.already_pending(UnitTypeId.PYLON),
            'planning_gas_buildings_count': self.already_pending(UnitTypeId.ASSIMILATOR),
            'planning_gateway_count': self.already_pending(UnitTypeId.GATEWAY),
            'planning_forge_count': self.already_pending(UnitTypeId.FORGE),
            'planning_photon_cannon_count': self.already_pending(UnitTypeId.PHOTONCANNON),
            'planning_shield_battery_count': self.already_pending(UnitTypeId.SHIELDBATTERY),
            'planning_warp_gate_count': self.already_pending(UnitTypeId.WARPGATE),
            'planning_cybernetics_core_count': self.already_pending(UnitTypeId.CYBERNETICSCORE),
            'planning_twilight_council_count': self.already_pending(UnitTypeId.TWILIGHTCOUNCIL),
            'planning_robotics_facility_count': self.already_pending(UnitTypeId.ROBOTICSFACILITY),
            'planning_stargate_count': self.already_pending(UnitTypeId.STARGATE),
            'planning_templar_archives_count': self.already_pending(UnitTypeId.TEMPLARARCHIVE),
            'planning_dark_shrine_count': self.already_pending(UnitTypeId.DARKSHRINE),
            'planning_robotics_bay_count': self.already_pending(UnitTypeId.ROBOTICSBAY),
            'planning_fleet_beacon_count': self.already_pending(UnitTypeId.FLEETBEACON),
            'planning_probe_count': self.already_pending(UnitTypeId.PROBE),
            'planning_Zealot_count': self.already_pending(UnitTypeId.ZEALOT),
            'planning_stalker_count': self.already_pending(UnitTypeId.STALKER),
            'planning_sentry_count': self.already_pending(UnitTypeId.SENTRY),
            'planning_adept_count': self.already_pending(UnitTypeId.ADEPT),
            'planning_high_templar_count': self.already_pending(UnitTypeId.HIGHTEMPLAR),
            'planning_dark_templar_count': self.already_pending(UnitTypeId.DARKTEMPLAR),
            'planning_immortal_count': self.already_pending(UnitTypeId.IMMORTAL),
            'planning_colossus_count': self.already_pending(UnitTypeId.COLOSSUS),
            'planning_disruptor_count': self.already_pending(UnitTypeId.DISRUPTOR),
            'planning_archon_count': self.already_pending(UnitTypeId.ARCHON),
            'planning_observer_count': self.already_pending(UnitTypeId.OBSERVER),
            'planning_warp_prism_count': self.already_pending(UnitTypeId.WARPPRISM),
            'planning_phoenix_count': self.already_pending(UnitTypeId.PHOENIX),
            'planning_voidray_count': self.already_pending(UnitTypeId.VOIDRAY),
            'planning_Oracle_count': self.already_pending(UnitTypeId.ORACLE),
            'planning_Carrier_count': self.already_pending(UnitTypeId.CARRIER),
            'planning_tempest_count': self.already_pending(UnitTypeId.TEMPEST),
            'planning_mothership_count': self.already_pending(UnitTypeId.MOTHERSHIP),

            'warpgate_research_status': self.already_pending_upgrade(UpgradeId.WARPGATERESEARCH),
            'protoss_air_armor_level_1_research_status': self.already_pending_upgrade(
                UpgradeId.PROTOSSAIRARMORSLEVEL1),
            'protoss_air_armor_level_2_research_status': self.already_pending_upgrade(
                UpgradeId.PROTOSSAIRARMORSLEVEL2),
            'protoss_air_armor_level_3_research_status': self.already_pending_upgrade(
                UpgradeId.PROTOSSAIRARMORSLEVEL3),
            "protoss_air_weapon_level_1_research_status": self.already_pending_upgrade(
                UpgradeId.PROTOSSAIRWEAPONSLEVEL1),
            "protoss_air_weapon_level_2_research_status": self.already_pending_upgrade(
                UpgradeId.PROTOSSAIRWEAPONSLEVEL2),
            "protoss_air_weapon_level_3_research_status": self.already_pending_upgrade(
                UpgradeId.PROTOSSAIRWEAPONSLEVEL3),
            'protoss_ground_armor_level_1_research_status': self.already_pending_upgrade(
                UpgradeId.PROTOSSGROUNDARMORSLEVEL1),
            'protoss_ground_armor_level_2_research_status': self.already_pending_upgrade(
                UpgradeId.PROTOSSGROUNDARMORSLEVEL2),
            'protoss_ground_armor_level_3_research_status': self.already_pending_upgrade(
                UpgradeId.PROTOSSGROUNDARMORSLEVEL3),
            'protoss_ground_weapon_level_1_research_status': self.already_pending_upgrade(
                UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1),
            'protoss_ground_weapon_level_2_research_status': self.already_pending_upgrade(
                UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2),
            'protoss_ground_weapon_level_3_research_status': self.already_pending_upgrade(
                UpgradeId.PROTOSSGROUNDWEAPONSLEVEL3),
            'protoss_shield_level_1_research_status': self.already_pending_upgrade(
                UpgradeId.PROTOSSSHIELDSLEVEL1),
            'protoss_shield_level_2_research_status': self.already_pending_upgrade(
                UpgradeId.PROTOSSSHIELDSLEVEL2),
            'protoss_shield_level_3_research_status': self.already_pending_upgrade(
                UpgradeId.PROTOSSSHIELDSLEVEL3),
            'blink_research_status': self.already_pending_upgrade(UpgradeId.BLINKTECH),
            'charge_research_status': self.already_pending_upgrade(UpgradeId.CHARGE),
            'resonating_glaives_research_status': self.already_pending_upgrade(UpgradeId.ADEPTPIERCINGATTACK),
            'extended_thermal_lance_research_status': self.already_pending_upgrade(UpgradeId.EXTENDEDTHERMALLANCE),
            "GRAVITICDRIVE_research_status": self.already_pending_upgrade(UpgradeId.GRAVITICDRIVE),
            "OBSERVERGRAVITICBOOSTER_research_status": self.already_pending_upgrade(
                UpgradeId.OBSERVERGRAVITICBOOSTER),
            "PHOENIXRANGEUPGRADE_research_status": self.already_pending_upgrade(UpgradeId.PHOENIXRANGEUPGRADE),

            "VOIDRAYSPEEDUPGRADE_research_status": self.already_pending_upgrade(UpgradeId.VOIDRAYSPEEDUPGRADE),
            "PSISTORMTECH_research_status": self.already_pending_upgrade(UpgradeId.PSISTORMTECH),
            "SHADOWSTRIKE_research_status": self.already_pending_upgrade(UpgradeId.DARKTEMPLARBLINKUPGRADE),
        }

        information = self._update_enemy_information(information)
        return information

    def _get_resource_information(self):
        """Retrieve resource-related information."""
        self.worker_supply = self.workers.amount
        self.army_supply = self.supply_army
        return {
            'game_time': self.time_formatted,
            'worker_supply': self.worker_supply,
            'mineral': self.minerals,
            'gas': self.vespene,
            'supply_left': self.supply_left,
            'supply_cap': self.supply_cap,
            'supply_used': self.supply_used,
            'army_supply': self.army_supply,
        }

    def _get_building_information(self):
        """Retrieve building-related information."""
        self.base_count = self.structures(UnitTypeId.NEXUS).amount
        self.base_pending = self.already_pending(UnitTypeId.NEXUS)
        self.pylon_count = self.structures(UnitTypeId.PYLON).amount
        self.gas_buildings_count = self.structures(UnitTypeId.ASSIMILATOR).amount
        self.gateway_count = self.structures(UnitTypeId.GATEWAY).amount
        self.forge_count = self.structures(UnitTypeId.FORGE).amount
        self.photon_cannon_count = self.structures(UnitTypeId.PHOTONCANNON).amount
        self.shield_battery_count = self.structures(UnitTypeId.SHIELDBATTERY).amount
        self.warp_gate_count = self.structures(UnitTypeId.WARPGATE).amount
        self.cybernetics_core_count = self.structures(UnitTypeId.CYBERNETICSCORE).amount
        self.twilight_council_count = self.structures(UnitTypeId.TWILIGHTCOUNCIL).amount
        self.robotics_facility_count = self.structures(UnitTypeId.ROBOTICSFACILITY).amount
        self.stargate_count = self.structures(UnitTypeId.STARGATE).amount
        self.templar_archives_count = self.structures(UnitTypeId.TEMPLARARCHIVE).amount
        self.dark_shrine_count = self.structures(UnitTypeId.DARKSHRINE).amount
        self.robotics_bay_count = self.structures(UnitTypeId.ROBOTICSBAY).amount
        self.fleet_beacon_count = self.structures(UnitTypeId.FLEETBEACON).amount
        return {
            'nexus_count': self.base_count,
            'pylon_count': self.pylon_count,
            'gas_buildings_count': self.gas_buildings_count,
            'gateway_count': self.gateway_count,
            'forge_count': self.forge_count,
            'photon_cannon_count': self.photon_cannon_count,
            'shield_battery_count': self.shield_battery_count,
            'warp_gate_count': self.warp_gate_count,
            'cybernetics_core_count': self.cybernetics_core_count,
            'twilight_council_count': self.twilight_council_count,
            'robotics_facility_count': self.robotics_facility_count,
            'statgate_count': self.stargate_count,
            'templar_archives_count': self.templar_archives_count,
            'dark_shrine_count': self.dark_shrine_count,
            'robotics_bay_count': self.robotics_bay_count,
            'fleet_beacon_count': self.fleet_beacon_count,
        }

    def _get_unit_information(self):
        """Retrieve unit-related information."""
        return {
            "probe_count": self.units(UnitTypeId.PROBE).amount,
            'Zealot_count': self.units(UnitTypeId.ZEALOT).amount,
            'stalker_count': self.units(UnitTypeId.STALKER).amount,
            'sentry_count': self.units(UnitTypeId.SENTRY).amount,
            'adept_count': self.units(UnitTypeId.ADEPT).amount,
            'high_templar_count': self.units(UnitTypeId.HIGHTEMPLAR).amount,
            'dark_templar_count': self.units(UnitTypeId.DARKTEMPLAR).amount,
            'immortal_count': self.units(UnitTypeId.IMMORTAL).amount,
            'colossus_count': self.units(UnitTypeId.COLOSSUS).amount,
            'disruptor_count': self.units(UnitTypeId.DISRUPTOR).amount,
            'archon_count': self.units(UnitTypeId.ARCHON).amount,
            'observer_count': self.units(UnitTypeId.OBSERVER).amount,
            'warp_prism_count': self.units(UnitTypeId.WARPPRISM).amount,
            'phoenix_count': self.units(UnitTypeId.PHOENIX).amount,
            'voidray_count': self.units(UnitTypeId.VOIDRAY).amount,
            'Oracle_count': self.units(UnitTypeId.ORACLE).amount,
            'Carrier_count': self.units(UnitTypeId.CARRIER).amount,
            'tempest_count': self.units(UnitTypeId.TEMPEST).amount,
            'mothership_count': self.units(UnitTypeId.MOTHERSHIP).amount,
        }

    def _get_planning_information(self):
        """Retrieve planning-related information."""
        return {

            "planning_structure": {
                'planning_nexus_count': self.already_pending(UnitTypeId.NEXUS),
                'planning_pylon_count': self.already_pending(UnitTypeId.PYLON),
                'planning_gas_buildings_count': self.already_pending(UnitTypeId.ASSIMILATOR),
                'planning_gateway_count': self.already_pending(UnitTypeId.GATEWAY),
                'planning_forge_count': self.already_pending(UnitTypeId.FORGE),
                'planning_photon_cannon_count': self.already_pending(UnitTypeId.PHOTONCANNON),
                'planning_shield_battery_count': self.already_pending(UnitTypeId.SHIELDBATTERY),
                'planning_warp_gate_count': self.already_pending(UnitTypeId.WARPGATE),
                'planning_cybernetics_core_count': self.already_pending(UnitTypeId.CYBERNETICSCORE),
                'planning_twilight_council_count': self.already_pending(UnitTypeId.TWILIGHTCOUNCIL),
                'planning_robotics_facility_count': self.already_pending(UnitTypeId.ROBOTICSFACILITY),
                'planning_stargate_count': self.already_pending(UnitTypeId.STARGATE),
                'planning_templar_archives_count': self.already_pending(UnitTypeId.TEMPLARARCHIVE),
                'planning_dark_shrine_count': self.already_pending(UnitTypeId.DARKSHRINE),
                'planning_robotics_bay_count': self.already_pending(UnitTypeId.ROBOTICSBAY),
                'planning_fleet_beacon_count': self.already_pending(UnitTypeId.FLEETBEACON),
            },

            "planning_unit": {
                'planning_probe_count': self.already_pending(UnitTypeId.PROBE),
                'planning_Zealot_count': self.already_pending(UnitTypeId.ZEALOT),
                'planning_stalker_count': self.already_pending(UnitTypeId.STALKER),
                'planning_sentry_count': self.already_pending(UnitTypeId.SENTRY),
                'planning_adept_count': self.already_pending(UnitTypeId.ADEPT),
                'planning_high_templar_count': self.already_pending(UnitTypeId.HIGHTEMPLAR),
                'planning_dark_templar_count': self.already_pending(UnitTypeId.DARKTEMPLAR),
                'planning_immortal_count': self.already_pending(UnitTypeId.IMMORTAL),
                'planning_colossus_count': self.already_pending(UnitTypeId.COLOSSUS),
                'planning_disruptor_count': self.already_pending(UnitTypeId.DISRUPTOR),
                'planning_archon_count': self.already_pending(UnitTypeId.ARCHON),
                'planning_observer_count': self.already_pending(UnitTypeId.OBSERVER),
                'planning_warp_prism_count': self.already_pending(UnitTypeId.WARPPRISM),
                'planning_phoenix_count': self.already_pending(UnitTypeId.PHOENIX),
                'planning_voidray_count': self.already_pending(UnitTypeId.VOIDRAY),
                'planning_Oracle_count': self.already_pending(UnitTypeId.ORACLE),
                'planning_Carrier_count': self.already_pending(UnitTypeId.CARRIER),
                'planning_tempest_count': self.already_pending(UnitTypeId.TEMPEST),
                'planning_mothership_count': self.already_pending(UnitTypeId.MOTHERSHIP),
            }
        }

    def _get_research_information(self):
        return {

            "cybernetics_core": {
                'warpgate_research_status': self.already_pending_upgrade(UpgradeId.WARPGATERESEARCH),

                'protoss_air_armor_level_1_research_status': self.already_pending_upgrade(
                    UpgradeId.PROTOSSAIRARMORSLEVEL1),
                'protoss_air_armor_level_2_research_status': self.already_pending_upgrade(
                    UpgradeId.PROTOSSAIRARMORSLEVEL2),
                'protoss_air_armor_level_3_research_status': self.already_pending_upgrade(
                    UpgradeId.PROTOSSAIRARMORSLEVEL3),

                "protoss_air_weapon_level_1_research_status": self.already_pending_upgrade(
                    UpgradeId.PROTOSSAIRWEAPONSLEVEL1),
                "protoss_air_weapon_level_2_research_status": self.already_pending_upgrade(
                    UpgradeId.PROTOSSAIRWEAPONSLEVEL2),
                "protoss_air_weapon_level_3_research_status": self.already_pending_upgrade(
                    UpgradeId.PROTOSSAIRWEAPONSLEVEL3),
            },

            "forge": {

                'protoss_ground_armor_level_1_research_status': self.already_pending_upgrade(
                    UpgradeId.PROTOSSGROUNDARMORSLEVEL1),
                'protoss_ground_armor_level_2_research_status': self.already_pending_upgrade(
                    UpgradeId.PROTOSSGROUNDARMORSLEVEL2),
                'protoss_ground_armor_level_3_research_status': self.already_pending_upgrade(
                    UpgradeId.PROTOSSGROUNDARMORSLEVEL3),

                'protoss_ground_weapon_level_1_research_status': self.already_pending_upgrade(
                    UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1),
                'protoss_ground_weapon_level_2_research_status': self.already_pending_upgrade(
                    UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2),
                'protoss_ground_weapon_level_3_research_status': self.already_pending_upgrade(
                    UpgradeId.PROTOSSGROUNDWEAPONSLEVEL3),

                'protoss_shield_level_1_research_status': self.already_pending_upgrade(
                    UpgradeId.PROTOSSSHIELDSLEVEL1),
                'protoss_shield_level_2_research_status': self.already_pending_upgrade(
                    UpgradeId.PROTOSSSHIELDSLEVEL2),
                'protoss_shield_level_3_research_status': self.already_pending_upgrade(
                    UpgradeId.PROTOSSSHIELDSLEVEL3),
            },

            "twilight_council": {
                'blink_research_status': self.already_pending_upgrade(UpgradeId.BLINKTECH),
                'charge_research_status': self.already_pending_upgrade(UpgradeId.CHARGE),
                'resonating_glaives_research_status': self.already_pending_upgrade(UpgradeId.ADEPTPIERCINGATTACK),

            },

            "robotics_bay": {
                'extended_thermal_lance_research_status': self.already_pending_upgrade(UpgradeId.EXTENDEDTHERMALLANCE),
                "GRAVITICDRIVE_research_status": self.already_pending_upgrade(UpgradeId.GRAVITICDRIVE),
                "OBSERVERGRAVITICBOOSTER_research_status": self.already_pending_upgrade(
                    UpgradeId.OBSERVERGRAVITICBOOSTER),
            },

            "fleet_beacon": {
                "PHOENIXRANGEUPGRADE_research_status": self.already_pending_upgrade(UpgradeId.PHOENIXRANGEUPGRADE),

                "VOIDRAYSPEEDUPGRADE_research_status": self.already_pending_upgrade(UpgradeId.VOIDRAYSPEEDUPGRADE),
            },

            "templar_archives": {
                "PSISTORMTECH_research_status": self.already_pending_upgrade(UpgradeId.PSISTORMTECH),
            },

            "dark_shrine": {
                "SHADOWSTRIKE_research_status": self.already_pending_upgrade(UpgradeId.DARKTEMPLARBLINKUPGRADE), },
        }

    def _update_enemy_information(self, information):

        self.enemy_unit_type_amount = self.get_enemy_unity()
        self.enemy_structure_type_amount = self.get_enemy_structure()

        enemy_info = {}

        enemy_info['enemy_unit_total'] = self.enemy_units.amount

        enemy_info['enemy_structure_total'] = self.enemy_structures.amount

        try:
            units_info = {}
            unit_count = 0
            for enemy_unit in self.enemy_units:
                unit_key = f'enemy_unit_{unit_count}'
                unit_details = {
                    'type_id': str(enemy_unit.type_id),
                    'position_x': enemy_unit.position.x,
                    'position_y': enemy_unit.position.y,
                    'health': enemy_unit.health,
                    'health_max': enemy_unit.health_max,
                    'shield': enemy_unit.shield,
                    'shield_max': enemy_unit.shield_max,
                    'is_flying': enemy_unit.is_flying,
                    'is_attacking': enemy_unit.is_attacking,
                    'is_moving': enemy_unit.is_moving,
                    'distance_to_our_base': enemy_unit.distance_to(self.start_location),
                }
                units_info[unit_key] = unit_details
                unit_count += 1

            if units_info:
                enemy_info['units'] = units_info

        except Exception as e:
            print(f"Error updating detailed enemy unit information: {e}")

        try:
            structures_info = {}
            structure_count = 0
            for enemy_structure in self.enemy_structures:
                structure_key = f'enemy_structure_{structure_count}'
                structure_details = {
                    'type_id': str(enemy_structure.type_id),
                    'position_x': enemy_structure.position.x,
                    'position_y': enemy_structure.position.y,
                    'health': enemy_structure.health,
                    'health_max': enemy_structure.health_max,
                    'shield': enemy_structure.shield,
                    'shield_max': enemy_structure.shield_max,
                    'is_ready': enemy_structure.is_ready,
                    'build_progress': enemy_structure.build_progress,
                    'distance_to_our_base': enemy_structure.distance_to(self.start_location),
                    'is_tech_structure': self._is_tech_structure(enemy_structure.type_id),
                    'is_production_structure': self._is_production_structure(enemy_structure.type_id)
                }
                structures_info[structure_key] = structure_details
                structure_count += 1

            if structures_info:
                enemy_info['structures'] = structures_info

        except Exception as e:
            print(f"Error updating detailed enemy structure information: {e}")

        information.update(enemy_info)

        return information

    def _is_tech_structure(self, type_id):
        """Check whether a unit is a technology structure."""
        tech_structures = {
            UnitTypeId.ENGINEERINGBAY, UnitTypeId.ARMORY, UnitTypeId.FUSIONCORE,
            UnitTypeId.GHOSTACADEMY, UnitTypeId.TECHLAB
        }
        return type_id in tech_structures

    def _is_production_structure(self, type_id):
        """Check whether a unit is a production structure."""
        production_structures = {
            UnitTypeId.BARRACKS, UnitTypeId.FACTORY, UnitTypeId.STARPORT,
            UnitTypeId.COMMANDCENTER, UnitTypeId.ORBITALCOMMAND, UnitTypeId.PLANETARYFORTRESS
        }
        return type_id in production_structures

    def get_enemy_unity(self):

        if self.enemy_units:

            unit_names = ['enemy_' + str(unit.type_id) for unit in self.enemy_units]

            unit_type_amount = Counter(unit_names)

            return unit_type_amount
        else:

            return {}

    def get_enemy_structure(self):

        if self.enemy_structures:

            structure_names = ['enemy_' + str(structure.type_id) for structure in self.enemy_structures]

            structure_type_amount = Counter(structure_names)

        else:

            return {}

    def get_action_dict(self):
        action_dict = self.sc2dict.action_descriptions
        flat_dict = {}
        for key, value in action_dict.items():
            for inner_key, inner_value in value.items():
                flat_dict[inner_key] = inner_value
        self.action_dict = flat_dict

    async def handle_action_attack(self, action_id):

        await self.attack()

    async def handle_action_0(self):
        action_id = 0

        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'No Nexus available')

        for nexus in self.townhalls:
            if self.workers.amount + self.already_pending(UnitTypeId.PROBE) > 75:
                return self.record_failure(action_id, 'Too many probes (more than 75)')
            if self.supply_left <= 0:
                return self.record_failure(action_id, 'No supply left')
            if not self.can_afford(UnitTypeId.PROBE):
                return self.record_failure(action_id, 'Cannot afford Probe')
            if nexus.is_idle:
                nexus.train(UnitTypeId.PROBE)

                return

        return self.record_failure(action_id, 'All Nexus are busy')

    def record_failure(self, action, reason):
        self.temp_failure_list.append(f'Action failed: {self.action_dict[action]}, Reason: {reason}')

    async def warp_unit(self, action_id: int, unit_type: UnitTypeId, initial_reference_point: Point2,
                        required_buildings: List[UnitTypeId] = None,
                        max_attempts=5) -> None:
        """
        Warp a unit near a reference point.
        :param action_id: Action ID used for failure tracking.
        :param unit_type: Unit type to warp in.
        :param initial_reference_point: Initial warp reference point.
        :param max_attempts: Maximum number of placement attempts.
        :param required_buildings: Required technology structures.
        """

        if required_buildings:
            for building in required_buildings:
                if not self.structures(building).exists:
                    return self.record_failure(action_id, f'{building.name} not available')

        if unit_type not in {UnitTypeId.ZEALOT, UnitTypeId.STALKER, UnitTypeId.DARKTEMPLAR}:
            return self.record_failure(action_id, f"Unit type {unit_type.name} not warpable")

        if not self.structures(UnitTypeId.WARPGATE).exists:
            return self.record_failure(action_id, 'No Warp Gate available')

        idle_warpgates = self.structures(UnitTypeId.WARPGATE).ready.idle
        if not idle_warpgates:
            return self.record_failure(action_id, 'No idle Warp Gate available')
        warpgate = idle_warpgates.first
        warp_ability = getattr(AbilityId, f"WARPGATETRAIN_{unit_type.name}")

        abilities = await self.get_available_abilities(warpgate)

        if warp_ability not in abilities:
            return self.record_failure(action_id, f'Cannot train {unit_type.name} from Warp Gate')
        elif not self.can_afford(unit_type):
            return self.record_failure(action_id, f'Cannot afford {unit_type.name}')
        elif self.supply_left < self.calculate_supply_cost(unit_type):
            return self.record_failure(action_id, 'Not enough supply left')

        pylons = [p for p in self.structures(UnitTypeId.PYLON) if p.is_ready]
        pylons.sort(key=lambda p: p.distance_to(initial_reference_point))
        if not pylons:
            return self.record_failure(action_id, 'No powered pylons available')
        pylon = pylons[0]

        for distance in range(1, 6):
            pos = pylon.position.random_on_distance(distance)
            placement = await self.find_placement(warp_ability, pos, placement_step=1)
            if placement and placement.distance_to(pylon.position) <= 5:
                warpgate.warp_in(unit_type, placement)

        return self.record_failure(action_id, f"Couldn't find a placement near pylon for {unit_type.name}")

    async def train_from_gateway(self, action_id: int, unit_type: UnitTypeId,
                                 required_buildings: List[UnitTypeId] = None):
        """
        Train a unit from a Gateway.

        :param action_id: Action ID used for failure tracking.
        :param unit_type: Unit type to train.
        :param required_buildings: Additional required structures.
        :return: True on success, otherwise False.
        """

        if not self.structures(UnitTypeId.GATEWAY).exists:
            return self.record_failure(action_id, 'No Gateway available')

        required_supply = self.calculate_supply_cost(unit_type)

        if self.supply_left < required_supply:
            return self.record_failure(action_id, 'Not enough supply left')

        if required_buildings:
            for building in required_buildings:
                if not self.structures(building).exists:
                    return self.record_failure(action_id, f'{building.name} not available')

        idle_gates = [gate for gate in self.structures(UnitTypeId.GATEWAY) if gate.is_idle]
        if not idle_gates:

            gates_sorted_by_build_progress = sorted(self.structures(UnitTypeId.GATEWAY), key=lambda g: g.build_progress,
                                                    reverse=True)
            gate = gates_sorted_by_build_progress[0]
        else:
            gate = idle_gates[0]

        if not self.can_afford(unit_type):
            return self.record_failure(action_id, f'Cannot afford {unit_type.name}')

        gate.train(unit_type)

        return True

    async def produce_bg_unit(self, action_id: int, unit_type: UnitTypeId, required_buildings: List[UnitTypeId] = None):

        warp_gate_research_status = self.already_pending_upgrade(UpgradeId.WARPGATERESEARCH)

        if warp_gate_research_status == 1:

            if not self.structures(UnitTypeId.PYLON).ready.exists:
                return self.record_failure(action_id, 'No pylons available')

            proxy = self.structures(UnitTypeId.PYLON).closest_to(self.enemy_start_locations[0]).position

            await self.warp_unit(action_id, unit_type, proxy, required_buildings=required_buildings)
            print(f'Warping {unit_type.name}')
        else:

            success = await self.train_from_gateway(action_id, unit_type, required_buildings)
            if success:
                print(f'Training {unit_type.name}')

    async def handle_action_1(self):
        action_id = 1

        success = await self.produce_bg_unit(action_id, UnitTypeId.ZEALOT)
        if success:
            print('Training Zealot')

    async def handle_action_2(self):
        action_id = 2

        required_buildings = [UnitTypeId.CYBERNETICSCORE]

        success = await self.produce_bg_unit(action_id, UnitTypeId.ADEPT, required_buildings)
        if success:
            print('Training Adept')

    async def handle_action_3(self):
        action_id = 3

        required_buildings = [UnitTypeId.CYBERNETICSCORE]

        success = await self.produce_bg_unit(action_id, UnitTypeId.STALKER, required_buildings)
        if success:
            print('Training Stalker')

    async def handle_action_4(self):
        action_id = 4

        required_buildings = [UnitTypeId.CYBERNETICSCORE]

        success = await self.produce_bg_unit(action_id, UnitTypeId.SENTRY, required_buildings)
        if success:
            print('Training Sentry')

    async def handle_action_5(self):
        action_id = 5

        required_buildings = [UnitTypeId.TEMPLARARCHIVE]

        success = await self.produce_bg_unit(action_id, UnitTypeId.HIGHTEMPLAR, required_buildings)
        if success:
            print('Training High Templar')

    async def train_from_stargate(self, action_id: int, unit_type: UnitTypeId,
                                  required_buildings: List[UnitTypeId] = None,
                                  unit_limit: int = None) -> None:
        """
        Train one unit at a time from a Stargate.
        """

        if required_buildings:
            for building in required_buildings:
                if not self.structures(building).exists:
                    return self.record_failure(action_id, f'{building.name} not available')

        stargates = self.structures(UnitTypeId.STARGATE)
        if stargates.empty:
            return self.record_failure(action_id, 'No Stargate available')

        if stargates.filter(lambda facility: facility.is_idle).empty:

            print('Stargate is busy')
            return

        if unit_limit is not None and (self.units(unit_type).amount + self.already_pending(unit_type)) >= unit_limit:
            return self.record_failure(action_id, f'{unit_type.name} limit reached')

        if self.supply_left < self.calculate_supply_cost(unit_type):
            return self.record_failure(action_id, 'Not enough supply left')
        if not self.can_afford(unit_type):
            return self.record_failure(action_id, f'Cannot afford {unit_type.name}')

        idle_stargate = stargates.idle.first
        if idle_stargate:
            idle_stargate.train(unit_type)
            print(f'Training {unit_type.name}')
            return

        return self.record_failure(action_id, 'No idle Stargate available')

    async def handle_action_6(self):
        action_id = 6

        required_buildings = [UnitTypeId.DARKSHRINE]

        success = await self.produce_bg_unit(action_id, UnitTypeId.DARKTEMPLAR, required_buildings)
        if success:
            print('Training Dark Templar')

    async def handle_action_7(self):
        action_id = 7

        await self.train_from_stargate(action_id, UnitTypeId.VOIDRAY)

    async def handle_action_8(self):
        action_id = 8

        await self.train_from_stargate(action_id, UnitTypeId.CARRIER)

    async def handle_action_9(self):
        action_id = 9

        await self.train_from_stargate(action_id, UnitTypeId.TEMPEST)

    async def handle_action_10(self):
        action_id = 10

        await self.train_from_stargate(action_id, UnitTypeId.ORACLE, unit_limit=1)

    async def handle_action_11(self):
        action_id = 11

        await self.train_from_stargate(action_id, UnitTypeId.PHOENIX, unit_limit=4)

    async def handle_action_12(self):
        action_id = 12

        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'No Nexus available')

        if not self.structures(UnitTypeId.STARGATE).exists:
            return self.record_failure(action_id, 'No Stargate available')

        if not self.structures(UnitTypeId.FLEETBEACON).exists:
            return self.record_failure(action_id, 'Fleet Beacon not available')

        if self.supply_left < 10:
            return self.record_failure(action_id, 'Not enough supply left')

        nexuses = self.structures(UnitTypeId.NEXUS)
        abilities = await self.get_available_abilities(nexuses)

        if AbilityId.NEXUSTRAINMOTHERSHIP_MOTHERSHIP not in abilities:
            return self.record_failure(action_id, 'Cannot train Mothership')

        if not self.can_afford(UnitTypeId.MOTHERSHIP):
            return self.record_failure(action_id, 'Cannot afford Mothership')

        for base in self.townhalls:
            if base.is_idle:
                base.train(UnitTypeId.MOTHERSHIP)
                reward = 0.010
                print('Training Mothership')
                return reward

        return self.record_failure(action_id, 'No idle Nexus available')

    async def train_from_robotics(self, action_id: int, unit_type: UnitTypeId,
                                  required_buildings: List[UnitTypeId] = None,
                                  unit_limit: int = None) -> None:
        """
        Train one unit at a time from a Robotics Facility.
        """

        if required_buildings:
            for building in required_buildings:
                if not self.structures(building).exists:
                    return self.record_failure(action_id, f'{building.name} not available')

        robotics_facilities = self.structures(UnitTypeId.ROBOTICSFACILITY)
        if robotics_facilities.empty:
            return self.record_failure(action_id, 'No Robotics Facility available')

        if robotics_facilities.filter(lambda facility: facility.is_idle).empty:

            print('Robotics Facility is busy')
            return

        if unit_limit is not None and (self.units(unit_type).amount + self.already_pending(unit_type)) >= unit_limit:
            return self.record_failure(action_id, f'{unit_type.name} limit reached')

        if self.supply_left < self.calculate_supply_cost(unit_type):
            return self.record_failure(action_id, 'Not enough supply left')
        if not self.can_afford(unit_type):
            return self.record_failure(action_id, f'Cannot afford {unit_type.name}')

        idle_facility = robotics_facilities.idle.first
        if idle_facility:
            idle_facility.train(unit_type)
            print(f'Training {unit_type.name}')
            return

        return self.record_failure(action_id, 'No idle Robotics Facility available')

    async def handle_action_13(self):
        action_id = 13

        await self.train_from_robotics(action_id, UnitTypeId.OBSERVER, unit_limit=2)
        print('Training Observer')
        return

    async def handle_action_14(self):
        action_id = 14

        await self.train_from_robotics(action_id, UnitTypeId.IMMORTAL)
        print('Training Immortal')
        return

    async def handle_action_15(self):
        action_id = 15

        await self.train_from_robotics(action_id, UnitTypeId.WARPPRISM, unit_limit=1)
        print('Training Warp Prism')
        return

    async def handle_action_16(self):
        action_id = 16

        required_buildings = [UnitTypeId.ROBOTICSBAY]
        await self.train_from_robotics(action_id, UnitTypeId.COLOSSUS, required_buildings=required_buildings)
        print('Training Colossus')
        return

    async def handle_action_17(self):
        action_id = 17

        required_buildings = [UnitTypeId.DISRUPTOR]
        await self.train_from_robotics(action_id, UnitTypeId.DISRUPTOR, required_buildings=required_buildings,
                                       unit_limit=1)
        print('Training Disruptor')
        return

    async def handle_action_18(self):
        action_id = 18

        if self.units(UnitTypeId.HIGHTEMPLAR).exists:
            hts = self.units(UnitTypeId.HIGHTEMPLAR)
            for ht in hts:
                ht(AbilityId.MORPH_ARCHON)
            return 0.005

        elif self.units(UnitTypeId.DARKTEMPLAR).exists:
            dts = self.units(UnitTypeId.DARKTEMPLAR)
            for dt in dts:
                dt(AbilityId.MORPH_ARCHON)
            print('Merging Archon')
            return 0.005

        return self.record_failure(action_id, 'No High Templar or Dark Templar available for Archon morph')

    def find_best_base_for_pylon(self):
        """
        Find the best base for Pylon placement.
        """
        pylons = self.structures(UnitTypeId.PYLON)
        base_positions = [base.position for base in self.townhalls]
        pylon_positions = [pylon.position for pylon in pylons]
        base_pylon_counts = {}

        for base_pos in base_positions:
            count = sum(1 for pylon_pos in pylon_positions if base_pos.distance_to(pylon_pos) <= 10)
            base_pylon_counts[base_pos] = count

        sorted_bases = sorted(base_pylon_counts.keys(), key=lambda base: base_pylon_counts[base])
        return sorted_bases[0]

    @staticmethod
    def neighbors4(position, distance=1) -> Set[Point2]:
        """
        Returns the four adjacent positions (top, bottom, left, right) to the given position.

        Args:
        - position (Point2): The reference position.
        - distance (int, default=1): The distance from the reference position.

        Returns:
        - Set[Point2]: A set containing the four adjacent positions.
        """
        p = position
        d = distance
        return {
            Point2((p.x - d, p.y)),
            Point2((p.x + d, p.y)),
            Point2((p.x, p.y - d)),
            Point2((p.x, p.y + d))
        }

    def neighbors8(self, position, distance=1) -> Set[Point2]:
        """
        Returns the eight adjacent positions (top, bottom, left, right, and the four diagonals) to the given position.

        Args:
        - position (Point2): The reference position.
        - distance (int, default=1): The distance from the reference position.

        Returns:
        - Set[Point2]: A set containing the eight adjacent positions.
        """
        p = position
        d = distance

        direct_neighbors = self.neighbors4(position, distance)

        diagonal_neighbors = {
            Point2((p.x - d, p.y - d)),
            Point2((p.x - d, p.y + d)),
            Point2((p.x + d, p.y - d)),
            Point2((p.x + d, p.y + d))
        }
        return direct_neighbors | diagonal_neighbors

    def is_position_blocking_resources(self, position: Point2) -> bool:
        """
        Check whether a position blocks resource collection.
        """
        for resource in self.resources:
            if position.distance_to(resource.position) < MIN_DISTANCE:
                return True
        return False

    def is_position_valid_for_pylon(self, position: Point2) -> bool:
        """
        Check whether a position is valid for a Pylon.
        """

        for structure in self.structures:
            if position.distance_to(structure.position) < MIN_DISTANCE:
                return False

        for assimilator in self.structures(UnitTypeId.ASSIMILATOR):
            if position.distance_to(assimilator.position) < MIN_DISTANCE * 1.5:
                return False

        for resource in self.resources:
            if position.distance_to(resource.position) < MIN_DISTANCE:
                return False

        return True

    def find_optimal_pylon_position_for_base(self, base_position: Point2) -> Point2:
        """
        Find the optimal Pylon position.
        """

        candidate_positions = [base_position] + [pos for i in range(1, 15) for pos in
                                                 self.neighbors8(base_position, distance=i)]

        valid_positions = [pos for pos in candidate_positions if
                           not self.is_position_blocking_resources(pos) and self.is_position_valid_for_pylon(pos)]

        if not valid_positions:
            return base_position

        pylons = self.structures(UnitTypeId.PYLON)
        pylon_positions = [pylon.position for pylon in pylons]

        def score_position(pos):
            total_score = 0
            for pylon_pos in pylon_positions:
                distance = pos.distance_to(pylon_pos)

                score = 1 / (1 + math.exp(-distance + 5))
                total_score += score
            return total_score

        best_position = max(valid_positions, key=score_position)
        return best_position

    async def handle_action_19(self):
        action_id = 19

        if not (self.structures(UnitTypeId.NEXUS).exists and self.units(UnitTypeId.PROBE).exists):
            return self.record_failure(action_id, 'No Nexus or Probe available')

        supply_left_threshold, pending_pylon_threshold = 3, 2
        current_time = self.time_formatted

        if '06:00' <= current_time <= '07:00':
            supply_left_threshold, pending_pylon_threshold = 5, 4
        elif '06:00' <= current_time <= '08:00':
            supply_left_threshold, pending_pylon_threshold = 5, 3
        elif '08:00' <= current_time <= '10:00':
            supply_left_threshold, pending_pylon_threshold = 7, 4
        elif '10:00' <= current_time:
            supply_left_threshold, pending_pylon_threshold = 7, 4

        if (
                self.supply_left <= supply_left_threshold and
                self.already_pending(UnitTypeId.PYLON) <= pending_pylon_threshold and
                not self.supply_cap == 200
        ):
            if not self.can_afford(UnitTypeId.PYLON):
                return self.record_failure(action_id, 'Cannot afford Pylon')

            best_base = self.find_best_base_for_pylon()
            best_position = self.find_optimal_pylon_position_for_base(best_base)

            construction_result = await self.build(UnitTypeId.PYLON, near=best_position, placement_step=2)
            if not construction_result:
                return self.record_failure(action_id, 'Could not build Pylon for unknown reasons')

        else:

            return self.record_failure(action_id, 'Not the right conditions to build a Pylon')

    async def handle_action_20(self):
        action_id = 20

        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'No Nexus available')
        if not self.units(UnitTypeId.PROBE).exists:
            return self.record_failure(action_id, 'No Probe available')
        if not self.can_afford(UnitTypeId.ASSIMILATOR):
            return self.record_failure(action_id, 'Cannot afford Assimilator')
        if not self.vespene_geyser.exists:
            return self.record_failure(action_id, 'No Vespene Geyser available')
        if not self.structures(UnitTypeId.PYLON).exists:
            return self.record_failure(action_id, 'No Pylon available')

        building_assimilators_num = self.already_pending(UnitTypeId.ASSIMILATOR)

        if building_assimilators_num < 2:
            for nexus in self.townhalls:
                for vespene in self.vespene_geyser.closer_than(10, nexus):

                    moving_workers = [worker for worker in self.workers.closer_than(5, vespene) if worker.is_moving]
                    if moving_workers:
                        continue

                    if self.can_afford(UnitTypeId.ASSIMILATOR) and not self.structures(
                            UnitTypeId.ASSIMILATOR).closer_than(2, vespene):
                        await self.build(UnitTypeId.ASSIMILATOR, vespene)
                        print('Building Assimilator')
                        return
        else:
            print('Already building 2 Assimilators, waiting...')

    async def handle_action_21(self):
        action_id = 21

        if not self.units(UnitTypeId.PROBE).exists:
            return self.record_failure(action_id, 'No Probe available')
        if not self.can_afford(UnitTypeId.NEXUS):
            return self.record_failure(action_id, 'Cannot afford Nexus')
        if self.can_afford(UnitTypeId.NEXUS) and self.already_pending(UnitTypeId.NEXUS) <= 1 and self.structures(
                UnitTypeId.NEXUS).amount <= 8:
            await self.expand_now()
            print('Expanding the base')

    def find_optimal_building_position_for_base(self, base_position: Point2, building_type: UnitTypeId,
                                                max_distance=15) -> Point2:
        """
        Find the optimal structure position near a base.

        Args:
        - base_position (Point2): Base position used as the search center.
        - building_type (UnitTypeId): Structure type to build.
        - max_distance (int): Maximum search radius around the base.

        Returns:
        - Point2 or None: Best position, or None when no valid location exists.
        """

        pylons = self.structures(UnitTypeId.PYLON)

        if not pylons.exists:
            return self.original_building_position_search(base_position, building_type, max_distance)

        pylon_scores = {}
        for pylon in pylons:
            distance_to_base = pylon.distance_to(base_position)
            nearby_buildings = sum(1 for building in self.structures if building.distance_to(pylon) < 12)

            score = 1 * distance_to_base - 2 * nearby_buildings
            pylon_scores[pylon] = score

        best_pylon = max(pylon_scores, key=pylon_scores.get)

        best_position = self.original_building_position_search(best_pylon.position, building_type, max_distance)

        return best_position

    def original_building_position_search(self, base_position: Point2, building_type: UnitTypeId,
                                          max_distance=15) -> Point2:
        """
        Use the original structure-placement search.
        """
        enemy_base_position = self.enemy_start_locations[0]
        weight_own_base = 1.0
        weight_enemy_base = -2.0

        def compute_score(pos):
            distance_to_own_base = pos.distance_to(base_position)
            distance_to_enemy_base = pos.distance_to(enemy_base_position)

            score = weight_own_base * distance_to_own_base + weight_enemy_base / (distance_to_enemy_base + 1)
            return score

        for distance in range(1, max_distance + 1):
            candidate_positions = list(self.neighbors8(base_position, distance))
            valid_positions = [pos for pos in candidate_positions if self.is_position_valid_for_building(pos)]

            if valid_positions:
                best_position = max(valid_positions, key=compute_score)
                return best_position

        return None

    def is_position_valid_for_building(self, position: Point2) -> bool:
        """
        Check whether a position is valid for a structure.
        :param position: Position to evaluate.
        :return: True when construction is valid, otherwise False.
        """

        for structure in self.structures:

            if position.distance_to(structure.position) < BUILDING_DISTANCE + structure.radius:
                return False

        for resource in self.resources:

            if position.distance_to(resource.position) < MIN_DISTANCE + 2:
                return False

        critical_structures = self.structures({UnitTypeId.ASSIMILATOR, UnitTypeId.NEXUS})
        for critical_structure in critical_structures:

            if position.distance_to(
                    critical_structure.position) < CRITICAL_BUILDING_DISTANCE + critical_structure.radius:
                return False

        for building in self.structures(
                {UnitTypeId.STARGATE, UnitTypeId.ROBOTICSFACILITY, UnitTypeId.GATEWAY, ...}):

            if position.distance_to(building.position) < BUILDING_DISTANCE + building.radius:
                return False

        return True

    def find_best_base_for_building(self, building_type: UnitTypeId):
        base_positions = [base.position for base in self.townhalls]
        building_positions = [building.position for building in self.structures(building_type)]
        base_building_counts = {}
        for base_pos in base_positions:
            count = sum(1 for building_pos in building_positions if base_pos.distance_to(building_pos) <= 12)
            base_building_counts[base_pos] = count
        sorted_bases = sorted(base_building_counts.keys(), key=lambda base: base_building_counts[base])
        return sorted_bases[0] if sorted_bases else None

    async def handle_action_build_building(self, building_type: UnitTypeId, building_limits=None):
        """
        building_limits maps time ranges to the allowed number of structures.
        For example, {"00:00-02:00": 1} permits one structure before 02:00.

        """
        if not hasattr(self, 'state') or self.state.game_loop == 0:
            print("The game has ended; skipping construction")
            return

        if not self.can_afford(building_type):
            print(f"Cannot afford {building_type}")
            return

        current_time = self.time_formatted

        if building_limits:
            for time_range, limit in building_limits.items():
                start_time, end_time = time_range.split('-')
                if start_time <= current_time <= end_time:

                    current_building_count = self.structures(building_type).amount + self.already_pending(building_type)

                    if current_building_count >= limit:
                        print(f"Building limit reached for {building_type} for the time range {time_range}.")
                        return

        if building_type in {UnitTypeId.PHOTONCANNON, UnitTypeId.SHIELDBATTERY}:

            enemy_start = self.enemy_start_locations[0]
            outermost_base = min(self.townhalls, key=lambda base: base.distance_to(enemy_start))

            best_position = self.find_optimal_building_position_for_base(outermost_base.position, max_distance=8,
                                                                         building_type=building_type)

        else:
            best_base = self.find_best_base_for_building(building_type)
            if not best_base:
                print(f"No suitable base found for {building_type}")
                return

            best_position = self.find_optimal_building_position_for_base(best_base, building_type=building_type)

            if not best_position:
                print(f"No suitable position found near best base for {building_type}. Trying alternative bases.")

                for base in self.townhalls:
                    if base.position != best_base.position:
                        best_position = self.find_optimal_building_position_for_base(base.position,
                                                                                     building_type=building_type)
                        if best_position:
                            break

            if not best_position:
                print("Trying suboptimal positions...")

                adjusted_building_distance = 10
                best_position = self.find_optimal_building_position_for_base(best_base.position,
                                                                             max_distance=adjusted_building_distance,
                                                                             building_type=building_type)

                if not best_position:
                    print("Expanding search range...")
                    expanded_search_range = 30
                    best_position = self.find_optimal_building_position_for_base(best_base.position,
                                                                                 max_distance=expanded_search_range,
                                                                                 building_type=building_type)

        if not best_position:
            print(f"Still no suitable position found for {building_type}. Aborting.")
            return

        await self.build(building_type, near=best_position)
        print(f"Building {building_type}")

    async def handle_action_22(self):
        action_id = 22

        building_limits = {
            '00:00-03:00': 1,
            '03:00-05:00': 3,
            '05:00-07:00': 6,
            '07:00-10:00': 8,
        }
        if not self.structures(UnitTypeId.PYLON).exists:
            return self.record_failure(action_id, 'No Pylon available')
        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'No Nexus available')
        if not self.units(UnitTypeId.PROBE).exists:
            return self.record_failure(action_id, 'No Probe available')
        if not self.can_afford(UnitTypeId.GATEWAY):
            return self.record_failure(action_id, 'Cannot afford Gateway')
        await self.handle_action_build_building(UnitTypeId.GATEWAY, building_limits=building_limits)
        print('Building Gateway')

    async def handle_action_23(self):
        action_id = 23

        building_limits = {
            '00:00-05:00': 1,
            '05:00-20:00': 2,
        }
        if not self.structures(UnitTypeId.PYLON).exists:
            return self.record_failure(action_id, 'No Pylon available')
        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'No Nexus available')
        if not self.units(UnitTypeId.PROBE).exists:
            return self.record_failure(action_id, 'No Probe available')
        if not self.structures(UnitTypeId.GATEWAY).exists:
            return self.record_failure(action_id, 'No Gateway available')
        if not self.can_afford(UnitTypeId.CYBERNETICSCORE):
            return self.record_failure(action_id, 'Cannot afford Cybernetics Core')
        await self.handle_action_build_building(UnitTypeId.CYBERNETICSCORE, building_limits=building_limits)
        print('Building Cybernetics Core')

    async def handle_action_24(self):
        action_id = 24
        building_limits = None

        if not self.structures(UnitTypeId.PYLON).exists:
            return self.record_failure(action_id, 'No Pylon available')
        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'No Nexus available')
        if not self.units(UnitTypeId.PROBE).exists:
            return self.record_failure(action_id, 'No Probe available')
        if not self.can_afford(UnitTypeId.FORGE):
            return self.record_failure(action_id, 'Cannot afford Forge')
        await self.handle_action_build_building(UnitTypeId.FORGE, building_limits=building_limits)
        print('Building Forge')

    async def handle_action_25(self):
        action_id = 25

        if not self.structures(UnitTypeId.PYLON).exists:
            return self.record_failure(action_id, 'No Pylon available')
        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'No Nexus available')
        if not self.units(UnitTypeId.PROBE).exists:
            return self.record_failure(action_id, 'No Probe available')
        if not self.structures(UnitTypeId.CYBERNETICSCORE).exists:
            return self.record_failure(action_id, 'No Cybernetics Core available')
        if not self.can_afford(UnitTypeId.TWILIGHTCOUNCIL):
            return self.record_failure(action_id, 'Cannot afford Twilight Council')
        await self.handle_action_build_building(UnitTypeId.TWILIGHTCOUNCIL)
        print('Building Twilight Council')

    async def handle_action_26(self):
        action_id = 26

        building_limits = {
            '00:00-05:00': 1,
            '05:00-10:00': 2,
            '10:00-15:00': 3,
        }
        if not self.structures(UnitTypeId.PYLON).exists:
            return self.record_failure(action_id, 'No Pylon available')
        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'No Nexus available')
        if not self.units(UnitTypeId.PROBE).exists:
            return self.record_failure(action_id, 'No Probe available')
        if not self.structures(UnitTypeId.CYBERNETICSCORE).exists:
            return self.record_failure(action_id, 'No Cybernetics Core available')
        if not self.can_afford(UnitTypeId.ROBOTICSFACILITY):
            return self.record_failure(action_id, 'Cannot afford Robotics Facility')
        await self.handle_action_build_building(UnitTypeId.ROBOTICSFACILITY, building_limits=building_limits)
        print('Building Robotics Facility')

    async def handle_action_27(self):
        action_id = 27

        buildind_limits = {
            '00:00-05:00': 1,
            '05:00-10:00': 3,
            '10:00-15:00': 4,
        }
        if not self.structures(UnitTypeId.PYLON).exists:
            return self.record_failure(action_id, 'No Pylon available')
        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'No Nexus available')
        if not self.units(UnitTypeId.PROBE).exists:
            return self.record_failure(action_id, 'No Probe available')
        if not self.structures(UnitTypeId.CYBERNETICSCORE).exists:
            return self.record_failure(action_id, 'No Cybernetics Core available')
        if not self.can_afford(UnitTypeId.STARGATE):
            return self.record_failure(action_id, 'Cannot afford Stargate')
        await self.handle_action_build_building(UnitTypeId.STARGATE, building_limits=buildind_limits)
        print('Building Stargate')

    async def handle_action_28(self):
        action_id = 28

        if not self.structures(UnitTypeId.PYLON).exists:
            return self.record_failure(action_id, 'No Pylon available')
        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'No Nexus available')
        if not self.units(UnitTypeId.PROBE).exists:
            return self.record_failure(action_id, 'No Probe available')
        if not self.structures(UnitTypeId.CYBERNETICSCORE).exists:
            return self.record_failure(action_id, 'No Cybernetics Core available')
        if not self.structures(UnitTypeId.TWILIGHTCOUNCIL).exists:
            return self.record_failure(action_id, 'No Twilight Council available')
        if not self.can_afford(UnitTypeId.TEMPLARARCHIVE):
            return self.record_failure(action_id, 'Cannot afford Templar Archive')
        await self.handle_action_build_building(UnitTypeId.TEMPLARARCHIVE)
        print('Building Templar Archives')

    async def handle_action_29(self):
        action_id = 29

        if not self.structures(UnitTypeId.PYLON).exists:
            return self.record_failure(action_id, 'No Pylon available')
        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'No Nexus available')
        if not self.units(UnitTypeId.PROBE).exists:
            return self.record_failure(action_id, 'No Probe available')
        if not self.structures(UnitTypeId.CYBERNETICSCORE).exists:
            return self.record_failure(action_id, 'No Cybernetics Core available')
        if not self.structures(UnitTypeId.TWILIGHTCOUNCIL).exists:
            return self.record_failure(action_id, 'No Twilight Council available')
        if not self.can_afford(UnitTypeId.DARKSHRINE):
            return self.record_failure(action_id, 'Cannot afford Dark Shrine')
        await self.handle_action_build_building(UnitTypeId.DARKSHRINE)

        print('Building Dark Shrine')

    async def handle_action_30(self):
        action_id = 30

        if not self.structures(UnitTypeId.PYLON).exists:
            return self.record_failure(action_id, 'No Pylon available')
        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'No Nexus available')
        if not self.units(UnitTypeId.PROBE).exists:
            return self.record_failure(action_id, 'No Probe available')
        if not self.structures(UnitTypeId.CYBERNETICSCORE).exists:
            return self.record_failure(action_id, 'No Cybernetics Core available')
        if not self.structures(UnitTypeId.ROBOTICSFACILITY).exists:
            return self.record_failure(action_id, 'No Robotics Facility available')
        if not self.can_afford(UnitTypeId.ROBOTICSBAY):
            return self.record_failure(action_id, 'Cannot afford Robotics Bay')
        await self.handle_action_build_building(UnitTypeId.ROBOTICSBAY)
        print('Building Robotics Bay')

    async def handle_action_31(self):
        action_id = 31

        if not self.structures(UnitTypeId.PYLON).exists:
            return self.record_failure(action_id, 'No Pylon available')
        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'No Nexus available')
        if not self.units(UnitTypeId.PROBE).exists:
            return self.record_failure(action_id, 'No Probe available')
        if not self.structures(UnitTypeId.CYBERNETICSCORE).exists:
            return self.record_failure(action_id, 'No Cybernetics Core available')
        if not self.structures(UnitTypeId.STARGATE).exists:
            return self.record_failure(action_id, 'No Stargate available')
        if not self.can_afford(UnitTypeId.FLEETBEACON):
            return self.record_failure(action_id, 'Cannot afford Fleet Beacon')
        await self.handle_action_build_building(UnitTypeId.FLEETBEACON)
        print('Building Fleet Beacon')

    async def handle_action_32(self):
        action_id = 32

        building_limits = {
            '00:00-05:00': 1,
            '05:00-10:00': 3,
            '10:00-15:00': 5,
            '15:00-20:00': 7,
        }
        if not self.structures(UnitTypeId.PYLON).exists:
            return self.record_failure(action_id, 'No Pylon available')
        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'No Nexus available')
        if not self.units(UnitTypeId.PROBE).exists:
            return self.record_failure(action_id, 'No Probe available')
        if not self.structures(UnitTypeId.CYBERNETICSCORE).exists:
            return self.record_failure(action_id, 'No Cybernetics Core available')
        if not self.structures(UnitTypeId.FORGE).exists:
            return self.record_failure(action_id, 'No Forge')
        if self.structures(UnitTypeId.PHOTONCANNON).amount + self.already_pending(UnitTypeId.PHOTONCANNON) > 3:
            return self.record_failure(action_id, 'Photon Cannon limit reached')
        if not self.can_afford(UnitTypeId.PHOTONCANNON):
            return self.record_failure(action_id, 'Cannot afford  Photon Cannon')
        await self.handle_action_build_building(UnitTypeId.PHOTONCANNON, building_limits=building_limits)
        print('Building Photon Cannon')

    async def handle_action_33(self):
        action_id = 33

        building_limits = {
            '00:00-05:00': 1,
            '05:00-10:00': 3,
            '10:00-15:00': 5,
            '15:00-20:00': 7,
        }
        if not self.structures(UnitTypeId.PYLON).exists:
            return self.record_failure(action_id, 'No Pylon available')
        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'No Nexus available')
        if not self.units(UnitTypeId.PROBE).exists:
            return self.record_failure(action_id, 'No Probe available')
        if not self.structures(UnitTypeId.CYBERNETICSCORE).exists:
            return self.record_failure(action_id, 'No Cybernetics Core available')
        if not self.structures(UnitTypeId.FORGE).exists:
            return self.record_failure(action_id, 'No Forge')
        if self.structures(UnitTypeId.SHIELDBATTERY).amount + self.already_pending(UnitTypeId.SHIELDBATTERY) > 3:
            return self.record_failure(action_id, 'Shield Battery limit reached')
        if not self.can_afford(UnitTypeId.SHIELDBATTERY):
            return self.record_failure(action_id, 'Cannot afford Shield Battery')
        await self.handle_action_build_building(UnitTypeId.SHIELDBATTERY, building_limits=building_limits)
        print('Building Shield Battery')

    async def handle_action_34(self):
        action_id = 34

        if not self.structures(UnitTypeId.PYLON).exists:
            return self.record_failure(action_id, 'Pylon does not exist')

        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'Nexus does not exist')

        if not self.structures(UnitTypeId.CYBERNETICSCORE).exists:
            return self.record_failure(action_id, 'Cybernetics Core does not exist')

        by = self.structures(UnitTypeId.CYBERNETICSCORE).random
        abilities = await self.get_available_abilities(by)
        if self.structures(UnitTypeId.CYBERNETICSCORE).ready and self.already_pending(UpgradeId.WARPGATERESEARCH) == 0:
            if self.can_afford(UpgradeId.WARPGATERESEARCH) and by.is_idle and AbilityId.RESEARCH_WARPGATE in abilities:
                by.research(UpgradeId.WARPGATERESEARCH)
                print('Researching Warp Gate')
                return 0.010

        return self.record_failure(action_id, 'Cannot afford Warp Gate research or Cybernetics Core is not idle')

    async def research_upgrade(self, action_id, upgrade_id, ability_id, description):

        if not self.structures(UnitTypeId.PYLON).exists:
            return self.record_failure(action_id, 'Pylon does not exist')

        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'Nexus does not exist')

        if not self.structures(UnitTypeId.CYBERNETICSCORE).exists:
            return self.record_failure(action_id, 'Cybernetics Core does not exist')

        by = self.structures(UnitTypeId.CYBERNETICSCORE).random
        abilities = await self.get_available_abilities(by)

        if self.structures(UnitTypeId.CYBERNETICSCORE).ready and self.already_pending(upgrade_id) == 0:
            if self.can_afford(upgrade_id) and by.is_idle and ability_id in abilities:
                by.research(upgrade_id)
                print(description)

        return self.record_failure(action_id, f'Cannot afford {description} or Cybernetics Core is not idle')

    async def handle_action_35(self):
        upgrade_id = UpgradeId.PROTOSSAIRWEAPONSLEVEL1
        ability_id = AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL1
        description = 'Research air weapons level 1'
        return await self.research_upgrade(35, upgrade_id, ability_id, description)

    async def handle_action_36(self):
        upgrade_id = UpgradeId.PROTOSSAIRWEAPONSLEVEL2
        ability_id = AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL2
        description = 'Research air weapons level 2'
        return await self.research_upgrade(36, upgrade_id, ability_id, description)

    async def handle_action_37(self):
        upgrade_id = UpgradeId.PROTOSSAIRWEAPONSLEVEL3
        ability_id = AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL3
        description = 'Research air weapons level 3'
        return await self.research_upgrade(37, upgrade_id, ability_id, description)

    async def handle_action_38(self):
        upgrade_id = UpgradeId.PROTOSSAIRARMORSLEVEL1
        ability_id = AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRARMORLEVEL1
        description = 'Research air armor level 1'
        return await self.research_upgrade(38, upgrade_id, ability_id, description)

    async def handle_action_39(self):
        upgrade_id = UpgradeId.PROTOSSAIRARMORSLEVEL2
        ability_id = AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRARMORLEVEL2
        description = 'Research air armor level 2'
        return await self.research_upgrade(39, upgrade_id, ability_id, description)

    async def handle_action_40(self):
        upgrade_id = UpgradeId.PROTOSSAIRARMORSLEVEL3
        ability_id = AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRARMORLEVEL3
        description = 'Research air armor level 3'
        return await self.research_upgrade(40, upgrade_id, ability_id, description)

    async def research_twilight_upgrade(self, action_id, upgrade_id, ability_id, description):

        if not self.structures(UnitTypeId.TWILIGHTCOUNCIL).exists:
            return self.record_failure(action_id, 'Twilight Council does not exist')

        vc = self.structures(UnitTypeId.TWILIGHTCOUNCIL).random
        abilities = await self.get_available_abilities(vc)

        if self.structures(UnitTypeId.TWILIGHTCOUNCIL).ready and self.already_pending(upgrade_id) == 0:
            if self.can_afford(upgrade_id) and self.structures(
                    UnitTypeId.TWILIGHTCOUNCIL).idle.exists and ability_id in abilities:
                vc = self.structures(UnitTypeId.TWILIGHTCOUNCIL).idle.random
                vc.research(upgrade_id)
                print(description)

        return self.record_failure(action_id, f'Cannot afford {description} or Twilight Council is not idle')

    async def handle_action_41(self):
        upgrade_id = UpgradeId.ADEPTPIERCINGATTACK
        ability_id = AbilityId.RESEARCH_ADEPTRESONATINGGLAIVES
        description = 'Research Resonating Glaives'
        return await self.research_twilight_upgrade(41, upgrade_id, ability_id, description)

    async def handle_action_42(self):
        upgrade_id = UpgradeId.BLINKTECH
        ability_id = AbilityId.RESEARCH_BLINK
        description = 'Research Blink'
        return await self.research_twilight_upgrade(42, upgrade_id, ability_id, description)

    async def handle_action_43(self):
        upgrade_id = UpgradeId.CHARGE
        ability_id = AbilityId.RESEARCH_CHARGE
        description = 'Research Charge'
        return await self.research_twilight_upgrade(43, upgrade_id, ability_id, description)

    async def research_forge_upgrade(self, action_id, upgrade_id, ability_id, description):

        if not (self.structures(UnitTypeId.PYLON).exists
                and self.units(UnitTypeId.PROBE).exists
                and self.structures(UnitTypeId.NEXUS).exists
                and self.structures(UnitTypeId.FORGE).exists):
            return self.record_failure(action_id, 'Required structures or units do not exist')

        bf = self.structures(UnitTypeId.FORGE).random
        abilities = await self.get_available_abilities(bf)

        if self.structures(UnitTypeId.FORGE).ready and self.already_pending(upgrade_id) == 0:
            if self.can_afford(upgrade_id) and self.structures(
                    UnitTypeId.FORGE).idle.exists and ability_id in abilities:
                bf = self.structures(UnitTypeId.FORGE).idle.random
                bf.research(upgrade_id)
                print(description)

        return self.record_failure(action_id, f'Cannot afford {description} or Forge is not idle')

    async def handle_action_44(self):
        return await self.research_forge_upgrade(44, UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1,
                                                 AbilityId.FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL1, 'Research ground weapons level 1')

    async def handle_action_45(self):
        return await self.research_forge_upgrade(45, UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2,
                                                 AbilityId.FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL2, 'Research ground weapons level 2')

    async def handle_action_46(self):
        return await self.research_forge_upgrade(46, UpgradeId.PROTOSSGROUNDWEAPONSLEVEL3,
                                                 AbilityId.FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL3, 'Research ground weapons level 3')

    async def handle_action_47(self):
        return await self.research_forge_upgrade(47, UpgradeId.PROTOSSGROUNDARMORSLEVEL1,
                                                 AbilityId.FORGERESEARCH_PROTOSSGROUNDARMORLEVEL1, 'Research ground armor level 1')

    async def handle_action_48(self):
        return await self.research_forge_upgrade(48, UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2,
                                                 AbilityId.FORGERESEARCH_PROTOSSGROUNDWEAPONSLEVEL2, 'Research ground armor level 2')

    async def handle_action_49(self):
        return await self.research_forge_upgrade(49, UpgradeId.PROTOSSGROUNDARMORSLEVEL3,
                                                 AbilityId.FORGERESEARCH_PROTOSSGROUNDARMORLEVEL3, 'Research ground armor level 3')

    async def handle_action_50(self):
        return await self.research_forge_upgrade(50, UpgradeId.PROTOSSSHIELDSLEVEL1,
                                                 AbilityId.FORGERESEARCH_PROTOSSSHIELDSLEVEL1, 'Research shields level 1')

    async def handle_action_51(self):
        return await self.research_forge_upgrade(51, UpgradeId.PROTOSSSHIELDSLEVEL2,
                                                 AbilityId.FORGERESEARCH_PROTOSSSHIELDSLEVEL2, 'Research shields level 2')

    async def handle_action_52(self):
        return await self.research_forge_upgrade(52, UpgradeId.PROTOSSSHIELDSLEVEL3,
                                                 AbilityId.FORGERESEARCH_PROTOSSSHIELDSLEVEL3, 'Research shields level 3')

    async def research_roboticsbay_upgrade(self, action_id, upgrade_id, ability_id, description):

        if not (self.structures(UnitTypeId.PYLON).exists
                and self.units(UnitTypeId.PROBE).exists
                and self.structures(UnitTypeId.NEXUS).exists
                and self.structures(UnitTypeId.ROBOTICSFACILITY).exists
                and self.structures(UnitTypeId.ROBOTICSBAY).exists):
            return self.record_failure(action_id, 'Required structures or units do not exist')

        vb = self.structures(UnitTypeId.ROBOTICSBAY).random
        abilities = await self.get_available_abilities(vb)

        if self.structures(UnitTypeId.ROBOTICSBAY).ready and self.already_pending(upgrade_id) == 0:
            if self.can_afford(upgrade_id) and self.structures(
                    UnitTypeId.ROBOTICSBAY).idle.exists and ability_id in abilities:
                vb = self.structures(UnitTypeId.ROBOTICSBAY).idle.random
                vb.research(upgrade_id)
                print(description)

        return self.record_failure(action_id, f'Cannot afford {description} or RoboticsBay is not idle')

    async def handle_action_53(self):
        return await self.research_roboticsbay_upgrade(53, UpgradeId.EXTENDEDTHERMALLANCE,
                                                       AbilityId.RESEARCH_EXTENDEDTHERMALLANCE, 'Research Extended Thermal Lance')

    async def handle_action_54(self):
        return await self.research_roboticsbay_upgrade(54, UpgradeId.GRAVITICDRIVE, AbilityId.RESEARCH_GRAVITICDRIVE,
                                                       'Research Gravitic Drive')

    async def handle_action_55(self):
        return await self.research_roboticsbay_upgrade(55, UpgradeId.OBSERVERGRAVITICBOOSTER,
                                                       AbilityId.RESEARCH_GRAVITICBOOSTER, 'Research Gravitic Boosters')

    async def handle_action_56(self):
        action_id = 56

        if not self.structures(UnitTypeId.PYLON).exists:
            return self.record_failure(action_id, 'Pylon does not exist')

        if not self.units(UnitTypeId.PROBE).exists:
            return self.record_failure(action_id, 'Probe does not exist')

        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'Nexus does not exist')

        if not self.structures(UnitTypeId.TEMPLARARCHIVE).exists:
            return self.record_failure(action_id, 'Templar archive does not exist')

        vt = self.structures(UnitTypeId.TEMPLARARCHIVE).random
        abilities = await self.get_available_abilities(vt)
        upgrade_id = UpgradeId.PSISTORMTECH
        description = 'psi storm'
        ability_id = AbilityId.RESEARCH_PSISTORM

        if not self.structures(UnitTypeId.TEMPLARARCHIVE).ready:
            return self.record_failure(action_id, 'Templar archive is not ready')

        if self.already_pending(upgrade_id) != 0:
            return self.record_failure(action_id, f'Upgrade {description} is already pending')

        if not self.can_afford(upgrade_id):
            return self.record_failure(action_id, f'Cannot afford {description}')

        if not vt.is_idle:
            return self.record_failure(action_id, 'Templar archive is not idle')

        if ability_id not in abilities:
            return self.record_failure(action_id, f'Ability {description} is not available')

        vt.research(upgrade_id)
        print(description)
        return 0.010

    async def research_fleetbeacon_upgrade(self, action_id, upgrade_id, ability_id, description):

        if not self.structures(UnitTypeId.PYLON).exists:
            return self.record_failure(action_id, 'Pylon does not exist')

        if not self.units(UnitTypeId.PROBE).exists:
            return self.record_failure(action_id, 'Probe does not exist')

        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'Nexus does not exist')

        if not self.structures(UnitTypeId.STARGATE).exists:
            return self.record_failure(action_id, 'Stargate does not exist')

        if not self.structures(UnitTypeId.FLEETBEACON).exists:
            return self.record_failure(action_id, 'FleetBeacon does not exist')

        vf = self.structures(UnitTypeId.FLEETBEACON).random
        abilities = await self.get_available_abilities(vf)

        if not self.structures(UnitTypeId.FLEETBEACON).ready:
            return self.record_failure(action_id, 'FleetBeacon is not ready')

        if self.already_pending(upgrade_id) != 0:
            return self.record_failure(action_id, f'Upgrade {description} is already pending')

        if not self.can_afford(upgrade_id):
            return self.record_failure(action_id, f'Cannot afford {description}')

        if not self.structures(UnitTypeId.FLEETBEACON).idle.exists:
            return self.record_failure(action_id, 'FleetBeacon is not idle')

        if ability_id not in abilities:
            return self.record_failure(action_id, f'Ability {description} is not available')
        vf = self.structures(UnitTypeId.FLEETBEACON).idle.random
        vf.research(upgrade_id)
        print(description)

    async def handle_action_57(self):
        return await self.research_fleetbeacon_upgrade(57, UpgradeId.VOIDRAYSPEEDUPGRADE,
                                                       AbilityId.FLEETBEACONRESEARCH_RESEARCHVOIDRAYSPEEDUPGRADE,
                                                       'Research Flux Vanes')

    async def handle_action_58(self):
        return await self.research_fleetbeacon_upgrade(58, UpgradeId.PHOENIXRANGEUPGRADE,
                                                       AbilityId.RESEARCH_PHOENIXANIONPULSECRYSTALS, 'Research Anion Pulse-Crystals')

    async def handle_action_59(self):

        return

    async def handle_scouting(self, unit_type, action_id):
        """
        Scout with the specified unit type.

        :param unit_type: Unit type used for scouting.
        :param action_id: Action ID used for failure tracking.
        """

        if self.iteration - self.last_scouting_iteration < SCOUTING_INTERVAL:
            return

        if not self.units(unit_type).exists:
            return self.record_failure(action_id, f'No {unit_type} available for scouting')

        if self.units(unit_type).idle.exists:
            scout_unit = random.choice(self.units(unit_type).idle)
        else:
            scout_unit = random.choice(self.units(unit_type))

        target_location = self.enemy_start_locations[0]

        scout_unit.attack(target_location)

        self.last_scouting_iteration = self.iteration

        print(f'{unit_type} scouting towards {target_location}')

    async def handle_action_60(self):
        return await self.handle_scouting(UnitTypeId.PROBE, 60)

    async def handle_action_61(self):
        return await self.handle_scouting(UnitTypeId.OBSERVER, 61)

    async def handle_action_62(self):
        return await self.handle_scouting(UnitTypeId.ZEALOT, 62)

    async def handle_action_63(self):
        return await self.handle_scouting(UnitTypeId.PHOENIX, 63)

    async def handle_action_64(self):
        return await self.handle_action_attack(64)

    async def handle_action_65(self):
        action_id = 65

        if self.supply_army <= 0:
            return self.record_failure(action_id, 'Army supply is 0 or less')

        retreating_units = {
            UnitTypeId.ZEALOT, UnitTypeId.ARCHON, UnitTypeId.STALKER,
            UnitTypeId.ADEPT, UnitTypeId.HIGHTEMPLAR, UnitTypeId.DARKTEMPLAR,
            UnitTypeId.OBSERVER, UnitTypeId.PHOENIX, UnitTypeId.CARRIER, UnitTypeId.VOIDRAY,
            UnitTypeId.TEMPEST, UnitTypeId.ORACLE, UnitTypeId.COLOSSUS,
            UnitTypeId.DISRUPTOR, UnitTypeId.WARPPRISM, UnitTypeId.IMMORTAL,
            UnitTypeId.CHANGELINGZEALOT
        }

        units_to_retreat = self.units.of_type(retreating_units)
        if not units_to_retreat.exists:
            return self.record_failure(action_id, 'No units available to retreat')

        try:

            closest_nexus = units_to_retreat.closest_to(self.enemy_start_locations[0])
            for retreat_unit in units_to_retreat:
                retreat_unit.move(closest_nexus)
            print('Retreat initiated')
        except Exception as e:
            return self.record_failure(action_id, f'Retreat failed due to: {e}')

    async def apply_chronoboost(self, action_id, target_type, supply_requirement, description):

        if not self.structures(target_type).exists:
            return self.record_failure(action_id, f'{target_type} does not exist')

        if not self.structures(UnitTypeId.NEXUS).exists:
            return self.record_failure(action_id, 'Nexus does not exist')

        target = self.structures(target_type).random

        if target.is_idle or target.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
            return self.record_failure(action_id, f'{description} is idle or already chronoboosted')

        if self.supply_left < supply_requirement:
            return self.record_failure(action_id, f'Not enough supply for {description}')

        nexuses = self.structures(UnitTypeId.NEXUS)
        abilities = await self.get_available_abilities(nexuses)
        for loop_nexus, abilities_nexus in zip(nexuses, abilities):
            if AbilityId.EFFECT_CHRONOBOOSTENERGYCOST in abilities_nexus:
                loop_nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, target)
                print(f'Chronoboost applied to {description}')

        return self.record_failure(action_id, f'No available Nexus to chronoboost {description}')

    async def handle_action_66(self):
        print("chronoboost nexus")
        return await self.apply_chronoboost(66, UnitTypeId.NEXUS, 2, 'base')

    async def handle_action_67(self):
        print("chronoboost cybernetics core")
        return await self.apply_chronoboost(67, UnitTypeId.CYBERNETICSCORE, 0, 'cybernetics core (by)')

    async def handle_action_68(self):
        print("chronoboost twilight council")
        return await self.apply_chronoboost(68, UnitTypeId.TWILIGHTCOUNCIL, 4, 'twilight council (vc)')

    async def handle_action_69(self):
        print("chronoboost stargate")
        return await self.apply_chronoboost(69, UnitTypeId.STARGATE, 4, 'stargate (vs)')

    async def handle_action_70(self):
        print("chronoboost forge")
        return await self.apply_chronoboost(70, UnitTypeId.FORGE, 4, 'forge (bf)')

    async def handle_action_71(self):
        print("empty action")
        pass

    async def handle_action_72(self):
        """Assign idle workers to mineral and gas collection."""
        action_id = 72

        idle_workers = self.workers.idle
        if not idle_workers.exists:
            print(f"Action {action_id}: no idle workers")
            return

        print(f"Action {action_id}: found {idle_workers.amount} idle workers")

        ready_assimilators = self.structures(UnitTypeId.ASSIMILATOR).ready
        for assimilator in ready_assimilators:
            if not idle_workers.exists:
                break

            gas_workers = self.workers.closer_than(2, assimilator).amount

            while gas_workers < 3 and idle_workers.exists:
                worker = idle_workers.closest_to(assimilator)
                worker.gather(assimilator)
                print("Assigned a worker to gas")

                idle_workers = self.workers.idle
                gas_workers += 1

        minerals = self.mineral_field
        if idle_workers.exists and minerals.exists:
            for worker in idle_workers:
                closest_mineral = minerals.closest_to(worker)
                worker.gather(closest_mineral)
                print("Assigned a worker to minerals")

        print(f"Action {action_id} completed")

    async def handle_action_73(self):
        """Handle kiting tactics (action_id=73)."""

        attack_force = []

        stalkers = self.units(UnitTypeId.STALKER).ready
        voidrays = self.units(UnitTypeId.VOIDRAY).ready
        phoenixes = self.units(UnitTypeId.PHOENIX).ready

        if stalkers.exists:
            attack_force.extend(stalkers[:4])
        if voidrays.exists:
            attack_force.extend(voidrays[:2])
        if phoenixes.exists:
            attack_force.extend(phoenixes[:3])

        if attack_force:
            await self._execute_kiting_strategy(attack_force)
        else:
            await self.attack()

    async def _execute_kiting_strategy(self, attack_force):
        """Execute kiting tactics."""
        if not self.enemy_units.exists:

            await self.attack()
            return

        for unit in attack_force:
            closest_enemy = self.enemy_units.closest_to(unit)
            if closest_enemy:
                distance = unit.distance_to(closest_enemy)

                if distance < 7:
                    retreat_pos = unit.position.towards(self.start_location, 8)
                    unit.move(retreat_pos)
                else:
                    unit.attack(closest_enemy)

    async def handle_action_74(self):
        """Handle flanking tactics (action_id=74)."""

        attack_force = []

        zealots = self.units(UnitTypeId.ZEALOT).ready
        stalkers = self.units(UnitTypeId.STALKER).ready
        adepts = self.units(UnitTypeId.ADEPT).ready

        if zealots.exists:
            attack_force.extend(zealots[:3])
        if stalkers.exists:
            attack_force.extend(stalkers[:3])
        if adepts.exists:
            attack_force.extend(adepts[:2])

        if attack_force:
            await self._execute_flanking_strategy(attack_force)
        else:
            await self.attack()

    async def _execute_flanking_strategy(self, attack_force):
        """Execute flanking tactics."""
        if len(attack_force) < 6:
            await self._execute_default_attack_strategy(attack_force)
            return

        half = len(attack_force) // 2
        main_force = attack_force[:half]
        flank_force = attack_force[half:]

        for unit in main_force:
            unit.attack(self.enemy_start_locations[0])

        flank_pos = self._calculate_flank_position()
        for unit in flank_force:
            unit.attack(flank_pos)

    def _calculate_flank_position(self):
        """Calculate a flanking position."""
        enemy_base = self.enemy_start_locations[0]

        if hasattr(self, 'expansion_locations_list') and self.expansion_locations_list:
            closest_expansion = min(self.expansion_locations_list,
                                    key=lambda p: p.distance_to(enemy_base))
            return closest_expansion
        else:

            return enemy_base.offset((15, 15))

    async def handle_action_75(self):
        """Handle harassment tactics (action_id=75)."""

        attack_force = []

        phoenixes = self.units(UnitTypeId.PHOENIX).ready
        oracles = self.units(UnitTypeId.ORACLE).ready
        adepts = self.units(UnitTypeId.ADEPT).ready

        if phoenixes.exists:
            attack_force.extend(phoenixes[:2])
        if oracles.exists:
            attack_force.extend(oracles[:1])
        if adepts.exists:
            attack_force.extend(adepts[:2])

        if attack_force:
            await self._execute_hit_and_run_strategy(attack_force)
        else:
            await self.attack()

    def _find_harass_target(self):
        """Find a harassment target."""

        if self.enemy_units.of_type([UnitTypeId.SCV, UnitTypeId.DRONE, UnitTypeId.PROBE]).exists:
            return self.enemy_units.of_type([UnitTypeId.SCV, UnitTypeId.DRONE, UnitTypeId.PROBE]).first

        if self.enemy_structures.of_type([UnitTypeId.ASSIMILATOR, UnitTypeId.EXTRACTOR, UnitTypeId.REFINERY]).exists:
            return self.enemy_structures.of_type(
                [UnitTypeId.ASSIMILATOR, UnitTypeId.EXTRACTOR, UnitTypeId.REFINERY]).first

        if self.enemy_structures.of_type([UnitTypeId.GATEWAY, UnitTypeId.BARRACKS, UnitTypeId.SPAWNINGPOOL]).exists:
            return self.enemy_structures.of_type(
                [UnitTypeId.GATEWAY, UnitTypeId.BARRACKS, UnitTypeId.SPAWNINGPOOL]).first

        return self.enemy_start_locations[0]

    async def _execute_hit_and_run_strategy(self, attack_force):
        """Execute harassment tactics."""

        target = self._find_harass_target()

        if target:
            for unit in attack_force:
                unit.attack(target)

            if not hasattr(self, '_harass_retreat_time'):
                self._harass_retreat_time = self.time + 5.0

            if self.time > self._harass_retreat_time:
                for unit in attack_force:
                    closest_base = self.townhalls.closest_to(unit)
                    unit.move(closest_base.position)
                del self._harass_retreat_time

    async def _execute_deathball_strategy(self, attack_force):
        """Execute an advance."""

        center = self.game_info.map_center
        for unit in attack_force:
            unit.attack(center)

    async def handle_action_76(self):
        """Handle advance tactics (action_id=76)."""

        attack_force = []

        combat_units = [UnitTypeId.ZEALOT, UnitTypeId.STALKER, UnitTypeId.ADEPT,
                        UnitTypeId.IMMORTAL, UnitTypeId.COLOSSUS, UnitTypeId.ARCHON]

        for unit_type in combat_units:
            units = self.units(unit_type).ready
            if units.exists:
                attack_force.extend(units[:6])

        if attack_force:
            await self._execute_deathball_strategy(attack_force)
        else:
            await self.attack()

    async def _execute_containment_strategy(self, attack_force):
        """Execute pressure tactics."""

        enemy_base = self.enemy_start_locations[0]
        contain_pos = enemy_base.towards(self.start_location, 20)

        for unit in attack_force:
            unit.attack(contain_pos)

    async def handle_action_77(self):
        """Handle pressure tactics (action_id=77)."""

        attack_force = []

        immortals = self.units(UnitTypeId.IMMORTAL).ready
        stalkers = self.units(UnitTypeId.STALKER).ready
        observers = self.units(UnitTypeId.OBSERVER).ready

        if immortals.exists:
            attack_force.extend(immortals[:2])
        if stalkers.exists:
            attack_force.extend(stalkers[:4])
        if observers.exists:
            attack_force.append(observers.first)

        if attack_force:
            await self._execute_containment_strategy(attack_force)
        else:
            await self.attack()

    async def _execute_base_trade_strategy(self, attack_force):
        """Execute a base-trade strategy."""

        enemy_attacking = False
        for nexus in self.townhalls:
            if self.enemy_units.closer_than(30, nexus).exists:
                enemy_attacking = True
                break

        if enemy_attacking:

            for unit in attack_force:
                unit.attack(self.enemy_start_locations[0])
        else:

            await self._execute_default_attack_strategy(attack_force)

    async def _execute_default_attack_strategy(self, attack_force):
        """Execute the default attack strategy."""

        await self.attack()

    async def handle_action_78(self):
        """Handle base-trade tactics (action_id=78)."""

        attack_force = []

        zealots = self.units(UnitTypeId.ZEALOT).ready
        dts = self.units(UnitTypeId.DARKTEMPLAR).ready
        oracles = self.units(UnitTypeId.ORACLE).ready

        if zealots.exists:
            attack_force.extend(zealots[:4])
        if dts.exists:
            attack_force.extend(dts[:2])
        if oracles.exists:
            attack_force.extend(oracles[:1])

        if attack_force:
            await self._execute_base_trade_strategy(attack_force)
        else:
            await self.attack()

    async def handle_action_79(self):
        """Retreat to the base when disadvantaged (action_id=79)."""

        await self.handle_action_1()
        await self.handle_action_1()
        await self.handle_action_1()
        await self.handle_action_1()
        await self.handle_action_1()
        await self.attack_simple()

    async def _execute_retreat_strategy(self):
        """Execute the retreat strategy."""

        combat_units = self.units.of_type([
            UnitTypeId.ZEALOT, UnitTypeId.STALKER, UnitTypeId.ADEPT,
            UnitTypeId.IMMORTAL, UnitTypeId.COLOSSUS, UnitTypeId.ARCHON,
            UnitTypeId.VOIDRAY, UnitTypeId.PHOENIX
        ]).ready

        if not combat_units.exists:
            return

        for unit in combat_units:
            closest_nexus = self.townhalls.closest_to(unit)
            if closest_nexus:
                unit.move(closest_nexus.position)

        print(f"Ordered {combat_units.amount} units to return to base")
        self.defense_mode = True

    async def defend(self):
        """Execute a random defensive action from 73 through 79."""

        import random
        defense_action = random.randint(73, 78)

        method_name = f"handle_action_{defense_action}"
        handler = getattr(self, method_name, None)

        if handler and callable(handler):
            await handler()
            return True
        else:
            print(f"Warning: handler {method_name} was not found")
            return False

    async def assign_units_to_resource_clusters(self, unit_type):
        units = self.units(unit_type).idle
        resource_clusters = self.expansion_locations_list

        if units.exists and resource_clusters:

            for unit in units:
                target = random.choice(resource_clusters)
                unit.attack(target)

    async def attack(self):
        if self.army_supply >= 10:

            attack_units = [UnitTypeId.ZEALOT, UnitTypeId.ARCHON, UnitTypeId.STALKER,
                            UnitTypeId.ADEPT, UnitTypeId.IMMORTAL, UnitTypeId.COLOSSUS,
                            UnitTypeId.VOIDRAY, UnitTypeId.CARRIER, UnitTypeId.TEMPEST]

            all_attack_units = self.units.of_type(attack_units)

            if all_attack_units.amount > 0:

                attack_count = max(1, all_attack_units.amount // 2)
                attack_force = all_attack_units[:attack_count]

                enemy_start_location_cleared = not self.enemy_units.exists and self.is_visible(
                    self.enemy_start_locations[0])

                if enemy_start_location_cleared:

                    expansion_locations = self.expansion_locations_list
                    for i, unit in enumerate(attack_force):
                        if i < len(expansion_locations):
                            unit.attack(expansion_locations[i])
                        else:
                            unit.attack(self.enemy_start_locations[0])
                else:

                    for unit in attack_force:
                        unit.attack(self.enemy_start_locations[0])

    def system_prompt(self, summary):
        return (
            "You are a StarCraft II Protoss decision assistant. Select five valid actions "
            "from the observation, history summary, and action dictionary. Consider resources, "
            "supply, production, and threats. Return only five comma-separated action IDs."
            f"\nAction dictionary: {self.action_dict}\nHistory summary: {summary}"
        )

    def user_prompt(self, observation, history, actions):
        return (
            f"Current observation: {observation}\n"
            f"Recent history: {history}\n"
            f"Similar actions: {actions}\n"
            "Select five actions."
        )

    async def attack_simple(self, action_id=None):
        """
        Attack the nearest enemy and advance in unit groups.
        """
        attack_units_types = {
            UnitTypeId.ZEALOT, UnitTypeId.STALKER, UnitTypeId.ADEPT,
            UnitTypeId.IMMORTAL, UnitTypeId.COLOSSUS, UnitTypeId.ARCHON,
            UnitTypeId.VOIDRAY, UnitTypeId.CARRIER, UnitTypeId.TEMPEST
        }

        all_attack_units = self.units.of_type(attack_units_types).ready

        if all_attack_units.amount == 0:
            if action_id:
                self.record_failure(action_id, 'No attack units available')
            return

        if self.time < 300:
            attack_ratio = 0.3
        elif self.time < 600:
            attack_ratio = 0.5
        else:
            attack_ratio = 0.7

        attack_count = max(1, int(all_attack_units.amount * attack_ratio))
        attack_force = all_attack_units.take(attack_count)

        enemy_targets = self.enemy_units | self.enemy_structures

        if enemy_targets.exists:

            avg_position = Point2((0, 0))
            for unit in attack_force:
                nearest_enemy = enemy_targets.closest_to(unit)
                if nearest_enemy:
                    avg_position += nearest_enemy.position

            if len(attack_force) > 0:
                avg_position = avg_position / len(attack_force)

            for unit in attack_force:
                unit.attack(avg_position)
        else:

            if self.enemy_start_locations:
                for unit in attack_force:
                    unit.attack(self.enemy_start_locations[0])

    async def auto_defense_check(self):
        """Detect attacks and organize a counterattack."""

        enemy_near_base = False
        for nexus in self.townhalls:
            nearby_enemies = self.enemy_units.closer_than(25, nexus)
            if nearby_enemies.exists:
                enemy_near_base = True
                break

        workers_under_attack = False
        for worker in self.workers:
            if worker.is_attacking or worker.health_percentage < 0.8:
                workers_under_attack = True
                break

        if enemy_near_base or workers_under_attack:
            print(f"Attack detected: enemies near base={enemy_near_base}, workers under attack={workers_under_attack}")

            await self.defend()

            if self.supply_army >= 8:
                await self.attack()

    def get_game_phase(self, game_time_str):
        """
        Determine the game phase from an MM:SS timestamp.
        """
        if not game_time_str:
            return "early"

        try:

            minutes_str = game_time_str.split(':')[0]
            minutes = int(minutes_str)

            if minutes < 5:
                return "early"
            elif minutes < 10:
                return "mid"
            else:
                return "late"
        except:
            return "early"

    async def on_step(self, iteration):

        await self.auto_defense_check()
        await self.handle_action_1()
        observation = self.get_information()
        obs = observation.copy()

        phase = "early"
        if iteration % 1700 == 0:
            game_time = observation.get('game_time', '00:00')
            phase = self.get_game_phase(game_time)
            print(f"Switching phase: {phase}")
            self.gw._init_phases_graphs(phase)
            phase = "mid"

        t1 = time.time()
        current_obs_adj = self.gw.assign_weights(observation)
        combine_delta_H = combine_delta(0.2, 0.8, self.gw.G, self.gw.tree, current_obs_adj, self.gw.H_G, self.gw.H_T)
        observation = prune_obs(self.gw.G, 111, combine_delta_H, observation)
        t2 = time.time()
        self.prun_times.append(t2 - t1)

        if iteration % 10 == 0:

            sys_prompt = self.system_prompt(summary)
            print(f"summary: {summary}")

            similar_actions = []
            similar_history = eva.search_similar(phase, observation, threshold=0.9)
            if not len(similar_history) == 0:
                for i, res in enumerate(similar_history):
                    similar_actions.append(res['current_action'])

            print(f"Retrieved similar actions: {similar_actions}")
            usr_prompt = self.user_prompt(observation, self.history, similar_actions)

            encoding = tiktoken.get_encoding("cl100k_base")
            input_tokens = len(encoding.encode(sys_prompt)) + len(encoding.encode(usr_prompt))
            self.input_tokens.append(input_tokens)

            action, completion = await self.llm_decision._get_actions(sys_prompt, usr_prompt)
            self.inference_times.append(self.llm_decision.inference_time)
            print(f"LLM response time: {self.llm_decision.inference_time}")

            self.current_action_index = 0
            self.action_queue = action

            inference_result = {
                "observation": observation,
                "action": action
            }
            ep_history.append(inference_result)

        if self.current_action_index < len(self.action_queue):
            current_action = self.action_queue[self.current_action_index]

            method_name = f"handle_action_{current_action}"
            handler = getattr(self, method_name, None)
            if handler and callable(handler):
                await handler()
            else:
                print(f"Warning: No handler found for action ID {current_action}")

            self.current_action_index += 1

        self.history.append(observation)
        await self.auto_defense_check()

    async def on_end(self, game_result):
        """Handle the end of a game."""
        self.ep_time = self.time
        if game_result == Result.Defeat:
            self.ep_result = 0
        else:
            self.ep_result = 1

if __name__ == "__main__":

    summary_api_base = os.getenv("ESMA_SUMMARY_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    summary_api_key = os.environ["ESMA_SUMMARY_API_KEY"]

    summary_model = os.getenv("ESMA_SUMMARY_MODEL", "qwen3-30b-a3b-instruct-2507")

    api_base = os.getenv("ESMA_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    api_key = os.environ["ESMA_API_KEY"]

    model = os.getenv("ESMA_MODEL", "qwen3-next-80b-a3b-instruct")

    for episode in range(1):
        try:
            summary = ""

            llmdm = DecisionAgent(api_base, api_key, model)
            llmsm = AnalyzationExpert(summary_api_base, summary_api_key, summary_model)

            eva = EvaluationExpert(
                os.getenv("ESMA_MEMORY_EARLY"),
                os.getenv("ESMA_MEMORY_MID"),
                os.getenv("ESMA_MEMORY_LATE"),
            )

            gw = GraphBuilder()
            gw._init_phases_graphs("early")

            bot = ProtossBot(llmdm, llmsm, gw)
            bot.get_action_dict()

            his_obs = list(ep_history)
            summary = llmsm._get_summary(
                llmsm.system_prompt(), llmsm.user_prompt(his_obs)
            )

            run_game(
                maps.get("Simple64"),
                [
                    Bot(Race.Protoss, bot),
                    Computer(Race.Terran, Difficulty.VeryEasy)
                ],
                realtime=True,
            )
        except Exception as e:
            print(f"Game {episode} failed: {e}")

        ep_history = deque(maxlen=10)
