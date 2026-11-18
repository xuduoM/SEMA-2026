"""StarCraft II action definitions."""

from sc2.data import Race
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId

class ActionDictionary:
    def __init__(self, race):
        self.flattened_actions = {}
        self.race = race
        self.action_descriptions = {}
        self.action_dict_len = 0
        self.action_descriptions = self.init_actions_by_race(race)
        self.action_dict_len = len(self.action_descriptions)
        self._flatten_actions()
        self.empty_action_id = self.action_dict_len - 1
        self.abilities = {
            UnitTypeId.ZEALOT: AbilityId.WARPGATETRAIN_ZEALOT, UnitTypeId.ADEPT: AbilityId.TRAINWARP_ADEPT,
            UnitTypeId.STALKER: AbilityId.WARPGATETRAIN_STALKER,
            UnitTypeId.SENTRY: AbilityId.WARPGATETRAIN_SENTRY,
            UnitTypeId.HIGHTEMPLAR: AbilityId.WARPGATETRAIN_HIGHTEMPLAR,
            UnitTypeId.DARKTEMPLAR: AbilityId.WARPGATETRAIN_DARKTEMPLAR,
            UnitTypeId.MOTHERSHIP: AbilityId.NEXUSTRAINMOTHERSHIP_MOTHERSHIP
        }

        self.buildings = {
            Race.Protoss: {UnitTypeId.NEXUS: 71, UnitTypeId.PYLON: 18, UnitTypeId.ASSIMILATOR: 21, UnitTypeId.GATEWAY: 46, UnitTypeId.CYBERNETICSCORE: 36,
                           UnitTypeId.FORGE: 32, UnitTypeId.TWILIGHTCOUNCIL: 36, UnitTypeId.ROBOTICSFACILITY: 46, UnitTypeId.STARGATE: 43, UnitTypeId.TEMPLARARCHIVE: 36,
                           UnitTypeId.DARKSHRINE: 71, UnitTypeId.ROBOTICSBAY: 46, UnitTypeId.FLEETBEACON: 43, UnitTypeId.PHOTONCANNON: 29, UnitTypeId.SHIELDBATTERY: 30},
            Race.Zerg: {UnitTypeId.EXTRACTOR: 21, UnitTypeId.HATCHERY: 71, UnitTypeId.SPAWNINGPOOL: 46, UnitTypeId.BANELINGNEST: 43, UnitTypeId.ROACHWARREN: 39,
                        UnitTypeId.HYDRALISKDEN: 29, UnitTypeId.LAIR: 57, UnitTypeId.HIVE: 71, UnitTypeId.EVOLUTIONCHAMBER: 25, UnitTypeId.INFESTATIONPIT: 36,
                        UnitTypeId.SPIRE: 71, UnitTypeId.GREATERSPIRE: 71, UnitTypeId.ULTRALISKCAVERN: 46, UnitTypeId.LURKERDENMP: 57, UnitTypeId.SPORECRAWLER: 21, UnitTypeId.SPINECRAWLER: 36},
            Race.Terran: {UnitTypeId.COMMANDCENTER: 71, UnitTypeId.SUPPLYDEPOT: 21, UnitTypeId.REFINERY: 21, UnitTypeId.BARRACKS: 46, UnitTypeId.ENGINEERINGBAY: 25, UnitTypeId.BUNKER: 29,
                          UnitTypeId.FACTORY: 43, UnitTypeId.STARPORT: 36, UnitTypeId.ORBITALCOMMAND: 25, UnitTypeId.PLANETARYFORTRESS: 36, UnitTypeId.BARRACKSREACTOR: 36, UnitTypeId.BARRACKSTECHLAB: 18,
                          UnitTypeId.FACTORYREACTOR: 36, UnitTypeId.FACTORYTECHLAB: 18, UnitTypeId.STARPORTREACTOR: 36, UnitTypeId.STARPORTTECHLAB: 18,
                          UnitTypeId.GHOSTACADEMY: 29, UnitTypeId.ARMORY: 46, UnitTypeId.FUSIONCORE: 46, UnitTypeId.MISSILETURRET: 18, UnitTypeId.SENSORTOWER: 18}
        }

        self.tactic = {
            """
            1. Zealot & STALKER tactic:
                -Key buildings:
                    military: GATEWAY/WARPGATE (6+).
                    technolegy: CYBERNETICSCORE(1 and only 1), TWILIGHTCOUNCIL(1 and only 1, CYBERNETICSCORE needed), FORGE(1 and only 1).
                -Key technologies:
                    Necessary: WARPGATERESEARCH(CYBERNETICSCORE needed), CHARGE(TWILIGHTCOUNCIL needed).
                    Optimal: PROTOSSGROUNDWEAPONSLEEVEL(1,2, FORGE needed), PROTOSSGROUNDARMORSLEVEL(1,2, FORGE needed), PROTOSSSHIELDSLEEVEL(1,2, FORGE needed).
                -Key forces: ZEALOT: STALKER=1:1 or 2:1, as many as possible.
                -Key timing:
                    At 2 minutes: Complete the construction of the second NEXUS and 2 GATEWAY.
                    At 4 minutes: 3+ WARPGATE, 1 CYBERNETICSCORE built and WARPGATERESEARCH research started.
                    At 8 minutes: 6+ WARPGATE, 1 CYBERNETICSCORE, 1 TWILIGHTCOUNCIL(CYBERNETICSCORE needed) built and all necessary key technology research completed.
                -Applicable situation:
                    Only in the first 8 minutes of the game.
            2. Carrier tactic:
                -Key buildings:
                    military: STARGATE(6, CYBERNETICSCORE needed)
                    technolegy: CYBERNETICSCORE(1 and only 1), FLEETBEACON(1 and only 1, STARGATE and CYBERNETICSCORE needed), FORGE(1 and only 1).
                -Key technologies:
                    Necessary:  WARPGATERESEARCH(CYBERNETICSCORE needed), PROTOSSAIRWEAPONSLEEVEL(1,2, CYBERNETICSCORE needed).
                    Optimal: PROTOSSAIRMARMORSLEEVEL(1,2, CYBERNETICSCORE needed), PROTOSSSHIELDSLEEVEL(1,2, FORGE needed), PROTOSSGROUNDARMORSLEVEL(1,2, FORGE needed).
                -Key forces:
                    First 6 minutes, ZEALOT: STALKER=1:1, as many as possible, with little CARRIER.
                    After 6 minutes, Mainly CARRIER(STARGATE and FLEETBEACON need), with a small amount of STALKER or ZEALOT.
                -Key timing:
                    At 2 minutes: Complete the construction of the second NEXUS and 2 GATEWAY.
                    At 6 minutes: 2 STARGATE, 1 CYBERNETICSCORE, 1 FLEETBEACON, at least 1 CARRIER in trainning.
                    At 10 minutes: 6 STARGATE, 1 CYBERNETICSCORE, 1 FLEETBEACON, 6+ CARRIER and necessary technology research completed.
                -Applicable situation:
                    Usually after 6 minutes of the game, or enemies have air units / heavy units.
            """,
        }

        self.researches = {
            Race.Protoss: {
                'ProtossGroundWeaponsLevel1': [UpgradeId.PROTOSSGROUNDWEAPONSLEVEL1,
                                               AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL1],
                'ProtossGroundWeaponsLevel2': [UpgradeId.PROTOSSGROUNDWEAPONSLEVEL2,
                                               AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL2],
                'ProtossGroundWeaponsLevel3': [UpgradeId.PROTOSSGROUNDWEAPONSLEVEL3,
                                               AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL3],
                'ProtossGroundArmorsLevel1': [UpgradeId.PROTOSSGROUNDARMORSLEVEL1,
                                              AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRARMORLEVEL1],
                'ProtossGroundArmorsLevel2': [UpgradeId.PROTOSSGROUNDARMORSLEVEL2,
                                              AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRARMORLEVEL2],
                'ProtossGroundArmorsLevel3': [UpgradeId.PROTOSSGROUNDARMORSLEVEL3,
                                              AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRARMORLEVEL3],
                'ProtossShieldsLevel1': [UpgradeId.PROTOSSSHIELDSLEVEL1, AbilityId.FORGERESEARCH_PROTOSSSHIELDSLEVEL1],
                'ProtossShieldsLevel2': [UpgradeId.PROTOSSSHIELDSLEVEL2, AbilityId.FORGERESEARCH_PROTOSSSHIELDSLEVEL2],
                'ProtossShieldsLevel3': [UpgradeId.PROTOSSSHIELDSLEVEL3, AbilityId.FORGERESEARCH_PROTOSSSHIELDSLEVEL3],
                'ProtossAirWeaponsLevel1': [UpgradeId.PROTOSSAIRWEAPONSLEVEL1,
                                            AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL1],
                'ProtossAirWeaponsLevel2': [UpgradeId.PROTOSSAIRWEAPONSLEVEL2,
                                            AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL2],
                'ProtossAirWeaponsLevel3': [UpgradeId.PROTOSSAIRWEAPONSLEVEL3,
                                            AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRWEAPONSLEVEL3],
                'ProtossAirArmorsLevel1': [UpgradeId.PROTOSSAIRARMORSLEVEL1,
                                           AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRARMORLEVEL1],
                'ProtossAirArmorsLevel2': [UpgradeId.PROTOSSAIRARMORSLEVEL2,
                                           AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRARMORLEVEL2],
                'ProtossAirArmorsLevel3': [UpgradeId.PROTOSSAIRARMORSLEVEL3,
                                           AbilityId.CYBERNETICSCORERESEARCH_PROTOSSAIRARMORLEVEL3],
                'WarpGateResearch': [UpgradeId.WARPGATERESEARCH, AbilityId.RESEARCH_WARPGATE],
                'BlinkTech': [UpgradeId.BLINKTECH, AbilityId.RESEARCH_BLINK],
                'Charge': [UpgradeId.CHARGE, AbilityId.RESEARCH_CHARGE],
                'AdeptPiercingAttack': [UpgradeId.ADEPTPIERCINGATTACK, AbilityId.RESEARCH_ADEPTRESONATINGGLAIVES],
                'PsiStormTech': [UpgradeId.PSISTORMTECH, AbilityId.RESEARCH_PSISTORM],
                'DarkTemplarBlinkUpgrade': [UpgradeId.DARKTEMPLARBLINKUPGRADE, AbilityId.RESEARCH_SHADOWSTRIKE],
                'ObserverGraviticBooster': [UpgradeId.OBSERVERGRAVITICBOOSTER, AbilityId.RESEARCH_GRAVITICBOOSTER],
                'GraviticDrive': [UpgradeId.GRAVITICDRIVE, AbilityId.RESEARCH_GRAVITICDRIVE],
                'ExtendedThermalLance': [UpgradeId.EXTENDEDTHERMALLANCE, AbilityId.RESEARCH_EXTENDEDTHERMALLANCE],
                'PhoenixRangeUpgrade': [UpgradeId.PHOENIXRANGEUPGRADE, AbilityId.RESEARCH_PHOENIXANIONPULSECRYSTALS],
                'VoidRaySpeedUpgrade': [UpgradeId.VOIDRAYSPEEDUPGRADE,
                                        AbilityId.FLEETBEACONRESEARCH_RESEARCHVOIDRAYSPEEDUPGRADE],
            },
            Race.Zerg: {
                'ZergMeleeWeaponsLevel1': [UpgradeId.ZERGMELEEWEAPONSLEVEL1, AbilityId.RESEARCH_ZERGMELEEWEAPONSLEVEL1],
                'ZergMeleeWeaponsLevel2': [UpgradeId.ZERGMELEEWEAPONSLEVEL2, AbilityId.RESEARCH_ZERGMELEEWEAPONSLEVEL2],
                'ZergMeleeWeaponsLevel3': [UpgradeId.ZERGMELEEWEAPONSLEVEL3, AbilityId.RESEARCH_ZERGMELEEWEAPONSLEVEL3],
                'ZergMissileWeaponsLevel1': [UpgradeId.ZERGMISSILEWEAPONSLEVEL1,
                                             AbilityId.RESEARCH_ZERGMISSILEWEAPONSLEVEL1],
                'ZergMissileWeaponsLevel2': [UpgradeId.ZERGMISSILEWEAPONSLEVEL2,
                                             AbilityId.RESEARCH_ZERGMISSILEWEAPONSLEVEL2],
                'ZergMissileWeaponsLevel3': [UpgradeId.ZERGMISSILEWEAPONSLEVEL3,
                                             AbilityId.RESEARCH_ZERGMISSILEWEAPONSLEVEL3],
                'ZergGroundArmorsLevel1': [UpgradeId.ZERGGROUNDARMORSLEVEL1, AbilityId.RESEARCH_ZERGGROUNDARMORLEVEL1],
                'ZergGroundArmorsLevel2': [UpgradeId.ZERGGROUNDARMORSLEVEL2, AbilityId.RESEARCH_ZERGGROUNDARMORLEVEL2],
                'ZergGroundArmorsLevel3': [UpgradeId.ZERGGROUNDARMORSLEVEL3, AbilityId.RESEARCH_ZERGGROUNDARMORLEVEL3],
                'ZergFlyerWeaponsLevel1': [UpgradeId.ZERGFLYERWEAPONSLEVEL1, AbilityId.RESEARCH_ZERGFLYERATTACKLEVEL1],
                'ZergFlyerWeaponsLevel2': [UpgradeId.ZERGFLYERWEAPONSLEVEL2, AbilityId.RESEARCH_ZERGFLYERATTACKLEVEL2],
                'ZergFlyerWeaponsLevel3': [UpgradeId.ZERGFLYERWEAPONSLEVEL3, AbilityId.RESEARCH_ZERGFLYERATTACKLEVEL3],
                'ZergFlyerArmorsLevel1': [UpgradeId.ZERGFLYERARMORSLEVEL1, AbilityId.RESEARCH_ZERGFLYERARMORLEVEL1],
                'ZergFlyerArmorsLevel2': [UpgradeId.ZERGFLYERARMORSLEVEL2, AbilityId.RESEARCH_ZERGFLYERARMORLEVEL2],
                'ZergFlyerArmorsLevel3': [UpgradeId.ZERGFLYERARMORSLEVEL3, AbilityId.RESEARCH_ZERGFLYERARMORLEVEL3],
                'Burrow': [UpgradeId.BURROW, AbilityId.RESEARCH_BURROW],
                'overlordspeed': [UpgradeId.OVERLORDSPEED, AbilityId.RESEARCH_PNEUMATIZEDCARAPACE],
                'zerglingmovementspeed': [UpgradeId.ZERGLINGMOVEMENTSPEED, AbilityId.RESEARCH_ZERGLINGMETABOLICBOOST],
                'zerglingattackspeed': [UpgradeId.ZERGLINGATTACKSPEED, AbilityId.RESEARCH_ZERGLINGADRENALGLANDS],
                'GlialReconstitution': [UpgradeId.GLIALRECONSTITUTION, AbilityId.RESEARCH_GLIALREGENERATION],
                'TunnelingClaws': [UpgradeId.TUNNELINGCLAWS, AbilityId.RESEARCH_TUNNELINGCLAWS],
                'CentrificalHooks': [UpgradeId.CENTRIFICALHOOKS, AbilityId.RESEARCH_CENTRIFUGALHOOKS],
                'EvolveMuscularAugments': [UpgradeId.EVOLVEMUSCULARAUGMENTS, AbilityId.RESEARCH_MUSCULARAUGMENTS],
                'EvolveGroovedSpines': [UpgradeId.EVOLVEGROOVEDSPINES, AbilityId.RESEARCH_GROOVEDSPINES],
                'NeuralParasite': [UpgradeId.NEURALPARASITE, AbilityId.RESEARCH_NEURALPARASITE],
                'DiggingClaws': [UpgradeId.DIGGINGCLAWS, AbilityId.RESEARCH_ADAPTIVETALONS],
                'LurkerRange': [UpgradeId.LURKERRANGE, AbilityId.LURKERDENRESEARCH_RESEARCHLURKERRANGE],
                'ChitinousPlating': [UpgradeId.CHITINOUSPLATING, AbilityId.RESEARCH_CHITINOUSPLATING],
                'AnabolicSynthesis': [UpgradeId.ANABOLICSYNTHESIS, AbilityId.RESEARCH_ANABOLICSYNTHESIS],
            },
            Race.Terran: {
                'TerranInfantryWeaponsLevel1': [UpgradeId.TERRANINFANTRYWEAPONSLEVEL1,
                                                AbilityId.ENGINEERINGBAYRESEARCH_TERRANINFANTRYWEAPONSLEVEL1],
                'TerranInfantryWeaponsLevel2': [UpgradeId.TERRANINFANTRYWEAPONSLEVEL2,
                                                AbilityId.ENGINEERINGBAYRESEARCH_TERRANINFANTRYWEAPONSLEVEL2],
                'TerranInfantryWeaponsLevel3': [UpgradeId.TERRANINFANTRYWEAPONSLEVEL3,
                                                AbilityId.ENGINEERINGBAYRESEARCH_TERRANINFANTRYWEAPONSLEVEL3],
                'TerranInfantryArmorsLevel1': [UpgradeId.TERRANINFANTRYARMORSLEVEL1,
                                               AbilityId.ENGINEERINGBAYRESEARCH_TERRANINFANTRYARMORLEVEL1],
                'TerranInfantryArmorsLevel2': [UpgradeId.TERRANINFANTRYARMORSLEVEL2,
                                               AbilityId.ENGINEERINGBAYRESEARCH_TERRANINFANTRYARMORLEVEL2],
                'TerranInfantryArmorsLevel3': [UpgradeId.TERRANINFANTRYARMORSLEVEL3,
                                               AbilityId.ENGINEERINGBAYRESEARCH_TERRANINFANTRYARMORLEVEL3],
                'TerranVehicleWeaponsLevel1': [UpgradeId.TERRANVEHICLEWEAPONSLEVEL1,
                                               AbilityId.ARMORYRESEARCH_TERRANVEHICLEWEAPONSLEVEL1],
                'TerranVehicleWeaponsLevel2': [UpgradeId.TERRANVEHICLEWEAPONSLEVEL2,
                                               AbilityId.ARMORYRESEARCH_TERRANVEHICLEWEAPONSLEVEL2],
                'TerranVehicleWeaponsLevel3': [UpgradeId.TERRANVEHICLEWEAPONSLEVEL3,
                                               AbilityId.ARMORYRESEARCH_TERRANVEHICLEWEAPONSLEVEL3],
                'TerranShipWeaponsLevel1': [UpgradeId.TERRANSHIPWEAPONSLEVEL1,
                                            AbilityId.ARMORYRESEARCH_TERRANSHIPWEAPONSLEVEL1],
                'TerranShipWeaponsLevel2': [UpgradeId.TERRANSHIPWEAPONSLEVEL2,
                                            AbilityId.ARMORYRESEARCH_TERRANSHIPWEAPONSLEVEL2],
                'TerranShipWeaponsLevel3': [UpgradeId.TERRANSHIPWEAPONSLEVEL3,
                                            AbilityId.ARMORYRESEARCH_TERRANSHIPWEAPONSLEVEL3],
                'TerranVehicleAndShipArmorsLevel1': [UpgradeId.TERRANVEHICLEANDSHIPARMORSLEVEL1,
                                                     AbilityId.ARMORYRESEARCH_TERRANVEHICLEANDSHIPPLATINGLEVEL1],
                'TerranVehicleAndShipArmorsLevel2': [UpgradeId.TERRANVEHICLEANDSHIPARMORSLEVEL2,
                                                     AbilityId.ARMORYRESEARCH_TERRANVEHICLEANDSHIPPLATINGLEVEL2],
                'TerranVehicleAndShipArmorsLevel3': [UpgradeId.TERRANVEHICLEANDSHIPARMORSLEVEL3,
                                                     AbilityId.ARMORYRESEARCH_TERRANVEHICLEANDSHIPPLATINGLEVEL3],
                'TerranBuildingArmor': [UpgradeId.TERRANBUILDINGARMOR, AbilityId.RESEARCH_TERRANSTRUCTUREARMORUPGRADE],
                'HiSecAutoTracking': [UpgradeId.HISECAUTOTRACKING, AbilityId.RESEARCH_HISECAUTOTRACKING],
                'Stimpack': [UpgradeId.STIMPACK, AbilityId.BARRACKSTECHLABRESEARCH_STIMPACK],
                'ShieldWall': [UpgradeId.SHIELDWALL, AbilityId.RESEARCH_COMBATSHIELD],
                'PunisherGrenades': [UpgradeId.PUNISHERGRENADES, AbilityId.RESEARCH_CONCUSSIVESHELLS],
                'PersonalCloaking': [UpgradeId.PERSONALCLOAKING, AbilityId.RESEARCH_PERSONALCLOAKING],
                'SmartServos': [UpgradeId.SMARTSERVOS, AbilityId.RESEARCH_SMARTSERVOS],
                'HighCapacityBarrels': [UpgradeId.HIGHCAPACITYBARRELS, AbilityId.RESEARCH_INFERNALPREIGNITER],
                'DrillClaws': [UpgradeId.DRILLCLAWS, AbilityId.RESEARCH_DRILLINGCLAWS],
                'CycloneLockOnDamageUpgrade': [UpgradeId.CYCLONELOCKONDAMAGEUPGRADE,
                                               AbilityId.RESEARCH_CYCLONELOCKONDAMAGE],
                'MedivacIncreaseSpeedBoost': [UpgradeId.MEDIVACCADUCEUSREACTOR,
                                              AbilityId.FUSIONCORERESEARCH_RESEARCHMEDIVACENERGYUPGRADE],
                'LiberatorAGRangeUpgrade': [UpgradeId.LIBERATORAGRANGEUPGRADE,
                                            AbilityId.FUSIONCORERESEARCH_RESEARCHBALLISTICRANGE],
                'BansheeCloak': [UpgradeId.BANSHEECLOAK, AbilityId.RESEARCH_BANSHEECLOAKINGFIELD],
                'BansheeSpeed': [UpgradeId.BANSHEESPEED, AbilityId.RESEARCH_BANSHEEHYPERFLIGHTROTORS],
                'InterferenceMatrix': [UpgradeId.INTERFERENCEMATRIX,
                                       AbilityId.STARPORTTECHLABRESEARCH_RESEARCHRAVENINTERFERENCEMATRIX],
                'BattlecruiserEnableSpecializations': [UpgradeId.BATTLECRUISERENABLESPECIALIZATIONS,
                                                       AbilityId.RESEARCH_BATTLECRUISERWEAPONREFIT]
            }
        }

    def init_actions_by_race(self, race):
        if race == "Protoss":
            return self.init_protoss_actions()
        elif race == "Zerg":
            return self.init_zerg_actions()
        elif race == "Terran":
            return self.init_terran_actions()
        else:
            raise ValueError("Invalid race")

    def _flatten_actions(self):
        for category, actions in self.action_descriptions.items():
            self.flattened_actions.update(actions)

    def init_protoss_actions(self):
        return {
            'TRAIN UNIT': {
                0: 'TRAIN PROBE', 1: 'TRAIN ZEALOT', 2: 'TRAIN ADEPT', 3: 'TRAIN STALKER',
                4: 'TRAIN SENTRY', 5: 'TRAIN HIGHTEMPLAR', 6: 'TRAIN DARKTEMPLAR', 7: 'TRAIN VOIDRAY',
                8: 'TRAIN CARRIER', 9: 'TRAIN TEMPEST', 10: 'TRAIN ORACLE', 11: 'TRAIN PHOENIX',
                12: 'TRAIN MOTHERSHIP', 13: 'TRAIN OBSERVER', 14: 'TRAIN IMMORTAL',
                15: 'TRAIN WARPPRISM', 16: 'TRAIN COLOSSUS', 17: 'TRAIN DISRUPTOR', 18: 'MORPH ARCHON'},
            'BUILD STRUCTURE': {
                19: 'BUILD PYLON', 20: 'BUILD ASSIMILATOR', 21: 'BUILD NEXUS',
                22: 'BUILD GATEWAY', 23: 'BUILD CYBERNETICSCORE',
                24: 'BUILD FORGE', 25: 'BUILD TWILIGHTCOUNCIL',
                26: 'BUILD ROBOTICSFACILITY', 27: 'BUILD STARGATE', 28: 'BUILD TEMPLARARCHIVE',
                29: 'BUILD DARKSHRINE', 30: 'BUILD ROBOTICSBAY',
                31: 'BUILD FLEETBEACON', 32: 'BUILD PHOTONCANNON', 33: 'BUILD SHIELDBATTERY'},
            'RESEARCH TECHNIQUE': {
                34: 'RESEARCH WARPGATERESEARCH', 35: 'RESEARCH PROTOSSAIRWEAPONSLEVEL1',
                36: 'RESEARCH PROTOSSAIRWEAPONSLEVEL2',
                37: 'RESEARCH PROTOSSAIRWEAPONSLEVEL3',
                38: 'RESEARCH PROTOSSAIRARMORSLEVEL1', 39: 'RESEARCH PROTOSSAIRARMORSLEVEL2',
                40: 'RESEARCH PROTOSSAIRARMORSLEVEL3', 41: 'RESEARCH ADEPTPIERCINGATTACK',
                42: 'RESEARCH BLINKTECH', 43: 'RESEARCH CHARGE', 44: 'RESEARCH PROTOSSGROUNDWEAPONSLEVEL1',
                45: 'RESEARCH PROTOSSGROUNDWEAPONSLEVEL2', 46: 'RESEARCH PROTOSSGROUNDWEAPONSLEVEL3',
                47: 'RESEARCH PROTOSSGROUNDARMORSLEVEL1',
                48: 'RESEARCH PROTOSSGROUNDARMORSLEVEL2',
                49: 'RESEARCH PROTOSSGROUNDARMORSLEVEL3', 50: 'RESEARCH PROTOSSSHIELDSLEVEL1',
                51: 'RESEARCH PROTOSSSHIELDSLEVEL2', 52: 'RESEARCH PROTOSSSHIELDSLEVEL3',
                53: 'RESEARCH EXTENDEDTHERMALLANCE', 54: 'RESEARCH GRAVITICDRIVE',
                55: 'RESEARCH OBSERVERGRAVITICBOOSTER', 56: 'RESEARCH PSISTORMTECH',
                57: 'RESEARCH VOIDRAYSPEEDUPGRADE', 58: 'RESEARCH PHOENIXRANGEUPGRADE',
                59: 'RESEARCH TEMPESTGROUNDATTACKUPGRADE'},
            'OTHER ACTION': {
                60: 'SCOUTING PROBE',
                61: 'SCOUTING OBSERVER',
                62: 'SCOUTING ZEALOT',
                63: 'SCOUTING PHOENIX',
                64: 'MULTI-ATTACK',
                65: 'MULTI-RETREAT',
                66: 'CHRONOBOOST NEXUS',
                67: 'CHRONOBOOST CYBERNETICSCORE',
                68: 'CHRONOBOOST TWILIGHTCOUNCIL',
                69: 'CHRONOBOOST STARGATE',
                70: 'CHRONOBOOST FORGE',
                71: 'EMPTY ACTION',
                72: 'MINE RESOURCES'},
            'TACTICAL ACTION': {
                73: 'KITING STRATEGY',
                74: 'FLANKING STRATEGY',
                75: 'HIT AND RUN STRATEGY',
                76: 'DEATHBALL STRATEGY',
                77: 'CONTAINMENT STRATEGY',
                78: 'BASE TRADE STRATEGY',
                79: 'DEFAULT ATTACK STRATEGY'
            }
        }

    def init_zerg_actions(self):

        return {
            'TRAIN UNIT': {
                0: 'TRAIN OVERLORD', 1: 'TRAIN DRONE', 2: 'TRAIN QUEEN', 3: 'TRAIN ZERGLING',
                4: 'TRAIN ROACH', 5: 'TRAIN HYDRALISK', 6: 'TRAIN MUTALISK',
                7: 'TRAIN CORRUPTOR', 8: 'TRAIN INFESTOR ', 9: 'TRAIN SWARMHOST',
                10: 'TRAIN ULTRALISK', 11: 'TRAIN VIPER', 12: 'MORPH BANELING',
                13: 'MORPH RAVAGER', 14: 'MORPH LURKER', 15: 'MORPH BROODLORD',
                16: 'MORPH OVERSEER'},
            'BUILD STRUCTURE': {
                17: 'BUILD EXTRACTOR', 18: 'BUILD HATCHERY', 19: 'BUILD SPAWNINGPOOL',
                20: 'BUILD BANELINGNEST', 21: 'BUILD ROACHWARREN',
                22: 'BUILD HYDRALISKDEN', 23: 'BUILD LAIR',
                24: 'BUILD HIVE', 25: 'BUILD EVOLUTIONCHAMBER',
                26: 'BUILD INFESTATIONPIT', 27: 'BUILD SPIRE',
                28: 'BUILD GREATERSPIRE', 29: 'BUILD ULTRALISKCAVERN',
                30: 'BUILD LURKERDEN', 31: 'BUILD SPORECRAWLER', 32: 'BUILD SPINECRAWLER'},
            'RESEARCH TECHNIQUE': {
                33: 'RESEARCH ZERGLINGMOVEMENTSPEED', 34: 'RESEARCH ZERGLINGATTACKSPEED',
                35: 'RESEARCH BANELINGMOVEMENTSPEED', 36: 'RESEARCH ROACHMOVEMENTSPEED',
                37: 'RESEARCH TUNNELINGCLAWS', 38: 'RESEARCH OVERLORDSPEED', 39: 'RESEARCH BURROW',
                40: 'RESEARCH HYDRAMOVEMENTSPEED', 41: 'RESEARCH EVOLVEGROOVEDSPINES',
                42: 'RESEARCH ZERGMELEEWEAPONSLEVEL1',
                43: 'RESEARCH ZERGMELEEWEAPONSLEVEL2', 44: 'RESEARCH ZERGMELEEWEAPONSLEVEL3',
                45: 'RESEARCH ZERGMISSILEWEAPONSLEVEL1',
                46: 'RESEARCH ZERGMISSILEWEAPONSLEVEL2', 47: 'RESEARCH ZERGMISSILEWEAPONSLEVEL3',
                48: 'RESEARCH ZERGGROUNDARMORSLEVEL1',
                49: 'RESEARCH ZERGGROUNDARMORSLEVEL2', 50: 'RESEARCH ZERGGROUNDARMORSLEVEL3',
                51: 'RESEARCH INFESTORENERGYUPGRADE', 52: 'RESEARCH NEURALPARASITE',
                53: 'RESEARCH ZERGFLYERWEAPONSLEVEL1', 54: 'RESEARCH ZERGFLYERWEAPONSLEVEL2',
                55: 'RESEARCH ZERGFLYERWEAPONSLEVEL3', 56: 'RESEARCH ZERGFLYERARMORSLEVEL1',
                57: 'RESEARCH ZERGFLYERARMORSLEVEL2',
                58: 'RESEARCH ZERGFLYERARMORSLEVEL3',
                59: 'RESEARCH CHITINOUSPLATING', 60: 'RESEARCH ANABOLICSYNTHESIS', 61: 'RESEARCH LURKERRANGE',
                62: "RESEARCH DIGGINGCLAWS"},
            'OTHER ACTION': {
                63: 'SCOUTING ZERGLING', 64: 'SCOUTING OVERLORD', 65: 'SCOUTING OVERSEER', 66: 'MULTI-ATTACK',
                67: 'MULTI-RETREAT',
                68: 'QUEEN INJECTLARVA', 69: 'EMPTY ACTION',
                70: 'MINE RESOURCES'},
            'TACTICAL ACTION': {
                71: 'KITING STRATEGY',
                72: 'FLANKING STRATEGY',
                73: 'HIT AND RUN STRATEGY',
                74: 'DEATHBALL STRATEGY',
                75: 'CONTAINMENT STRATEGY',
                76: 'BASE TRADE STRATEGY',
                77: 'DEFAULT ATTACK STRATEGY'
            }
        }

    def init_terran_actions(self):
        return {
            'TRAIN UNIT': {
                0: 'TRAIN SCV', 1: 'TRAIN MARINE', 2: 'TRAIN MARAUDER', 3: 'TRAIN REAPER',
                4: 'TRAIN GHOST', 5: 'TRAIN HELLION', 6: 'TRAIN HELLBAT', 7: 'TRAIN WIDOWMINE',
                8: 'TRAIN CYCLONE', 9: 'TRAIN SIEGETANK', 10: 'TRAIN THOR', 11: 'TRAIN VIKING',
                12: 'TRAIN MEDIVAC', 13: 'TRAIN LIBERATOR', 14: 'TRAIN RAVEN', 15: 'TRAIN BANSHEE',
                16: 'TRAIN BATTLECRUISER'
            },
            'BUILD STRUCTURE': {
                17: 'BUILD SUPPLYDEPOT', 18: 'BUILD REFINERY', 19: 'BUILD COMMANDCENTER',
                20: 'BUILD BARRACKS', 21: 'BUILD FACTORY', 22: 'BUILD STARPORT',
                23: 'BUILD ENGINEERINGBAY', 24: 'BUILD ARMORY', 25: 'BUILD FUSIONCORE',
                26: 'BUILD GHOSTACADEMY', 27: 'BUILD BUNKER', 28: 'BUILD MISSILETURRET',
                29: 'BUILD SENSORTOWER', 30: 'BUILD TECHLAB', 31: 'BUILD REACTOR'
            },
            'RESEARCH TECHNIQUE': {
                32: 'RESEARCH TERRANINFANTRYWEAPONSLEVEL1', 33: 'RESEARCH TERRANINFANTRYWEAPONSLEVEL2',
                34: 'RESEARCH TERRANINFANTRYWEAPONSLEVEL3', 35: 'RESEARCH TERRANINFANTRYARMORSLEVEL1',
                36: 'RESEARCH TERRANINFANTRYARMORSLEVEL2', 37: 'RESEARCH TERRANINFANTRYARMORSLEVEL3',
                38: 'RESEARCH TERRANVEHICLEWEAPONSLEVEL1', 39: 'RESEARCH TERRANVEHICLEWEAPONSLEVEL2',
                40: 'RESEARCH TERRANVEHICLEWEAPONSLEVEL3', 41: 'RESEARCH TERRANVEHICLEARMORSLEVEL1',
                42: 'RESEARCH TERRANVEHICLEARMORSLEVEL2', 43: 'RESEARCH TERRANVEHICLEARMORSLEVEL3',
                44: 'RESEARCH TERRANSHIPWEAPONSLEVEL1', 45: 'RESEARCH TERRANSHIPWEAPONSLEVEL2',
                46: 'RESEARCH TERRANSHIPWEAPONSLEVEL3', 47: 'RESEARCH STIMPACK',
                48: 'RESEARCH COMBATSHIELD', 49: 'RESEARCH CONCUSSIVESHELLS',
                50: 'RESEARCH PERSONALCLOAKING', 51: 'RESEARCH INFERNALPREIGNITER',
                52: 'RESEARCH DRILLINGCLAWS', 53: 'RESEARCH TRANSFORMATIONSERVOS',
                54: 'RESEARCH SMARTSERVOS', 55: 'RESEARCH CADUCEUSREACTOR',
                56: 'RESEARCH CORVIDREACTOR', 57: 'RESEARCH BANSHEECLOAK',
                58: 'RESEARCH LIBERATORAGRANGEUPGRADE', 59: 'RESEARCH BATTLECRUISERSPECIALIZATIONS'
            },
            'OTHER ACTION': {
                60: 'SCOUTING SCV', 61: 'SCOUTING MARINE', 62: 'SCOUTING REAPER', 63: 'SCOUTING HELLION',
                64: 'SCOUTING VIKING', 65: 'SCOUTING BANSHEE', 66: 'MULTI-ATTACK', 67: 'MULTI-RETREAT',
                68: 'MULE DROP', 69: 'SCAN', 70: 'SIEGE MODE', 71: 'UNSIEGE MODE', 72: 'STIM PACK',
                73: 'EMPTY ACTION',
                74: 'MINE RESOURCES'
            },
            'TACTICAL ACTION': {
                75: 'KITING STRATEGY',
                76: 'FLANKING STRATEGY',
                77: 'HIT AND RUN STRATEGY',
                78: 'DEATHBALL STRATEGY',
                79: 'CONTAINMENT STRATEGY'
            }
        }

    def system_prompt(self, action_dict):
        return (
            "You control a Protoss agent in StarCraft II. "
            "Choose five valid actions that balance economy, supply, production, technology, and defense. "
            "Output only five comma-separated action IDs. "
            f"Action dictionary: {action_dict}"
        )
