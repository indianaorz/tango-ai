chip_dictionary = {
    1: {
        "name": "Cannon",
        "range": ["MatchY"],
        "damage": 40,
        "comments": "Basic projectile. Single hit."
    },
    2: {
        "name": "HiCannon",
        "range": ["MatchY"],
        "damage": 100,
        "comments": "Basic projectile. Single hit."
    },
    3: {
        "name": "M-Cannon",
        "range": ["MatchY"],
        "damage": 180,
        "comments": "Basic projectile. Single hit."
    },
    4: {
        "name": "AirShot1",
        "range": ["MatchY"],
        "damage": 20,
        "comments": "Knocks enemy back 1 square. Moves objects."
    },
    5: {
        "name": "Vulcan1",
        "range": ["MatchY"],
        "damage": 10,
        "comments": "Hits 3 times. Pierces 1 panel behind the target."
    },
    6: {
        "name": "Vulcan2",
        "range": ["MatchY"],
        "damage": 15,
        "comments": "Hits 4 times. Pierces 1 panel behind the target."
    },
    7: {
        "name": "Vulcan3",
        "range": ["MatchY"],
        "damage": 20,
        "comments": "Hits 5 times. Pierces 1 panel behind the target."
    },
    8: {
        "name": "SuprVulc",
        "range": ["MatchY"],
        "damage": 20,
        "comments": "Hits 10 times. Pierces 1 panel behind the target."
    },
    9: {
        "name": "Spreader1",
        "range": ["MatchY"],
        "damage": 30,
        "comments": "On hit, creates an explosion that damages all adjacent panels (3x3 area)."
    },
    10: {
        "name": "Spreader2",
        "range": ["MatchY"],
        "damage": 60,
        "comments": "On hit, creates an explosion that damages all adjacent panels (3x3 area)."
    },
    11: {
        "name": "Spreader3",
        "range": ["MatchY"],
        "damage": 90,
        "comments": "On hit, creates an explosion that damages all adjacent panels (3x3 area)."
    },
    12: {
        "name": "TankCan1",
        "range": ["MatchY", "BackColumn"],
        "damage": 120,
        "comments": "Knocks back. If it hits an enemy in the back column (or a wall), it explodes in a 3x3 blast."
    },
    13: {
        "name": "TankCan2",
        "range": ["MatchY", "BackColumn"],
        "damage": 160,
        "comments": "Knocks back. If it hits an enemy in the back column (or a wall), it explodes in a 3x3 blast."
    },
    14: {
        "name": "TankCan3",
        "range": ["MatchY", "BackColumn"],
        "damage": 200,
        "comments": "Knocks back. If it hits an enemy in the back column (or a wall), it explodes in a 3x3 blast."
    },
    15: {
        "name": "GunDelS1",
        "range": ["Pattern"],
        "pattern": [["0", "0", "X"], ["P", "0", "X"], ["0", "0", "X"]],
        "damage": 120,
        "comments": "Hits 2 panels ahead. Deals damage over time (60 frames). Ignores Holy Panels. Removes Barrier/Invis."
    },
    16: {
        "name": "GunDelS2",
        "range": ["Pattern"],
        "pattern": [["0", "0", "X"], ["P", "0", "X"], ["0", "0", "X"]],
        "damage": 180,
        "comments": "Hits 2 panels ahead. Deals damage over time (90 frames). Ignores Holy Panels. Removes Barrier/Invis."
    },
    17: {
        "name": "GunDelS3",
        "range": ["Pattern"],
        "pattern": [["0", "0", "X"], ["P", "0", "X"], ["0", "0", "X"]],
        "damage": 240,
        "comments": "Hits 2 panels ahead. Deals damage over time (120 frames). Ignores Holy Panels. Removes Barrier/Invis."
    },
    18: {
        "name": "GunDelEX*",
        "range": ["Pattern"],
        "damage": 80,
        "comments": "Spread sunbeam. Boosted damage if used in an outdoor area."
    },
    19: {
        "name": "YoYo",
        "range": ["Pattern"],
        "pattern": [["P", "X", "X", "X"]],
        "damage": 50,
        "comments": "Hits 3 panels ahead. Hits 1 time going out, 3 times at the apex, 1 time returning. Acts as shield."
    },
    20: {
        "name": "FireBrn1",
        "range": ["Pattern"],
        "pattern": [["P", "X", "X", "X"]],
        "damage": 70,
        "comments": "Fires flame 3 squares ahead. Cracks panels."
    },
    21: {
        "name": "FireBrn2",
        "range": ["Pattern"],
        "pattern": [["P", "X", "X", "X"]],
        "damage": 110,
        "comments": "Fires flame 3 squares ahead. Cracks panels."
    },
    22: {
        "name": "FireBrn3",
        "range": ["Pattern"],
        "pattern": [["P", "X", "X", "X"]],
        "damage": 150,
        "comments": "Fires flame 3 squares ahead. Cracks panels."
    },
    23: {
        "name": "WideSht",
        "range": ["MatchY"],
        "damage": 100,
        "comments": "Fires a 3-panel wide wave."
    },
    24: {
        "name": "TrnArrw1",
        "range": ["MatchY"],
        "damage": 30,
        "comments": "Number of arrows = distance to target + 1."
    },
    25: {
        "name": "TrnArrw2",
        "range": ["MatchY"],
        "damage": 40,
        "comments": "Number of arrows = distance to target + 1."
    },
    26: {
        "name": "TrnArrw3",
        "range": ["MatchY"],
        "damage": 50,
        "comments": "Number of arrows = distance to target + 1."
    },
    27: {
        "name": "BblStar1",
        "range": ["MatchY"],
        "damage": 60,
        "comments": "Traps enemy in bubble (Electric hits x2). Lasts 150F. Freezes on Ice panels."
    },
    28: {
        "name": "BblStar2",
        "range": ["MatchY"],
        "damage": 80,
        "comments": "Traps enemy in bubble (Electric hits x2). Lasts 150F. Freezes on Ice panels."
    },
    29: {
        "name": "BblStar3",
        "range": ["MatchY"],
        "damage": 100,
        "comments": "Traps enemy in bubble (Electric hits x2). Lasts 150F. Freezes on Ice panels."
    },
    30: {
        "name": "Thunder",
        "range": ["MatchY"],
        "damage": 40,
        "comments": "Slow moving ball. Paralyzes for 90F."
    },
    31: {
        "name": "DolThdr1",
        "range": ["MatchY"],
        "damage": 120,
        "comments": "Piercing thunder attack."
    },
    32: {
        "name": "DolThdr2",
        "range": ["MatchY"],
        "damage": 150,
        "comments": "Piercing thunder attack."
    },
    33: {
        "name": "DolThdr3",
        "range": ["MatchY"],
        "damage": 180,
        "comments": "Piercing thunder attack."
    },
    34: {
        "name": "ElcPuls1",
        "range": ["Pattern"],
        "pattern": [["0", "0", "X"], ["P", "X", "X"], ["0", "0", "X"]],
        "damage": 100,
        "comments": "Paralyzes for 90F. Pierces Invisible."
    },
    35: {
        "name": "ElcPuls2",
        "range": ["Pattern"],
        "pattern": [["0", "0", "X"], ["P", "X", "X"], ["0", "0", "X"]],
        "damage": 120,
        "comments": "Pulls enemy forward 1 square. Pierces Invisible."
    },
    36: {
        "name": "ElcPuls3",
        "range": ["Pattern"],
        "pattern": [["0", "0", "X"], ["P", "X", "X"], ["0", "0", "X"]],
        "damage": 140,
        "comments": "Inflicts HP Bug. Pierces Invisible."
    },
    37: {
        "name": "RskyHny1",
        "range": ["MatchY"],
        "damage": 10,
        "comments": "Block attack to counter. Counters with 5 hits (swarms)."
    },
    38: {
        "name": "RskyHny2",
        "range": ["MatchY"],
        "damage": 15,
        "comments": "Block attack to counter. Counters with 5 hits (swarms)."
    },
    39: {
        "name": "RskyHny3",
        "range": ["MatchY"],
        "damage": 20,
        "comments": "Block attack to counter. Counters with 5 hits (swarms)."
    },
    40: {
        "name": "RlngLog1",
        "range": ["MatchY"],
        "damage": 50,
        "comments": "Hits 2-wide row (Double width). Stops on holes."
    },
    41: {
        "name": "RlngLog2",
        "range": ["MatchY"],
        "damage": 70,
        "comments": "Hits 2-wide row (Double width). Stops on holes."
    },
    42: {
        "name": "RlngLog3",
        "range": ["MatchY"],
        "damage": 90,
        "comments": "Hits 2-wide row (Double width). Stops on holes."
    },
    43: {
        "name": "MachGun1",
        "range": ["MatchY"],
        "damage": 30,
        "comments": "Fires 9 shots at the row with the closest enemy."
    },
    44: {
        "name": "MachGun2",
        "range": ["MatchY"],
        "damage": 50,
        "comments": "Fires 9 shots at the row with the closest enemy."
    },
    45: {
        "name": "MachGun3",
        "range": ["MatchY"],
        "damage": 70,
        "comments": "Fires 9 shots at the row with the closest enemy."
    },
    46: {
        "name": "HeatDrgn",
        "range": ["MatchY"],
        "damage": 140,
        "comments": "Summons Dragon. Hits 2x3 area in U-shape."
    },
    47: {
        "name": "ElecDrgn",
        "range": ["MatchY"],
        "damage": 150,
        "comments": "Summons Dragon. Hits 2x3 area in U-shape. Cracks panels."
    },
    48: {
        "name": "AquaDrgn",
        "range": ["MatchY"],
        "damage": 120,
        "comments": "Summons Dragon. Hits 2x3 area in U-shape. Creates Ice panels."
    },
    49: {
        "name": "WoodDrgn",
        "range": ["MatchY"],
        "damage": 130,
        "comments": "Summons Dragon. Hits 2x3 area in U-shape. Creates Grass panels."
    },
    50: {
        "name": "AirHocky",
        "range": ["MatchY"],
        "damage": 60,
        "comments": "Bounces off walls. Hits up to 11 times. Break attribute."
    },
    51: {
        "name": "DrilArm",
        "range": ["MatchY"],
        "damage": 70,
        "comments": "Hits 3 times. Knocks enemy back 2 squares. Break attribute."
    },
    52: {
        "name": "Tornado",
        "range": ["Pattern"],
        "pattern": [["P", "0", "X"]],
        "damage": 20,
        "comments": "Hits 8 times. 2 squares ahead. Double damage and removes panel type on Ice/Grass/Magma."
    },
    53: {
        "name": "Static",
        "range": ["Pattern"],
        "pattern": [["P", "0", "X"]],
        "damage": 20,
        "comments": "Hits 8 times. Range increases based on number of bugs in NavCust."
    },
    54: {
        "name": "MiniBomb",
        "range": ["Pattern"],
        "pattern": [["P", "0", "0", "X"]],
        "damage": 50,
        "comments": "Throws bomb 3 squares ahead. Single hit."
    },
    55: {
        "name": "EnergBom",
        "range": ["Pattern"],
        "pattern": [["P", "0", "0", "X"]],
        "damage": 40,
        "comments": "Throws bomb 3 squares ahead. Hits 3 times."
    },
    56: {
        "name": "MegEnBom",
        "range": ["Pattern"],
        "pattern": [["P", "0", "0", "X"]],
        "damage": 60,
        "comments": "Throws bomb 3 squares ahead. Hits 3 times."
    },
    57: {
        "name": "FlshBom1",
        "range": ["Pattern"],
        "pattern": [["P", "0", "0", "X"]],
        "damage": 40,
        "comments": "Throws bomb 3 squares ahead. Stuns for 90F. Pierces Invisible."
    },
    58: {
        "name": "FlshBom2",
        "range": ["Pattern"],
        "pattern": [["P", "0", "0", "X"]],
        "damage": 70,
        "comments": "Throws bomb 3 squares ahead. Stuns for 150F. Pierces Invisible."
    },
    59: {
        "name": "FlshBom3",
        "range": ["Pattern"],
        "pattern": [["P", "0", "0", "X"]],
        "damage": 100,
        "comments": "Throws bomb 3 squares ahead. Stuns for 150F, Blinds for 480F. Pierces Invisible."
    },
    60: {
        "name": "BlkBomb",
        "range": ["Pattern"],
        "pattern": [["P", "0", "0", "X"]],
        "damage": 250,
        "comments": "Throws bomb 3 squares ahead. Fire element. Hitting with Fire activates it instantly."
    },
    61: {
        "name": "AquaNdl1",
        "range": ["Any"],
        "damage": 40,
        "comments": "Fires 3 needles. First hit causes flashing."
    },
    62: {
        "name": "AquaNdl2",
        "range": ["Any"],
        "damage": 60,
        "comments": "Fires 3 needles. First hit causes flashing."
    },
    63: {
        "name": "AquaNdl3",
        "range": ["Any"],
        "damage": 80,
        "comments": "Fires 3 needles. First hit causes flashing."
    },
    64: {
        "name": "CornSht1",
        "range": ["MatchY"],
        "damage": 50,
        "comments": "Hits 2 times. Creates Grass panel."
    },
    65: {
        "name": "CornSht2",
        "range": ["MatchY"],
        "damage": 60,
        "comments": "Hits 2 times. Creates Grass panel."
    },
    66: {
        "name": "CornSht3",
        "range": ["MatchY"],
        "damage": 70,
        "comments": "Hits 2 times. Creates Grass panel."
    },
    67: {
        "name": "BugBomb",
        "range": ["Pattern"],
        "pattern": [["P", "0", "0", "X"]],
        "damage": 0,
        "comments": "Throws bomb 3 squares ahead. Inflicts random bugs (Confusion, HP Drain, etc)."
    },
    68: {
        "name": "GrasSeed",
        "range": ["Pattern"],
        "pattern": [["P", "0", "0", "X"]],
        "damage": 0,
        "comments": "Turns 3x3 area into Grass panels (Wood)."
    },
    69: {
        "name": "IceSeed",
        "range": ["Pattern"],
        "pattern": [["P", "0", "0", "X"]],
        "damage": 0,
        "comments": "Turns 3x3 area into Ice panels (Aqua)."
    },
    70: {
        "name": "PoisSeed",
        "range": ["Pattern"],
        "pattern": [["P", "0", "0", "X"]],
        "damage": 0,
        "comments": "Turns 3x3 area into Poison panels (Drain HP)."
    },
    71: {
        "name": "Sword",
        "range": ["Pattern"],
        "pattern": [["P", "X"]],
        "damage": 80,
        "comments": "Basic sword slash 1 panel ahead."
    },
    72: {
        "name": "WideSwrd",
        "range": ["Pattern"],
        "pattern": [["0", "X"], ["P", "X"], ["0", "X"]],
        "damage": 80,
        "comments": "Vertical slash. Hits 1x3 column ahead."
    },
    73: {
        "name": "LongSwrd",
        "range": ["Pattern"],
        "pattern": [["P", "X", "X"]],
        "damage": 100,
        "comments": "Horizontal slash. Hits 2 panels ahead."
    },
    74: {
        "name": "WideBlde",
        "range": ["Pattern"],
        "pattern": [["0", "X"], ["P", "X"], ["0", "X"]],
        "damage": 150,
        "comments": "Vertical slash. Hits 1x3 column ahead."
    },
    75: {
        "name": "LongBlde",
        "range": ["Pattern"],
        "pattern": [["P", "X", "X"]],
        "damage": 150,
        "comments": "Horizontal slash. Hits 2 panels ahead."
    },
    76: {
        "name": "FireSwrd",
        "range": ["Pattern"],
        "pattern": [["0", "X"], ["P", "X"], ["0", "X"]],
        "damage": 140,
        "comments": "Vertical slash. Hits 1x3 column ahead. Fire element."
    },
    77: {
        "name": "AquaSwrd",
        "range": ["Pattern"],
        "pattern": [["0", "X"], ["P", "X"], ["0", "X"]],
        "damage": 160,
        "comments": "Vertical slash. Hits 1x3 column ahead. Aqua element."
    },
    78: {
        "name": "ElecSwrd",
        "range": ["Pattern"],
        "pattern": [["0", "X"], ["P", "X"], ["0", "X"]],
        "damage": 120,
        "comments": "Vertical slash. Hits 1x3 column ahead. Elec element. Paralyzes."
    },
    79: {
        "name": "BambSwrd",
        "range": ["Pattern"],
        "pattern": [["0", "X"], ["P", "X"], ["0", "X"]],
        "damage": 150,
        "comments": "Vertical slash. Hits 1x3 column ahead. Wood element."
    },
    80: {
        "name": "WindRack",
        "range": ["Pattern"],
        "pattern": [["0", "X"], ["P", "X"], ["0", "X"]],
        "damage": 140,
        "comments": "Vertical slash. Hits 1x3 column ahead. Wind element. Blows enemies to back row."
    },
    81: {
        "name": "StepSwrd",
        "range": ["Pattern"],
        "pattern": [["0", "0", "0", "X"], ["P", "0", "0", "X"], ["0", "0", "0", "X"]],
        "damage": 160,
        "comments": "Steps 2 panels forward then uses WideSword (1x3)."
    },
    82: {
        "name": "VarSwrd",
        "range": ["Pattern"],
        "pattern": [["P", "X"]],
        "damage": 160,
        "comments": "Normal sword range. Holding A and inputting D-Pad combos changes range/effect."
    },
    83: {
        "name": "NeoVari",
        "range": ["Pattern"],
        "pattern": [["P", "X"]],
        "damage": 220,
        "comments": "Normal sword range. Holding A and inputting D-Pad combos changes range/effect."
    },
    84: {
        "name": "MoonBld",
        "range": ["Pattern"],
        "pattern": [["X", "X", "X"], ["X", "P", "X"], ["X", "X", "X"]],
        "damage": 130,
        "comments": "Hits all adjacent panels (Circular). Inflicts HP Bug."
    },
    85: {
        "name": "Muramasa",
        "range": ["Pattern"],
        "pattern": [["P", "X", "X"]],
        "damage": 0,
        "comments": "Damage equals amount of HP lost (Max HP - Current HP). Cap 500."
    },
    86: {
        "name": "MchnSwrd",
        "range": ["MatchY"],
        "damage": 200,
        "comments": "Auto-targets enemy if they are paralyzed/stunned."
    },
    87: {
        "name": "ElemSwrd",
        "range": ["MatchY"],
        "damage": 220,
        "comments": "Auto-targets enemy if they are on Grass, Ice, or Magma panels."
    },
    88: {
        "name": "AssnSwrd",
        "range": ["MatchY"],
        "damage": 240,
        "comments": "Auto-targets enemy if they have a Status Bug (Panic, etc)."
    },
    89: {
        "name": "CrakShot",
        "range": ["MatchY"],
        "damage": 60,
        "comments": "Shotgun blast. Only cracks panels that have an object on them."
    },
    90: {
        "name": "DublShot",
        "range": ["MatchY"],
        "damage": 60,
        "comments": "Fires 2 shots. Only cracks panels that have an object on them."
    },
    91: {
        "name": "TrplShot",
        "range": ["MatchY"],
        "damage": 100,
        "comments": "Fires 3 shots. Only cracks panels that have an object on them."
    },
    92: {
        "name": "WaveArm1",
        "range": ["Any"],
        "damage": 80,
        "comments": "Sends a shockwave that tracks the enemy."
    },
    93: {
        "name": "WaveArm2",
        "range": ["Any"],
        "damage": 120,
        "comments": "Sends a shockwave that tracks the enemy."
    },
    94: {
        "name": "WaveArm3",
        "range": ["Any"],
        "damage": 160,
        "comments": "Sends a shockwave that tracks the enemy."
    },
    95: {
        "name": "AuraHed1",
        "range": ["Pattern"],
        "pattern": [["P", "X", "X", "X"]],
        "damage": 130,
        "comments": "Flying head. Break attribute (Destroys Aura/Barrier). +50 Dmg if user has Barrier."
    },
    96: {
        "name": "AuraHed2",
        "range": ["Pattern"],
        "pattern": [["P", "X", "X", "X"]],
        "damage": 150,
        "comments": "Flying head. Break attribute (Destroys Aura/Barrier). +50 Dmg if user has Barrier."
    },
    97: {
        "name": "AuraHed3",
        "range": ["Pattern"],
        "pattern": [["P", "X", "X", "X"]],
        "damage": 170,
        "comments": "Flying head. Break attribute (Destroys Aura/Barrier). +50 Dmg if user has Barrier."
    },
    98: {
        "name": "LilBolr1",
        "range": ["Pattern"],
        "pattern": [["0", "0", "X", "X", "X"], ["P", "0", "X", "X", "X"], ["0", "0", "X", "X", "X"]],
        "damage": 100,
        "comments": "Object. Boils and hits area after delay or if hit by Fire."
    },
    99: {
        "name": "LilBolr2",
        "range": ["Pattern"],
        "pattern": [["0", "0", "X", "X", "X"], ["P", "0", "X", "X", "X"], ["0", "0", "X", "X", "X"]],
        "damage": 140,
        "comments": "Object. Boils and hits area after delay or if hit by Fire."
    },
    100: {
        "name": "LilBolr3",
        "range": ["Pattern"],
        "pattern": [["0", "0", "X", "X", "X"], ["P", "0", "X", "X", "X"], ["0", "0", "X", "X", "X"]],
        "damage": 180,
        "comments": "Object. Boils and hits area after delay or if hit by Fire."
    },
    101: {
        "name": "SandWrm1",
        "range": ["Any"],
        "damage": 130,
        "comments": "Attacks enemy from the panel behind them. Fails if back row."
    },
    102: {
        "name": "SandWrm2",
        "range": ["Any"],
        "damage": 150,
        "comments": "Attacks enemy from the panel behind them. Fails if back row."
    },
    103: {
        "name": "SandWrm3",
        "range": ["Any"],
        "damage": 170,
        "comments": "Attacks enemy from the panel behind them. Fails if back row."
    },
    104: {
        "name": "AirRaid1",
        "range": ["Any"],
        "damage": 10,
        "comments": "Summons plane. Shoots 10 times at enemy column."
    },
    105: {
        "name": "AirRaid2",
        "range": ["Any"],
        "damage": 10,
        "comments": "Summons plane. Shoots 14 times at enemy column."
    },
    106: {
        "name": "AirRaid3",
        "range": ["Any"],
        "damage": 10,
        "comments": "Summons plane. Shoots 18 times at enemy column."
    },
    107: {
        "name": "FireHit1",
        "range": ["Any"],
        "damage": 60,
        "comments": "Fist drops from sky on closest enemy."
    },
    108: {
        "name": "FireHit2",
        "range": ["Any"],
        "damage": 120,
        "comments": "Fist drops from sky on closest enemy."
    },
    109: {
        "name": "FireHit3",
        "range": ["Any"],
        "damage": 180,
        "comments": "Fist drops from sky on closest enemy."
    },
    110: {
        "name": "BurnSqr1",
        "range": ["Any"],
        "damage": 100,
        "comments": "Target a 2x2 square area."
    },
    111: {
        "name": "BurnSqr2",
        "range": ["Any"],
        "damage": 120,
        "comments": "Target a 2x2 square area."
    },
    112: {
        "name": "BurnSqr3",
        "range": ["Any"],
        "damage": 140,
        "comments": "Target a 2x2 square area."
    },
    113: {
        "name": "Sensor1",
        "range": ["Any"],
        "damage": 100,
        "comments": "Place sensor. Zaps enemies that cross its line of sight. Pierces Invis."
    },
    114: {
        "name": "Sensor2",
        "range": ["Any"],
        "damage": 130,
        "comments": "Place sensor. Zaps enemies that cross its line of sight. Pierces Invis."
    },
    115: {
        "name": "Sensor3",
        "range": ["Any"],
        "damage": 160,
        "comments": "Place sensor. Zaps enemies that cross its line of sight. Pierces Invis."
    },
    116: {
        "name": "Boomer",
        "range": ["Pattern"],
        "pattern": [["X", "X", "X", "X", "X", "X"], ["0", "0", "0", "0", "0", "X"], ["X", "X", "X", "X", "X", "X"]],
        "damage": 100,
        "comments": "Boomerang follows the perimeter of the field."
    },
    117: {
        "name": "HiBoomer",
        "range": ["Pattern"],
        "pattern": [["X", "X", "X", "X", "X", "X"], ["0", "0", "0", "0", "0", "X"], ["X", "X", "X", "X", "X", "X"]],
        "damage": 140,
        "comments": "Boomerang follows the perimeter of the field."
    },
    118: {
        "name": "M-Boomer",
        "range": ["Pattern"],
        "pattern": [["X", "X", "X", "X", "X", "X"], ["0", "0", "0", "0", "0", "X"], ["X", "X", "X", "X", "X", "X"]],
        "damage": 170,
        "comments": "Boomerang follows the perimeter of the field."
    },
    119: {
        "name": "Lance",
        "range": ["BackColumn"],
        "damage": 150,
        "comments": "Lances fall on the back column."
    },
    120: {
        "name": "GolmHit1",
        "range": ["Pattern"],
        "pattern": [["0", "X"], ["P", "X"], ["0", "X"]],
        "damage": 140,
        "comments": "Target closest enemy. Hits 1x3 column. Break attribute."
    },
    121: {
        "name": "GolmHit2",
        "range": ["Pattern"],
        "pattern": [["0", "X"], ["P", "X"], ["0", "X"]],
        "damage": 190,
        "comments": "Target closest enemy. Hits 1x3 column. Break attribute."
    },
    122: {
        "name": "GolmHit3",
        "range": ["Pattern"],
        "pattern": [["0", "X"], ["P", "X"], ["0", "X"]],
        "damage": 250,
        "comments": "Target closest enemy. Hits 1x3 column. Break attribute."
    },
    123: {
        "name": "IronShl1",
        "range": ["MatchY"],
        "damage": 70,
        "comments": "Cannonball hits back column. Hits 2 times."
    },
    124: {
        "name": "IronShl2",
        "range": ["MatchY"],
        "damage": 100,
        "comments": "Cannonball hits back column. Hits 2 times."
    },
    125: {
        "name": "IronShl3",
        "range": ["MatchY"],
        "damage": 130,
        "comments": "Cannonball hits back column. Hits 2 times."
    },
    126: {
        "name": "AirSpin1",
        "range": ["MatchY"],
        "damage": 50,
        "comments": "Slides forward. Hitting it with Wind adds spin/damage."
    },
    127: {
        "name": "AirSpin2",
        "range": ["MatchY"],
        "damage": 50,
        "comments": "Slides forward. Hitting it with Wind adds spin/damage."
    },
    128: {
        "name": "AirSpin3",
        "range": ["MatchY"],
        "damage": 50,
        "comments": "Slides forward. Hitting it with Wind adds spin/damage."
    },
    129: {
        "name": "Wind",
        "range": ["Any"],
        "damage": 0,
        "comments": "Blower object. Pushes enemies to back row."
    },
    130: {
        "name": "Fan",
        "range": ["Any"],
        "damage": 0,
        "comments": "Vacuum object. Pulls enemies to front row."
    },
    131: {
        "name": "Rflectr1",
        "range": ["Any"],
        "damage": 60,
        "comments": "Shield. Reflects damage back at attacker."
    },
    132: {
        "name": "Rflectr2",
        "range": ["Any"],
        "damage": 120,
        "comments": "Shield. Reflects damage back at attacker."
    },
    133: {
        "name": "Rflectr3",
        "range": ["Any"],
        "damage": 200,
        "comments": "Shield. Reflects damage back at attacker."
    },
    134: {
        "name": "Snake",
        "range": ["Any"],
        "damage": 30,
        "comments": "Snakes emerge from any holes on the field."
    },
    135: {
        "name": "SumnBlk1",
        "range": ["Any"],
        "damage": 160,
        "comments": "Must be used next to a hole. Summons object to attack."
    },
    136: {
        "name": "SumnBlk2",
        "range": ["Any"],
        "damage": 200,
        "comments": "Must be used next to a hole. Summons object to attack."
    },
    137: {
        "name": "SumnBlk3",
        "range": ["Any"],
        "damage": 260,
        "comments": "Must be used next to a hole. Summons object to attack."
    },
    138: {
        "name": "NumbrBl",
        "range": ["MatchY"],
        "damage": 0,
        "comments": "Damage equals last 2 digits of user HP."
    },
    139: {
        "name": "Meteors",
        "range": ["Any"],
        "damage": 40,
        "comments": "Rains 30 meteors on enemy field."
    },
    140: {
        "name": "JustcOne",
        "range": ["Pattern"],
        "pattern": [["0", "0", "0", "X", "X", "X"], ["0", "0", "0", "X", "X", "X"], ["0", "0", "0", "X", "X", "X"]],
        "damage": 220,
        "comments": "Giant fist punches 3x3 area. Break attribute."
    },
    141: {
        "name": "Magnum",
        "range": ["Any"],
        "damage": 130,
        "comments": "Cursor moves through enemy field. Press A to break panel."
    },
    142: {
        "name": "CircGun",
        "range": ["Any"],
        "damage": 150,
        "comments": "Cursor moves around perimeter. Press A to fire."
    },
    143: {
        "name": "RockCube",
        "range": ["Any"],
        "damage": 0,
        "comments": "Places a 200HP defensive cube."
    },
    144: {
        "name": "TimeBom1",
        "range": ["Any"],
        "damage": 150,
        "comments": "Places bomb. Explodes after 3 seconds."
    },
    145: {
        "name": "Mine",
        "range": ["Any"],
        "damage": 200,
        "comments": "Hides a trap on an enemy panel. Explodes when stepped on."
    },
    146: {
        "name": "Fanfare",
        "range": ["Any"],
        "damage": 0,
        "comments": "Music box. Grants temporary Invincibility."
    },
    147: {
        "name": "Discord",
        "range": ["Any"],
        "damage": 0,
        "comments": "Music box. Confuses enemies."
    },
    148: {
        "name": "Timpani",
        "range": ["Any"],
        "damage": 0,
        "comments": "Music box. Immobilizes enemies."
    },
    149: {
        "name": "Silence",
        "range": ["Any"],
        "damage": 0,
        "comments": "Music box. Blinds enemies."
    },
    150: {
        "name": "VDoll",
        "range": ["Pattern"],
        "pattern": [["P", "0", "0", "X"]],
        "damage": 0,
        "comments": "Doll links to an enemy. Hitting doll damages enemy."
    },
    151: {
        "name": "Guardian",
        "range": ["Any"],
        "damage": 200,
        "comments": "Statue. Punishes anyone who attacks it with lightning."
    },
    152: {
        "name": "Anubis",
        "range": ["Any"],
        "damage": 0,
        "comments": "Statue. Poisons the field, draining enemy HP rapidly."
    },
    153: {
        "name": "Otenko*",
        "range": ["Any"],
        "damage": 0,
        "comments": "Statue. Raises attack power of chips."
    },
    154: {
        "name": "Recov10",
        "range": ["Any"],
        "damage": 0,
        "comments": "Recover 10 HP."
    },
    155: {
        "name": "Recov30",
        "range": ["Any"],
        "damage": 0,
        "comments": "Recover 30 HP."
    },
    156: {
        "name": "Recov50",
        "range": ["Any"],
        "damage": 0,
        "comments": "Recover 50 HP."
    },
    157: {
        "name": "Recov80",
        "range": ["Any"],
        "damage": 0,
        "comments": "Recover 80 HP."
    },
    158: {
        "name": "Recov120",
        "range": ["Any"],
        "damage": 0,
        "comments": "Recover 120 HP."
    },
    159: {
        "name": "Recov150",
        "range": ["Any"],
        "damage": 0,
        "comments": "Recover 150 HP."
    },
    160: {
        "name": "Recov200",
        "range": ["Any"],
        "damage": 0,
        "comments": "Recover 200 HP."
    },
    161: {
        "name": "Recov300",
        "range": ["Any"],
        "damage": 0,
        "comments": "Recover 300 HP."
    },
    162: {
        "name": "PanlGrab",
        "range": ["Any"],
        "damage": 0,
        "comments": "Steals the first enemy panel in the row."
    },
    163: {
        "name": "AreaGrab",
        "range": ["Any"],
        "damage": 0,
        "comments": "Steals the first column of the enemy area."
    },
    164: {
        "name": "GrabBnsh",
        "range": ["Any"],
        "damage": 20,
        "comments": "Deals damage for every stolen panel."
    },
    165: {
        "name": "GrabRvng",
        "range": ["Any"],
        "damage": 40,
        "comments": "Deals damage for every stolen panel."
    },
    166: {
        "name": "PnlRetrn",
        "range": ["Any"],
        "damage": 0,
        "comments": "Resets all panels to default ownership and state."
    },
    167: {
        "name": "Geddon",
        "range": ["Any"],
        "damage": 0,
        "comments": "Cracks/Breaks all empty panels on the field."
    },
    168: {
        "name": "HolyPanl",
        "range": ["Any"],
        "damage": 0,
        "comments": "Turns panel in front into Holy Panel (0.5x Dmg)."
    },
    169: {
        "name": "Snctuary",
        "range": ["Any"],
        "damage": 0,
        "comments": "Turns all player panels into Holy Panels."
    },
    170: {
        "name": "ComingRd",
        "range": ["Any"],
        "damage": 0,
        "comments": "Pulls enemy to front row (if possible)."
    },
    171: {
        "name": "GoingRd",
        "range": ["Any"],
        "damage": 0,
        "comments": "Pushes enemy to back row (if possible)."
    },
    172: {
        "name": "SloGauge",
        "range": ["Any"],
        "damage": 0,
        "comments": "Slows down the Custom Gauge fill rate."
    },
    173: {
        "name": "FstGauge",
        "range": ["Any"],
        "damage": 0,
        "comments": "Speeds up the Custom Gauge fill rate."
    },
    174: {
        "name": "FullCust",
        "range": ["Any"],
        "damage": 0,
        "comments": "Instantly fills the Custom Gauge."
    },
    175: {
        "name": "BusterUp",
        "range": ["Any"],
        "damage": 0,
        "comments": "Increases MegaBuster attack by 1."
    },
    176: {
        "name": "BugFix",
        "range": ["Any"],
        "damage": 0,
        "comments": "Removes current bugs from MegaMan."
    },
    177: {
        "name": "Invisibl",
        "range": ["Any"],
        "damage": 0,
        "comments": "Grants temporary Invisibility."
    },
    178: {
        "name": "Barrier",
        "range": ["Any"],
        "damage": 0,
        "comments": "Nullifies 10 HP of damage."
    },
    179: {
        "name": "Barr100",
        "range": ["Any"],
        "damage": 0,
        "comments": "Nullifies 100 HP of damage."
    },
    180: {
        "name": "Barr200",
        "range": ["Any"],
        "damage": 0,
        "comments": "Nullifies 200 HP of damage."
    },
    181: {
        "name": "BblWrap",
        "range": ["Any"],
        "damage": 0,
        "comments": "Regenerating barrier. Pop with Elec."
    },
    182: {
        "name": "LifeAur",
        "range": ["Any"],
        "damage": 0,
        "comments": "Nullifies all attacks dealing < 200 dmg."
    },
    183: {
        "name": "MagCoil",
        "range": ["Any"],
        "damage": 0,
        "comments": "Pulls specified enemy towards you."
    },
    184: {
        "name": "WhiCapsl",
        "range": ["Any"],
        "damage": 0,
        "comments": "Next chip paralyzes."
    },
    185: {
        "name": "Uninstll",
        "range": ["Any"],
        "damage": 0,
        "comments": "Next chip removes enemy NaviCust programs."
    },
    186: {
        "name": "AntiNavi",
        "range": ["Any"],
        "damage": 0,
        "comments": "Trap. Punishes enemy for using a Navi chip."
    },
    187: {
        "name": "AntiDmg",
        "range": ["Any"],
        "damage": 100,
        "comments": "Trap. Punishes enemy for dealing damage. Throws shurikens."
    },
    188: {
        "name": "AntiSwrd",
        "range": ["Any"],
        "damage": 100,
        "comments": "Trap. Punishes enemy for using Sword chips."
    },
    189: {
        "name": "AntiRecv",
        "range": ["Any"],
        "damage": 0,
        "comments": "Trap. Punishes enemy for recovering HP."
    },
    190: {
        "name": "CopyDmg",
        "range": ["Any"],
        "damage": 0,
        "comments": "Links damage from one enemy to another."
    },
    191: {
        "name": "LifeSync",
        "range": ["Any"],
        "damage": 0,
        "comments": "Equalizes HP of all enemies."
    },
    192: {
        "name": "Atk+10",
        "range": ["Any"],
        "damage": 0,
        "comments": "Adds +10 damage to the preceding chip."
    },
    193: {
        "name": "Navi+20",
        "range": ["Any"],
        "damage": 0,
        "comments": "Adds +20 damage to the preceding Navi chip."
    },
    194: {
        "name": "ColorPt",
        "range": ["Any"],
        "damage": 0,
        "comments": "Adds +10 damage to preceding chip. Uses 1000Z."
    },
    195: {
        "name": "Atk+30",
        "range": ["Any"],
        "damage": 0,
        "comments": "Adds +30 damage to the preceding chip."
    },
    196: {
        "name": "DblPoint",
        "range": ["Any"],
        "damage": 0,
        "comments": "Adds +20/40/60 damage based on sacrificed chips."
    },
    197: {
        "name": "ElemTrap",
        "range": ["Any"],
        "damage": 240,
        "comments": "Trap. Reacts to elemental attacks."
    },
    198: {
        "name": "ColArmy",
        "range": ["Any"],
        "damage": 40,
        "comments": "Turns objects into Army. Shoots 3 times."
    },
    199: {
        "name": "BlzrdBal",
        "range": ["Any"],
        "damage": 150,
        "comments": "Snowball rolls down row. Absorbs obstacles for more damage."
    },
    200: {
        "name": "TimeBom2",
        "range": ["Any"],
        "damage": 190,
        "comments": "Places bomb. Explodes after 3 seconds."
    },
    201: {
        "name": "TimeBom3",
        "range": ["Any"],
        "damage": 230,
        "comments": "Places bomb. Explodes after 3 seconds."
    },
    202: {
        "name": "BigBomb",
        "range": ["Pattern"],
        "pattern": [["0", "0", "X", "X", "X"], ["P", "0", "X", "X", "X"], ["0", "0", "X", "X", "X"]],
        "damage": 140,
        "comments": "Massive 3x3 explosion."
    },
    221: {
        "name": "Roll",
        "range": ["Any"],
        "damage": 20,
        "comments": "Hits 3 times. Heals user."
    },
    222: {
        "name": "Roll2",
        "range": ["Any"],
        "damage": 40,
        "comments": "Hits 3 times. Heals user."
    },
    223: {
        "name": "Roll3",
        "range": ["Any"],
        "damage": 60,
        "comments": "Hits 3 times. Heals user."
    },
    224: {
        "name": "ProtoMan",
        "range": ["Any"],
        "damage": 150,
        "comments": "Steps forward to slash enemy."
    },
    225: {
        "name": "ProtoMnEX",
        "range": ["Any"],
        "damage": 170,
        "comments": "Steps forward to slash enemy."
    },
    226: {
        "name": "ProtoMnSP",
        "range": ["Any"],
        "damage": 190,
        "comments": "Steps forward to slash enemy."
    },
    227: {
        "name": "HeatMan",
        "range": ["Pattern"],
        "pattern": [["0", "0", "X", "X"], ["P", "X", "X", "X"], ["0", "0", "X", "X"]],
        "damage": 100,
        "comments": "Flame wave. Fire element."
    },
    228: {
        "name": "HeatManEX",
        "range": ["Pattern"],
        "pattern": [["0", "0", "X", "X"], ["P", "X", "X", "X"], ["0", "0", "X", "X"]],
        "damage": 130,
        "comments": "Flame wave. Fire element."
    },
    229: {
        "name": "HeatManSP",
        "range": ["Pattern"],
        "pattern": [["0", "0", "X", "X"], ["P", "X", "X", "X"], ["0", "0", "X", "X"]],
        "damage": 160,
        "comments": "Flame wave. Fire element."
    },
    230: {
        "name": "ElecMan",
        "range": ["Any"],
        "damage": 120,
        "comments": "Hits 2 times. Strikes center then outer ring."
    },
    231: {
        "name": "ElecManEX",
        "range": ["Any"],
        "damage": 140,
        "comments": "Hits 2 times. Strikes center then outer ring."
    },
    232: {
        "name": "ElecManSP",
        "range": ["Any"],
        "damage": 160,
        "comments": "Hits 2 times. Strikes center then outer ring."
    },
    233: {
        "name": "SlashMan",
        "range": ["Any"],
        "damage": 80,
        "comments": "Auto-targets with Kunai."
    },
    234: {
        "name": "SlashMnEX",
        "range": ["Any"],
        "damage": 100,
        "comments": "Auto-targets with Kunai."
    },
    235: {
        "name": "SlashMnSP",
        "range": ["Any"],
        "damage": 120,
        "comments": "Auto-targets with Kunai."
    },
    236: {
        "name": "EraseMan",
        "range": ["Pattern", "MatchY"],
        "pattern": [["0", "0", "X"], ["0", "X", "0"], ["P", "0", "0"], ["0", "X", "X"], ["0", "0", "X"]],
        "damage": 120,
        "comments": "Hex Scythe beam. Paralyzes."
    },
    237: {
        "name": "EraseMnEX",
        "range": ["Pattern", "MatchY"],
        "pattern": [["0", "0", "X"], ["0", "X", "0"], ["P", "0", "0"], ["0", "X", "X"], ["0", "0", "X"]],
        "damage": 140,
        "comments": "Hex Scythe beam. Paralyzes."
    },
    238: {
        "name": "EraseMnSP",
        "range": ["Pattern", "MatchY"],
        "pattern": [["0", "0", "X"], ["0", "X", "0"], ["P", "0", "0"], ["0", "X", "X"], ["0", "0", "X"]],
        "damage": 160,
        "comments": "Hex Scythe beam. Paralyzes."
    },
    239: {
        "name": "ChrgeMan",
        "range": ["MatchY"],
        "damage": 60,
        "comments": "Train rushes forward."
    },
    240: {
        "name": "ChrgeMnEX",
        "range": ["MatchY"],
        "damage": 70,
        "comments": "Train rushes forward."
    },
    241: {
        "name": "ChrgeMnSP",
        "range": ["MatchY"],
        "damage": 80,
        "comments": "Train rushes forward."
    },
    242: {
        "name": "AquaMan",
        "range": ["BackColumn"],
        "damage": 50,
        "comments": "Hits 3 times in a line on back column."
    },
    243: {
        "name": "AquaMnEX",
        "range": ["BackColumn"],
        "damage": 60,
        "comments": "Hits 3 times in a line on back column."
    },
    244: {
        "name": "AquaMnSP",
        "range": ["BackColumn"],
        "damage": 70,
        "comments": "Hits 3 times in a line on back column."
    },
    245: {
        "name": "TmhkMan",
        "range": ["Pattern"],
        "pattern": [["0", "X", "X"], ["P", "X", "X"], ["0", "X", "X"]],
        "damage": 140,
        "comments": "Massive swing. 2x3 area."
    },
    246: {
        "name": "TmhkManEX",
        "range": ["Pattern"],
        "pattern": [["0", "X", "X"], ["P", "X", "X"], ["0", "X", "X"]],
        "damage": 160,
        "comments": "Massive swing. 2x3 area."
    },
    247: {
        "name": "TmhkManSP",
        "range": ["Pattern"],
        "pattern": [["0", "X", "X"], ["P", "X", "X"], ["0", "X", "X"]],
        "damage": 180,
        "comments": "Massive swing. 2x3 area."
    },
    248: {
        "name": "TenguMan",
        "range": ["Pattern"],
        "pattern": [["0", "0", "X", "X", "0", "0"], ["0", "0", "X", "X", "X", "0"], ["X", "X", "X", "X", "X", "X"]],
        "damage": 70,
        "comments": "Dash attack."
    },
    249: {
        "name": "TenguMnEX",
        "range": ["Pattern"],
        "pattern": [["0", "0", "X", "X", "0", "0"], ["0", "0", "X", "X", "X", "0"], ["X", "X", "X", "X", "X", "X"]],
        "damage": 90,
        "comments": "Dash attack."
    },
    250: {
        "name": "TenguMnSP",
        "range": ["Pattern"],
        "pattern": [["0", "0", "X", "X", "0", "0"], ["0", "0", "X", "X", "X", "0"], ["X", "X", "X", "X", "X", "X"]],
        "damage": 110,
        "comments": "Dash attack."
    },
    251: {
        "name": "GrndMan",
        "range": ["Any"],
        "damage": 60,
        "comments": "Spawns 3 sets of 3 rocks. 1 rock per set hits target."
    },
    252: {
        "name": "GrndManEX",
        "range": ["Any"],
        "damage": 70,
        "comments": "Spawns 3 sets of 3 rocks. 1 rock per set hits target."
    },
    253: {
        "name": "GrndManSP",
        "range": ["Any"],
        "damage": 80,
        "comments": "Spawns 3 sets of 3 rocks. 1 rock per set hits target."
    },
    254: {
        "name": "DustMan",
        "range": ["Any"],
        "damage": 110,
        "comments": "Sucks in and shoots junk. Cracks panel."
    },
    255: {
        "name": "DustManEX",
        "range": ["Any"],
        "damage": 130,
        "comments": "Sucks in and shoots junk. Cracks panel."
    },
    256: {
        "name": "DustManSP",
        "range": ["Any"],
        "damage": 150,
        "comments": "Sucks in and shoots junk. Cracks panel."
    },
    257: {
        "name": "BlastMan",
        "range": ["Any"],
        "damage": 120,
        "comments": "3 fireballs. Above, behind, and below user."
    },
    258: {
        "name": "BlastMnEX",
        "range": ["Any"],
        "damage": 140,
        "comments": "3 fireballs. Above, behind, and below user."
    },
    259: {
        "name": "BlastMnSP",
        "range": ["Any"],
        "damage": 150,
        "comments": "3 fireballs. Above, behind, and below user."
    },
    250: {
        "name": "DiveMan",
        "range": ["Pattern"],
        "pattern": [["0", "X", "X"], ["P", "X", "X"], ["0", "X", "X"]],
        "damage": 130,
        "comments": "Sends waves down row."
    },
    261: {
        "name": "DiveManEX",
        "range": ["Pattern"],
        "pattern": [["0", "X", "X"], ["P", "X", "X"], ["0", "X", "X"]],
        "damage": 150,
        "comments": "Sends waves down row."
    },
    262: {
        "name": "DiveManSP",
        "range": ["Pattern"],
        "pattern": [["0", "X", "X"], ["P", "X", "X"], ["0", "X", "X"]],
        "damage": 170,
        "comments": "Sends waves down row."
    },
    263: {
        "name": "CrcusMan",
        "range": ["Pattern"],
        "pattern": [["P", "0", "0", "X"]],
        "damage": 20,
        "comments": "Hits 6 times. Pierces Invis."
    },
    264: {
        "name": "CrcusMnEX",
        "range": ["Pattern"],
        "pattern": [["P", "0", "0", "X"]],
        "damage": 25,
        "comments": "Hits 6 times. Pierces Invis."
    },
    265: {
        "name": "CrcusMnSP",
        "range": ["Pattern"],
        "pattern": [["P", "0", "0", "X"]],
        "damage": 30,
        "comments": "Hits 6 times. Pierces Invis."
    },
    266: {
        "name": "JudgeMan",
        "range": ["Pattern"],
        "pattern": [["P", "X", "X", "X"]],
        "damage": 100,
        "comments": "Paralyzes. Spawns books to attack."
    },
    267: {
        "name": "JudgeMnEX",
        "range": ["Pattern"],
        "pattern": [["P", "X", "X", "X"]],
        "damage": 120,
        "comments": "Paralyzes. Spawns books to attack."
    },
    268: {
        "name": "JudgeMnSP",
        "range": ["Pattern"],
        "pattern": [["P", "X", "X", "X"]],
        "damage": 140,
        "comments": "Paralyzes. Spawns books to attack."
    },
    269: {
        "name": "ElmntMan",
        "range": ["Any"],
        "damage": 100,
        "comments": "Element cycles (Fire/Aqua/Elec/Wood). Press A to stop."
    },
    260: {
        "name": "ElmntMnEX",
        "range": ["Any"],
        "damage": 120,
        "comments": "Element cycles (Fire/Aqua/Elec/Wood). Press A to stop."
    },
    271: {
        "name": "ElmntMnSP",
        "range": ["Any"],
        "damage": 140,
        "comments": "Element cycles (Fire/Aqua/Elec/Wood). Press A to stop."
    },
    272: {
        "name": "Colonel",
        "range": ["Pattern"],
        "pattern": [["0", "0", "X", "X", "X"], ["P", "0", "0", "X", "0"], ["0", "0", "X", "X", "X"]],
        "damage": 160,
        "comments": "Cuts in Z-shape."
    },
    273: {
        "name": "ColonelEX",
        "range": ["Pattern"],
        "pattern": [["0", "0", "X", "X", "X"], ["P", "0", "0", "X", "0"], ["0", "0", "X", "X", "X"]],
        "damage": 180,
        "comments": "Cuts in Z-shape."
    },
    274: {
        "name": "ColonelSP",
        "range": ["Pattern"],
        "pattern": [["0", "0", "X", "X", "X"], ["P", "0", "0", "X", "0"], ["0", "0", "X", "X", "X"]],
        "damage": 200,
        "comments": "Cuts in Z-shape."
    },
    275: {
        "name": "Count*",
        "range": ["Any"],
        "damage": 20,
        "comments": "Hits 5+ times."
    },
    276: {
        "name": "CountEX*",
        "range": ["Any"],
        "damage": 25,
        "comments": "Hits 5+ times."
    },
    277: {
        "name": "CountSP*",
        "range": ["Any"],
        "damage": 30,
        "comments": "Hits 5+ times."
    },
    278: {
        "name": "Django*",
        "range": ["Any"],
        "damage": 0,
        "comments": "Solar gun attack."
    },
    279: {
        "name": "Django2*",
        "range": ["Any"],
        "damage": 0,
        "comments": "Solar gun attack."
    },
    270: {
        "name": "Django3*",
        "range": ["Any"],
        "damage": 0,
        "comments": "Solar gun attack."
    },
    300: {
        "name": "Bass",
        "range": ["Any"],
        "damage": 60,
        "comments": "Hits up to 8 times. Rake attack."
    },
    301: {
        "name": "BigHook",
        "range": ["Pattern"],
        "pattern": [["0", "0", "0", "X", "X", "X"], ["0", "0", "0", "X", "X", "X"], ["0", "0", "0", "X", "X", "X"]],
        "damage": 240,
        "comments": "Giant hook. Break attribute. Hits 2 cols."
    },
    302: {
        "name": "DeltaRay",
        "range": ["Any"],
        "damage": 260,
        "comments": "Press A to slash up to 3 times."
    },
    303: {
        "name": "ColForce",
        "range": ["Any"],
        "damage": 30,
        "comments": "Spawns soldiers. Hits 3 times. Paralyzes."
    },
    304: {
        "name": "BugRSwrd",
        "range": ["Any"],
        "damage": 200,
        "comments": "Consumes Bug Frags. 2x3 range."
    },
    311: {
        "name": "Gregar*",
        "range": ["Any"],
        "damage": 300,
        "comments": "Rockfall hits up to 5 times. Destroys all obstacles."
    },
    305: {
        "name": "BassAnly",
        "range": ["Pattern"],
        "pattern": [["P", "X"]],
        "damage": 160,
        "comments": "Hits 4 times."
    },
    306: {
        "name": "MetrKnuk",
        "range": ["Any"],
        "damage": 100,
        "comments": "Rains 16 fists. Break attribute."
    },
    307: {
        "name": "CrossDiv",
        "range": ["MatchY"],
        "damage": 250,
        "comments": "Cuts X-shape. 500 dmg to center."
    },
    308: {
        "name": "HubBatc",
        "range": ["Any"],
        "damage": 0,
        "comments": "Grants Max Stats and various buffs."
    },
    309: {
        "name": "BgDthThd",
        "range": ["Any"],
        "damage": 200,
        "comments": "Consumes Bug Frags. Tracking Thunder."
    },
    310: {
        "name": "DblBeast*",
        "range": ["Any"],
        "damage": 420,
        "comments": "4x1 attack."
    },
    312: {
        "name": "Falzar*",
        "range": ["Any"],
        "damage": 100,
        "comments": "Shoots 10 feathers. Hits up to 5 times."
    }
}