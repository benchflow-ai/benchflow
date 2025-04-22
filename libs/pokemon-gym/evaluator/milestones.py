# --- Pokémon Acquisition Difficulty Ratings (0.5 - 10.5) ---
# Higher score means rarer, harder to obtain, or acquired later in the game
pokemon_difficulty_ratings_refined = {
    # --- Difficulty 0.5: Virtually no extra effort (free or ubiquitous encounters)
    'PIDGEY': 0.5,      # Extremely common in grassy areas
    'RATTATA': 0.5,     # Ubiquitous early on
    'WEEDLE': 0.5,      # Common in Viridian Forest
    'CATERPIE': 0.5,    # Abundant in early forests
    'KAKUNA': 0.5,      # Easily obtained via evolving Weedle or wild encounter
    'METAPOD': 0.5,     # Similarly common as Caterpie
    'ZUBAT': 0.5,       # Found in nearly every cave
    'BULBASAUR': 0.5,   # Starter Pokémon (free choice at game start)
    'CHARMANDER': 0.5,  # Starter Pokémon (free choice at game start)
    'SQUIRTLE': 0.5,    # Starter Pokémon (free choice at game start)

    # --- Difficulty 1.0: Very common wild encounters
    'SPEAROW': 1.0,     # Frequently encountered on early routes
    'GEODUDE': 1.0,     # Abundant in early caves
    'NIDORAN_M': 1.0,   # Common on early routes (in one version)
    'NIDORAN_F': 1.0,   # Common on early routes (in one version)
    'MAGIKARP': 1.0,    # Common fishing encounter in nearly all waters

    # --- Difficulty 1.5: Common but with version restrictions or minor evolution steps
    'EKANS': 1.5,       # Red version exclusive; easy if playing Red
    'SANDSHREW': 1.5,   # Blue version exclusive; easy in that version
    'MANKEY': 1.5,      # Only in Red/Yellow; common in those games
    'MEOWTH': 1.5,      # Blue version exclusive; encountered regularly there
    'ODDISH': 1.5,      # Red exclusive; encountered in limited areas
    'BELLSPROUT': 1.5,  # Blue exclusive; encountered in specific areas
    'POLIWAG': 1.5,     # Common fishing encounter; slightly less ubiquitous
    'GOLDEEN': 1.5,     # Common in water; similar to Poliwag
    'BEEDRILL': 1.5,    # Typically obtained by evolving Weedle (no extra effort)
    'BUTTERFREE': 1.5,  # Obtained by evolving Caterpie; very easy to acquire

    # --- Difficulty 2.0: Evolved forms that are less common in the wild
    'PIDGEOTTO': 2.0,   # Evolved form of Pidgey; less common wild encounter
    'RATICATE': 2.0,    # Evolved form of Rattata; encountered less often
    'GOLBAT': 2.0,      # Wild evolution of Zubat; moderately common
    'GLOOM': 2.0,       # Evolves from Oddish; less abundant than base form
    'WEEPINBELL': 2.0,  # Evolves from Bellsprout; found with lower frequency
    'POLIWHIRL': 2.0,   # Evolves from Poliwag; not as common as Poliwag

    # --- Difficulty 2.5: Location-specific encounters or those requiring travel
    'DIGLETT': 2.5,     # Only found in Diglett’s Cave; extra area access required
    'DROWZEE': 2.5,     # Only on select routes; moderate encounter rate
    'GASTLY': 2.5,      # Found in Pokémon Tower; extra exploration needed
    'TENTACOOL': 2.5,   # Encountered while surfing; requires water travel
    'MACHOP': 2.5,      # Found in Rock Tunnel/Victory Road; extra progression required
    'GYARADOS': 2.5,    # Though evolved from Magikarp, wild encounters are rare

    # --- Difficulty 3.0: Moderately rare or needing specific conditions/tools
    'FEAROW': 3.0,      # Evolved from Spearow; seen less frequently
    'ARBOK': 3.0,       # Evolves from Ekans; extra evolution step increases difficulty
    'SANDSLASH': 3.0,   # Evolves from Sandshrew; similar reasoning as Arbok
    'NIDORINO': 3.0,    # Evolves from Nidoran♂; not encountered as commonly as the base form
    'NIDORINA': 3.0,    # Evolves from Nidoran♀; similarly rarer
    'CLEFAIRY': 3.0,    # Requires use of the Silph Scope in Mt. Moon; extra step needed
    'JIGGLYPUFF': 3.0,  # Encountered on select routes; not very frequent
    'PERSIAN': 3.0,     # Evolves from Meowth; rarity due to version exclusivity
    'PRIMEAPE': 3.0,    # Evolves from Mankey; less common in the wild

    # --- Difficulty 3.5: Involves extra items (evolution stones) or limited wild availability
    'ONIX': 3.5,        # Found in Rock Tunnel; relatively scarce wild encounter
    'SLOWPOKE': 3.5,    # Only in specific water-adjacent areas; limited spots
    'PSYDUCK': 3.5,     # Encountered in limited water areas
    'GROWLITHE': 3.5,   # Red exclusive; available only on certain routes
    'VULPIX': 3.5,      # Blue exclusive; similarly restricted in availability
    'KRABBY': 3.5,      # Requires fishing in certain areas; moderate rarity
    'MAGNEMITE': 3.5,   # Found near the Power Plant; not common outside that area
    'VOLTORB': 3.5,     # Similar to Magnemite; limited to specific zones
    'GRIMER': 3.5,      # Found in Pokémon Mansion/Power Plant; less frequently seen
    'KOFFING': 3.5,     # Limited encounters in specific routes or mansions
    'PARASECT': 3.5,    # Evolves from Paras; extra evolution step adds difficulty
    'NIDOKING': 3.5,    # Requires Moon Stone evolution of Nidorino; not catchable in the wild
    'NIDOQUEEN': 3.5,   # Same as Nidoking; extra item needed
    'CLEFABLE': 3.5,    # Evolves from Clefairy with a Moon Stone; extra resource needed
    'WIGGLYTUFF': 3.5,  # Evolves from Jigglypuff using a Moon Stone; similarly extra
    'SEAKING': 3.5,     # Evolves from Goldeen; extra evolution requirement

    # --- Difficulty 4.0: Even scarcer wild encounters or higher-level evolutions
    'GRAVELER': 4.0,    # Evolves from Geodude; wild encounters of this form are rare
    'MACHOKE': 4.0,     # Evolves from Machop; not typically found in the wild directly
    'HAUNTER': 4.0,     # Evolves from Gastly; rarer and often requires trade evolution for Gengar
    'HYPNO': 4.0,       # Evolves from Drowzee; lower encounter rates
    'PONYTA': 4.0,      # Found on select routes/Safari Zone; relatively limited
    'DODUO': 4.0,       # Encountered in the Safari Zone or select routes; not common
    'SEEL': 4.0,        # Found in the Seafoam Islands; extra area access required
    'PIDGEOT': 4.0,     # Final evolution of Pidgey; rarely seen wild
    'PIKACHU': 4.0,     # Very rare wild encounter in Viridian Forest/Power Plant

    # --- Difficulty 4.5: Special environments or fossil-based acquisition
    'EXEGGCUTE': 4.5,   # Found in the Safari Zone; limited by Safari mechanics
    'CUBONE': 4.5,      # Only obtainable via fossil revival; one-time opportunity
    'RHYHORN': 4.5,     # Found in Safari Zone/later caves; less common
    'STARYU': 4.5,      # Specific fishing encounter; moderately rare
    'SHELLDER': 4.5,    # Same as Staryu; limited to particular spots
    'HORSEA': 4.5,      # Similar to Shellder; specific fishing areas only
    'VENONAT': 4.5,     # Found in the Safari Zone; limited encounter rates

    # --- Difficulty 5.0: Requires an evolution stone (extra resource cost)
    'RAICHU': 5.0,      # Must evolve Pikachu using a Thunder Stone; extra item needed
    'NINETALES': 5.0,   # Requires a Fire Stone on Vulpix; added acquisition step
    'ARCANINE': 5.0,    # Fire Stone evolution from Growlithe; version-restricted and extra cost
    'VILEPLUME': 5.0,   # Evolved from Gloom with a Leaf Stone; extra resource required
    'VICTREEBEL': 5.0,  # Evolved from Weepinbell using a Leaf Stone; extra step
    'CLOYSTER': 5.0,    # Evolves from Shellder with a Water Stone; limited by stone availability
    'STARMIE': 5.0,     # Evolved from Staryu via Water Stone; similar difficulty as Cloyster
    'EXEGGUTOR': 5.0,   # Requires a Leaf Stone to evolve Exeggcute; extra item needed
    'MAROWAK': 5.0,     # Evolved from Cubone; fossil revival already limits its availability

    # --- Difficulty 5.5: Requires trade evolution (extra dependency on trading)
    'GENGAR': 5.5,      # Must be traded from Haunter; trading is a significant hurdle
    'ALAKAZAM': 5.5,    # Trade evolution from Kadabra; extra trade requirement
    'MACHAMP': 5.5,     # Trade evolution from Machoke; relies on trading
    'GOLEM': 5.5,       # Trade evolution from Graveler; extra step needed
    'DRAGONAIR': 5.5,   # High-level evolution of Dratini; requires extensive leveling
    'KABUTOPS': 5.5,    # Fossil evolution from Kabuto; limited fossil opportunity
    'OMASTAR': 5.5,     # Fossil evolution from Omanyte; similar fossil limitation
    'RHYDON': 5.5,      # Evolves from Rhyhorn; rarely encountered wild

    # --- Difficulty 9.0: Late-game, static or non-standard methods
    'SNORLAX': 9.0,     # Static encounter with only two in the game
    'DRAGONITE': 9.0,   # Final evolution of Dragonair; requires extensive leveling
    'AERODACTYL': 9.0,  # Only available via fossil revival; one-time catch
    'MISSINGNO_1F': 9.0, 'MISSINGNO_20': 9.0, 'MISSINGNO_32': 9.0, 'MISSINGNO_34': 9.0,
    'MISSINGNO_38': 9.0, 'MISSINGNO_3D': 9.0, 'MISSINGNO_3E': 9.0, 'MISSINGNO_3F': 9.0,
    'MISSINGNO_43': 9.0, 'MISSINGNO_44': 9.0, 'MISSINGNO_45': 9.0, 'MISSINGNO_4F': 9.0,
    'MISSINGNO_50': 9.0, 'MISSINGNO_51': 9.0, 'MISSINGNO_56': 9.0, 'MISSINGNO_57': 9.0,
    'MISSINGNO_5E': 9.0, 'MISSINGNO_5F': 9.0, 'MISSINGNO_73': 9.0, 'MISSINGNO_79': 9.0,
    'MISSINGNO_7A': 9.0, 'MISSINGNO_7F': 9.0, 'MISSINGNO_86': 9.0, 'MISSINGNO_87': 9.0,
    'MISSINGNO_89': 9.0, 'MISSINGNO_8C': 9.0, 'MISSINGNO_92': 9.0, 'MISSINGNO_9C': 9.0,
    'MISSINGNO_9F': 9.0, 'MISSINGNO_A0': 9.0, 'MISSINGNO_A1': 9.0, 'MISSINGNO_A2': 9.0,
    'MISSINGNO_AC': 9.0, 'MISSINGNO_AE': 9.0, 'MISSINGNO_AF': 9.0, 'MISSINGNO_B5': 9.0,
    'FOSSIL_KABUTOPS': 9.0, 'FOSSIL_AERODACTYL': 9.0,
    'MON_GHOST': 9.0,   # Ghost form cannot be caught normally

    # --- Difficulty 9.5: Legendary Birds (requires puzzle-solving and special access)
    'ARTICUNO': 9.5,    # Deep in Seafoam Islands; requires puzzles and Surf
    'ZAPDOS': 9.5,      # Found in the Power Plant; extra conditions apply
    'MOLTRES': 9.5,     # Located in a late-game maze (Victory Road); very limited

    # --- Difficulty 10.0: Ultimate challenge of a static late-game encounter
    'MEWTWO': 10.0,     # Only available after beating the game; very high effort required

    # --- Difficulty 10.5: Mythical Pokémon (not obtainable by normal means)
    'MEW': 10.5,        # Only available via special events or glitches; highest acquisition difficulty
}



# --- Badge Acquisition Difficulty Ratings (7.0 - 20.0) ---
# Score reflects the overall challenge of obtaining the badge, significantly higher than most Pokémon captures
badge_difficulty_ratings = {
    'BOULDER': 7.0,     # Boulder Badge: First gym, relatively low difficulty, but still requires defeating the leader
    'CASCADE': 8.5,     # Cascade Badge: Requires traversing Mt. Moon, Leader Misty's Starmie is strong
    'THUNDER': 10.0,    # Thunder Badge: Requires HM01 Cut (complete S.S. Anne), gym has a puzzle element
    'RAINBOW': 11.0,    # Rainbow Badge: Reach large city Celadon, Leader Erika uses status conditions
    'SOUL': 13.0,       # Soul Badge: Long journey to Fuchsia City, gym has invisible walls maze, Leader Koga uses tricky tactics
    'MARSH': 16.0,      # Marsh Badge: **Major difficulty spike**, requires clearing Silph Co. HQ (large maze + Rocket), complex gym warp puzzle, Leader Sabrina's Psychic-types are very strong
    'VOLCANO': 18.0,    # Volcano Badge: Requires HM03 Surf, explore Pokémon Mansion for Secret Key to enter gym, Leader Blaine has high levels
    'EARTH': 20.0,      # Earth Badge: Final gym, in Viridian City but challenged late-game, gym has a spinner tile maze, Leader Giovanni is powerful, the last hurdle before Victory Road
}


# --- Location Exploration Difficulty/Importance Ratings (1.0 - 7.0+) ---
# Score primarily reflects the timing of first meaningful exploration and the location's complexity/importance
# Locations containing final challenges (like Elite Four rooms) will have higher scores
location_scores_by_name = {
    # --- Difficulty 1.0: Starting Point / Very Early ---
    "PALLET_TOWN": 1.0,             # Pallet Town: Starting point
    "VIRIDIAN_CITY": 1.0,           # Viridian City: First city encountered
    "ROUTE_1": 1.0,                 # Route 1: Connects Pallet and Viridian
    "ROUTE_2": 1.0,                 # Route 2 (South): North of Viridian City
    "ROUTE_22": 1.0,                # Route 22 (Early): First rival battle location
    "PLAYERS_HOUSE_1F": 1.0,        # Player's House 1F
    "PLAYERS_HOUSE_2F": 1.0,        # Player's House 2F
    "RIVALS_HOUSE": 1.0,            # Rival's House
    "OAKS_LAB": 1.0,                # Oak's Lab
    "VIRIDIAN_POKECENTER": 1.0,     # Viridian City Pokémon Center
    "VIRIDIAN_MART": 1.0,           # Viridian City Poké Mart
    "VIRIDIAN_SCHOOL": 1.0,         # Viridian City School (not very useful)
    "VIRIDIAN_HOUSE": 1.0,          # Viridian City regular house
    "VIRIDIAN_FOREST_SOUTH_GATE": 1.0,# Viridian Forest South Gate
    "VIRIDIAN_FOREST": 1.0,         # Viridian Forest: First maze, but simple

    # --- Difficulty 1.5: Pewter City Area / First Gym Region ---
    "PEWTER_CITY": 1.5,             # Pewter City: First city with a Gym
    "ROUTE_3": 1.5,                 # Route 3: Towards Mt. Moon
    "DIGLETTS_CAVE_ROUTE2": 1.5,    # Diglett's Cave (Route 2 Entrance): Small section accessible early
    "VIRIDIAN_FOREST_NORTH_GATE": 1.5,# Viridian Forest North Gate
    "ROUTE_2_HOUSE": 1.5,           # Route 2 House (HM05 Flash)
    "ROUTE_2_GATE": 1.5,            # Route 2 Gate (Connects Viridian Forest North and Pewter City)
    "MUSEUM_1F": 1.5,               # Pewter Museum 1F
    "MUSEUM_2F": 1.5,               # Pewter Museum 2F (Can revive fossils, but needs later items)
    "PEWTER_GYM": 1.5,              # Pewter Gym (Need to defeat Brock)
    "PEWTER_HOUSE_1": 1.5,          # Pewter City House 1
    "PEWTER_MART": 1.5,             # Pewter City Poké Mart
    "PEWTER_HOUSE_2": 1.5,          # Pewter City House 2
    "PEWTER_POKECENTER": 1.5,       # Pewter City Pokémon Center
    "MT_MOON_POKECENTER": 1.5,      # Mt. Moon Pokémon Center (On Route 4)
    "ROUTE_4": 1.5,                 # Route 4 (East of Mt. Moon)
    "TRADE_CENTER": 1.5,            # Trade Center (All Pokémon Center 2F)
    "COLOSSEUM": 1.5,               # Colosseum (All Pokémon Center 2F)

    # --- Difficulty 2.0: Cerulean City Area / Mt. Moon ---
    "CERULEAN_CITY": 2.0,           # Cerulean City: Second Gym city
    "ROUTE_5": 2.0,                 # Route 5: South of Cerulean City
    "ROUTE_6": 2.0,                 # Route 6: North of Vermilion City
    "ROUTE_9": 2.0,                 # Route 9: Towards Rock Tunnel
    "ROUTE_10": 2.0,                # Route 10 (North): Before Rock Tunnel entrance
    "ROUTE_24": 2.0,                # Route 24 (Nugget Bridge)
    "ROUTE_25": 2.0,                # Route 25 (Towards Bill's House)
    "MT_MOON_1F": 2.0,              # Mt. Moon 1F
    "MT_MOON_B1F": 2.0,             # Mt. Moon B1F
    "MT_MOON_B2F": 2.0,             # Mt. Moon B2F (Fossil choice)
    "CERULEAN_TRASHED_HOUSE": 2.0,  # Cerulean City Trashed House (TM28 Dig)
    "CERULEAN_TRADE_HOUSE": 2.0,    # Cerulean City House (Trade for Jynx)
    "CERULEAN_POKECENTER": 2.0,     # Cerulean City Pokémon Center
    "CERULEAN_GYM": 2.0,            # Cerulean Gym (Need to defeat Misty)
    "BIKE_SHOP": 2.0,               # Cerulean City Bike Shop (Get Bike Voucher first)
    "CERULEAN_MART": 2.0,           # Cerulean City Poké Mart
    "ROUTE_5_GATE": 2.0,            # Route 5 Gate (To Underground Path)
    "UNDERGROUND_PATH_ROUTE5": 2.0, # Underground Path (Route 5 Entrance)
    "DAYCARE": 2.0,                 # Daycare Center (Route 5)
    "ROUTE_6_GATE": 2.0,            # Route 6 Gate (To Underground Path)
    "UNDERGROUND_PATH_ROUTE6": 2.0, # Underground Path (Route 6 Entrance)
    "ROCK_TUNNEL_POKECENTER": 2.0,  # Rock Tunnel Pokémon Center (On Route 10)
    "BILLS_HOUSE": 2.0,             # Bill's House (Route 25, get S.S. Ticket)
    "UNDERGROUND_PATH_NS": 2.0,     # Underground Path (Connects Route 5 and 6)
    "CERULEAN_BADGE_HOUSE": 2.0,    # Cerulean City House (Gives TM13 Ice Beam after badge check)

    # --- Difficulty 2.5: Vermilion City Area / S.S. Anne ---
    "VERMILION_CITY": 2.5,          # Vermilion City: Third Gym city, Port
    "ROUTE_7": 2.5,                 # Route 7: West of Celadon City
    "ROUTE_8": 2.5,                 # Route 8: West of Lavender Town
    "ROUTE_11": 2.5,                # Route 11: East of Vermilion (Snorlax blocks, need Diglett's Cave detour)
    "ROUTE_7_GATE": 2.5,            # Route 7 Gate (To Underground Path)
    "UNDERGROUND_PATH_ROUTE7": 2.5, # Underground Path (Route 7 Entrance)
    "ROUTE_8_GATE": 2.5,            # Route 8 Gate (To Underground Path)
    "UNDERGROUND_PATH_ROUTE8": 2.5, # Underground Path (Route 8 Entrance)
    "DIGLETTS_CAVE_ROUTE11": 2.5,   # Diglett's Cave (Route 11 Entrance)
    "ROUTE_11_GATE_1F": 2.5,        # Route 11 Gate 1F
    "ROUTE_11_GATE_2F": 2.5,        # Route 11 Gate 2F (Oak's Aide gives Itemfinder)
    "VERMILION_POKECENTER": 2.5,    # Vermilion City Pokémon Center
    "FAN_CLUB": 2.5,                # Pokémon Fan Club (Get Bike Voucher)
    "VERMILION_MART": 2.5,          # Vermilion City Poké Mart
    "VERMILION_GYM": 2.5,           # Vermilion Gym (Need to solve trash can switch puzzle)
    "VERMILION_HOUSE_1": 2.5,       # Vermilion City House 1 (Good Rod)
    "VERMILION_DOCK": 2.5,          # Vermilion City Dock (S.S. Anne Entrance)
    "SS_ANNE_1F": 2.5,              # S.S. Anne 1F
    "SS_ANNE_2F": 2.5,              # S.S. Anne 2F
    "SS_ANNE_3F": 2.5,              # S.S. Anne 3F (Bow)
    "SS_ANNE_B1F": 2.5,             # S.S. Anne B1F
    "SS_ANNE_BOW": 2.5,             # S.S. Anne Bow
    "SS_ANNE_KITCHEN": 2.5,         # S.S. Anne Kitchen
    "SS_ANNE_CAPTAINS_ROOM": 2.5,   # S.S. Anne Captain's Room (HM01 Cut)
    "SS_ANNE_1F_ROOMS": 2.5,        # S.S. Anne 1F Rooms
    "SS_ANNE_2F_ROOMS": 2.5,        # S.S. Anne 2F Rooms
    "SS_ANNE_B1F_ROOMS": 2.5,       # S.S. Anne B1F Rooms
    "VERMILION_HOUSE_2": 2.5,       # Vermilion City House 2
    "VERMILION_HOUSE_3": 2.5,       # Vermilion City House 3
    "VERMILION_HOUSE_4": 2.5,       # Vermilion City House 4
    "DIGLETTS_CAVE": 2.5,           # Diglett's Cave main body (Connects Route 11 and 2)
    "UNDERGROUND_PATH_WE": 2.5,     # Underground Path (Connects Route 7 and 8)

    # --- Difficulty 3.0: Lavender Town / Celadon City / Rock Tunnel ---
    "LAVENDER_TOWN": 3.0,           # Lavender Town: Location of Pokémon Tower
    "CELADON_CITY": 3.0,            # Celadon City: Large city, 4th Gym, Game Corner, Dept. Store
    "SAFFRON_CITY": 3.0,            # Saffron City: Can pass through early, but core areas blocked by Rocket
    "ROCK_TUNNEL_1F": 3.0,          # Rock Tunnel 1F (Requires HM05 Flash)
    "ROCK_TUNNEL_B1F": 3.0,         # Rock Tunnel B1F
    "CELADON_MART_1F": 3.0,         # Celadon Dept. Store 1F
    "CELADON_MART_2F": 3.0,         # Celadon Dept. Store 2F
    "CELADON_MART_3F": 3.0,         # Celadon Dept. Store 3F
    "CELADON_MART_4F": 3.0,         # Celadon Dept. Store 4F (Evolution Stones)
    "CELADON_MART_ROOF": 3.0,       # Celadon Dept. Store Roof (Vending Machines for TMs)
    "CELADON_MART_ELEVATOR": 3.0,   # Celadon Dept. Store Elevator
    "CELADON_MANSION_1F": 3.0,      # Celadon Mansion 1F
    "CELADON_MANSION_2F": 3.0,      # Celadon Mansion 2F
    "CELADON_MANSION_3F": 3.0,      # Celadon Mansion 3F
    "CELADON_MANSION_ROOF": 3.0,    # Celadon Mansion Roof
    "CELADON_MANSION_ROOF_HOUSE": 3.0,# Celadon Mansion Roof House (Eevee)
    "CELADON_POKECENTER": 3.0,      # Celadon City Pokémon Center
    "CELADON_GYM": 3.0,             # Celadon Gym (Requires HM01 Cut to enter)
    "GAME_CORNER": 3.0,             # Game Corner (Slots, Exchange Pokémon/TMs)
    "CELADON_MART_5F": 3.0,         # Celadon Dept. Store 5F (Stat enhancers)
    "GAME_CORNER_PRIZE_ROOM": 3.0,  # Game Corner Prize Exchange
    "CELADON_DINER": 3.0,           # Celadon City Diner (Get info)
    "CELADON_HOUSE": 3.0,           # Celadon City House
    "CELADON_HOTEL": 3.0,           # Celadon City Hotel (Game Freak Dev Room)
    "LAVENDER_POKECENTER": 3.0,     # Lavender Town Pokémon Center
    "POKEMON_TOWER_1F": 3.0,        # Pokémon Tower 1F
    "POKEMON_TOWER_2F": 3.0,        # Pokémon Tower 2F (Rival battle)
    "LAVENDER_HOUSE_1": 3.0,        # Lavender Town House 1 (Name Rater)
    "LAVENDER_MART": 3.0,           # Lavender Town Poké Mart
    "LAVENDER_HOUSE_2": 3.0,        # Lavender Town House 2 (Mr. Fuji's House)
    "COPYCATS_HOUSE_1F": 3.0,       # Copycat's House 1F (Saffron City)
    "COPYCATS_HOUSE_2F": 3.0,       # Copycat's House 2F (Get TM31 Mimic)
    "FIGHTING_DOJO": 3.0,           # Fighting Dojo (Saffron City, Choose Hitmonlee/Hitmonchan after winning)
    "SAFFRON_HOUSE_1": 3.0,         # Saffron City House 1 (Psychic gives TM29 Psychic)
    "SAFFRON_MART": 3.0,            # Saffron City Poké Mart
    "SAFFRON_POKECENTER": 3.0,      # Saffron City Pokémon Center
    "SAFFRON_HOUSE_2": 3.0,         # Saffron City House 2
    "NAME_RATERS_HOUSE": 3.0,       # Name Rater's House (Lavender Town)

    # --- Difficulty 3.5: Fuchsia City Area / Cycling Road / Requires Silph Scope ---
    "FUCHSIA_CITY": 3.5,            # Fuchsia City: Fifth Gym city, Safari Zone location
    "ROUTE_12": 3.5,                # Route 12 (Silence Bridge, has Snorlax)
    "ROUTE_13": 3.5,                # Route 13 (Maze-like grass)
    "ROUTE_14": 3.5,                # Route 14
    "ROUTE_15": 3.5,                # Route 15
    "ROUTE_16": 3.5,                # Route 16 (Cycling Road entrance, has Snorlax)
    "ROUTE_17": 3.5,                # Route 17 (Cycling Road)
    "ROUTE_18": 3.5,                # Route 18 (Cycling Road exit)
    "POKEMON_TOWER_3F": 3.5,        # Pokémon Tower 3F (Need Silph Scope to see ghosts)
    "POKEMON_TOWER_4F": 3.5,        # Pokémon Tower 4F
    "POKEMON_TOWER_5F": 3.5,        # Pokémon Tower 5F
    "POKEMON_TOWER_6F": 3.5,        # Pokémon Tower 6F
    "POKEMON_TOWER_7F": 3.5,        # Pokémon Tower 7F (Rescue Mr. Fuji, get Poké Flute)
    "FUCHSIA_MART": 3.5,            # Fuchsia City Poké Mart
    "FUCHSIA_HOUSE_1": 3.5,         # Fuchsia City Warden's House (Need to find Gold Teeth)
    "FUCHSIA_POKECENTER": 3.5,      # Fuchsia City Pokémon Center
    "FUCHSIA_HOUSE_2": 3.5,         # Fuchsia City House 2 (Move Deleter)
    "SAFARI_ZONE_ENTRANCE": 3.5,    # Safari Zone Entrance
    "FUCHSIA_GYM": 3.5,             # Fuchsia Gym (Invisible walls maze)
    "SILPH_CO_1F": 3.5,             # Silph Co. 1F (Accessible early, but blocked by Rocket)
    "ROUTE_15_GATE_1F": 3.5,        # Route 15 Gate 1F
    "ROUTE_15_GATE_2F": 3.5,        # Route 15 Gate 2F (Oak's Aide gives Exp. All)
    "ROUTE_16_GATE_1F": 3.5,        # Route 16 Gate 1F
    "ROUTE_16_GATE_2F": 3.5,        # Route 16 Gate 2F
    "ROUTE_16_HOUSE": 3.5,          # Route 16 House (HM02 Fly)
    "ROUTE_12_HOUSE": 3.5,          # Route 12 House (Super Rod)
    "ROUTE_18_GATE_1F": 3.5,        # Route 18 Gate 1F
    "ROUTE_18_GATE_2F": 3.5,        # Route 18 Gate 2F
    "ROUTE_12_GATE_1F": 3.5,        # Route 12 Gate 1F
    "ROUTE_12_GATE_2F": 3.5,        # Route 12 Gate 2F
    "ROCKET_HIDEOUT_B1F": 3.5,      # Rocket Hideout B1F (Below Game Corner, need Lift Key)
    "ROCKET_HIDEOUT_B2F": 3.5,      # Rocket Hideout B2F
    "ROCKET_HIDEOUT_B3F": 3.5,      # Rocket Hideout B3F
    "ROCKET_HIDEOUT_B4F": 3.5,      # Rocket Hideout B4F (Defeat Giovanni, get Silph Scope)
    "ROCKET_HIDEOUT_ELEVATOR": 3.5, # Rocket Hideout Elevator

    # --- Difficulty 4.0: Cinnabar Island Area / Safari Zone Interior / Requires Surf / Silph Co. ---
    "CINNABAR_ISLAND": 4.0,         # Cinnabar Island: Seventh Gym, Fossil Lab
    "ROUTE_19": 4.0,                # Route 19 (Water route, towards Seafoam Islands)
    "ROUTE_20": 4.0,                # Route 20 (Water route, around Seafoam Islands)
    "ROUTE_21": 4.0,                # Route 21 (Water route, North of Cinnabar)
    "FUCHSIA_MEETING_ROOM": 4.0,    # Fuchsia City Warden's Office (Get HM04 Strength after returning Gold Teeth)
    "SEAFOAM_ISLANDS_1F": 4.0,      # Seafoam Islands 1F (Requires Surf + Strength)
    "CINNABAR_LAB_1": 4.0,          # Cinnabar Lab - Lab 1 (Fossil Revival)
    "CINNABAR_LAB_2": 4.0,          # Cinnabar Lab - Lab 2
    "CINNABAR_LAB_3": 4.0,          # Cinnabar Lab - Lab 3 (NPC Trade)
    "CINNABAR_LAB_4": 4.0,          # Cinnabar Lab - Lab 4
    "CINNABAR_POKECENTER": 4.0,     # Cinnabar Island Pokémon Center
    "CINNABAR_MART": 4.0,           # Cinnabar Island Poké Mart
    "SAFFRON_GYM": 4.0,             # Saffron Gym (Need to liberate Silph Co. first)
    "SILPH_CO_2F": 4.0,             # Silph Co. 2F (Large maze begins, need Card Key)
    "SILPH_CO_3F": 4.0,             # Silph Co. 3F
    "SILPH_CO_4F": 4.0,             # Silph Co. 4F
    "SILPH_CO_5F": 4.0,             # Silph Co. 5F (Card Key location)
    "SILPH_CO_6F": 4.0,             # Silph Co. 6F
    "SILPH_CO_7F": 4.0,             # Silph Co. 7F (Rival battle, Lapras)
    "SILPH_CO_8F": 4.0,             # Silph Co. 8F
    "SAFARI_ZONE_EAST": 4.0,        # Safari Zone East Area
    "SAFARI_ZONE_NORTH": 4.0,       # Safari Zone North Area
    "SAFARI_ZONE_WEST": 4.0,        # Safari Zone West Area
    "SAFARI_ZONE_CENTER": 4.0,      # Safari Zone Center Area
    "SAFARI_ZONE_CENTER_REST_HOUSE": 4.0, # Safari Zone Center Rest House
    "SAFARI_ZONE_SECRET_HOUSE": 4.0,# Safari Zone Secret House (HM03 Surf, need Gold Teeth)
    "SAFARI_ZONE_WEST_REST_HOUSE": 4.0, # Safari Zone West Rest House
    "SAFARI_ZONE_EAST_REST_HOUSE": 4.0, # Safari Zone East Rest House
    "SAFARI_ZONE_NORTH_REST_HOUSE": 4.0, # Safari Zone North Rest House
    "SILPH_CO_9F": 4.0,             # Silph Co. 9F (Bed restores health)
    "SILPH_CO_10F": 4.0,            # Silph Co. 10F
    "SILPH_CO_11F": 4.0,            # Silph Co. 11F (Defeat Giovanni, rescue President, get Master Ball)
    "SILPH_CO_ELEVATOR": 4.0,       # Silph Co. Elevator

    # --- Difficulty 4.5: Late-game large mazes / Requires special key / Legendary bird locations ---
    "VIRIDIAN_GYM": 4.5,            # Viridian Gym (Need 7 badges to challenge Giovanni)
    "POWER_PLANT": 4.5,             # Power Plant (Requires Surf, contains Zapdos)
    "SEAFOAM_ISLANDS_B1F": 4.5,     # Seafoam Islands B1F
    "SEAFOAM_ISLANDS_B2F": 4.5,     # Seafoam Islands B2F
    "SEAFOAM_ISLANDS_B3F": 4.5,     # Seafoam Islands B3F
    "SEAFOAM_ISLANDS_B4F": 4.5,     # Seafoam Islands B4F (Articuno location)
    "POKEMON_MANSION_1F": 4.5,      # Pokémon Mansion 1F (Cinnabar Island, large maze)
    "CINNABAR_GYM": 4.5,            # Cinnabar Gym (Requires Secret Key from Pokémon Mansion)
    "POKEMON_MANSION_2F": 4.5,      # Pokémon Mansion 2F
    "POKEMON_MANSION_3F": 4.5,      # Pokémon Mansion 3F
    "POKEMON_MANSION_B1F": 4.5,     # Pokémon Mansion B1F (Secret Key location)

    # --- Difficulty 5.0: Victory Road Entrance ---
    "ROUTE_23": 5.0,                # Route 23 (Before Victory Road, requires checking all 8 badges)
    "ROUTE_22_GATE": 5.0,           # Route 22 Badge Check Gate (To Route 23)

    # --- Difficulty 5.5: Victory Road / Indigo Plateau Exterior ---
    "INDIGO_PLATEAU": 5.5,          # Indigo Plateau Exterior (Pokémon Center and Mart)
    "VICTORY_ROAD_1F": 5.5,         # Victory Road 1F (Requires Strength)
    "VICTORY_ROAD_2F": 5.5,         # Victory Road 2F (Moltres location in Red/Blue)
    "VICTORY_ROAD_3F": 5.5,         # Victory Road 3F
    "INDIGO_PLATEAU_LOBBY": 5.5,    # Indigo Plateau League Lobby

    # --- Difficulty 6.0: Early Elite Four Challenge ---
    "LORELEI": 6.0,                 # Elite Four Lorelei's Room
    "BRUNO": 6.0,                   # Elite Four Bruno's Room

    # --- Difficulty 6.5: Late Elite Four Challenge / End Game Marker ---
    "LANCE": 6.5,                   # Elite Four Lance's Room
    "HALL_OF_FAME": 6.5,            # Hall of Fame (Records team)
    "CHAMPIONS_ROOM": 6.5,          # Champion's Room (Challenge Rival)
    "AGATHA": 6.5,                  # Elite Four Agatha's Room

    # --- Difficulty 7.0: Post-Game Hidden Area ---
    "CERULEAN_CAVE_2F": 7.0,        # Cerulean Cave 2F (Opens post-game, high-level Pokémon)
    "CERULEAN_CAVE_B1F": 7.0,       # Cerulean Cave B1F (Mewtwo location)
    "CERULEAN_CAVE_1F": 7.0,        # Cerulean Cave 1F
}

# Combine all ratings into one large dictionary for easy lookup
all_difficulty_ratings = {}
all_difficulty_ratings.update(pokemon_difficulty_ratings_refined)
all_difficulty_ratings.update(badge_difficulty_ratings)
all_difficulty_ratings.update(location_scores_by_name)