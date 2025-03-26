from gym.envs.registration import register


all_event_types = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
all_event_attributes = {
    'fg': ['red', 'blue', 'green', 'orange', 'pink'],
    'bg': ['red', 'blue', 'green', 'orange', 'pink']
}

# Define noise event types and attributes
all_noise_types = ['K', 'L', 'M', 'N', 'O']
all_noise_attributes = {
    'fg': ['yellow', 'purple', 'black'],
    'bg': ['yellow', 'purple', 'black']
}


all_types = all_event_types + all_noise_types

all_attributes = {
  "bg": all_event_attributes["bg"] + all_noise_attributes["bg"],
  "fg": all_event_attributes["fg"] + all_noise_attributes["fg"]
}

instructions_easy = [
                            [
                              {'command': 'delay', 'parameters': 2},
                              {'command': 'noise', 'parameters': 0.1},
                              {'command': 'instantiate', 'parameters': ('A', {'bg': 'blue', 'fg': 'green'}, {'mu': 5, 'sigma': 2}), 'variable_name': 'distinct'},
                            ],
                            [
                              {'command': 'delay', 'parameters': 4},
                              {'command': 'noise', 'parameters': 0.2},
                              {'command': 'instantiate', 'parameters': ('B', {'bg': 'red', 'fg': 'pink'}, {'mu': 10, 'sigma': 3}), 'variable_name': 'distinct'},
                            ],
                            [
                              {'command': 'delay', 'parameters': 6},
                              {'command': 'noise', 'parameters': 0.2},
                              {'command': 'instantiate', 'parameters': ('C', {'bg': 'orange', 'fg': 'red'}, {'mu': 4, 'sigma': 1}), 'variable_name': 'distinct'},
                            ]
                      ]

instructions_medium = [
                            [
                                {'command': 'delay', 'parameters': 1},
                                {'command': 'noise', 'parameters': 0.1},
                                {'command': 'instantiate', 'parameters': ('A', {'bg': 'blue', 'fg': 'blue'}, {'mu': 5, 'sigma': 2}), 'variable_name': 'e1'},
                                {'command': 'instantiate', 'parameters': ('C', {'bg': 'blue', 'fg': 'blue'}, {'mu': 9, 'sigma': 2}), 'variable_name': 'e2'},
                                {'command': 'instantiate', 'parameters': ('C', {'bg': 'blue', 'fg': 'red'}, {'mu': 3, 'sigma': 2}), 'variable_name': 'e3'},
                                {'command': 'instantiate', 'parameters': ('A', {'bg': 'green', 'fg': 'blue'}, {'mu': 9, 'sigma': 2}), 'variable_name': 'e4'},
                                {'command': 'after', 'parameters': ['e2', 'e1'], 'variable_name': 'e2', 'other': {'gap_dist': {'mu': 2, 'sigma': 1}}},
                                {'command': 'during', 'parameters': ['e3', 'e2'], 'variable_name': 'e3', 'other': {}},
                                {'command': 'met_by', 'parameters': ['e4', 'e3'], 'variable_name': 'e4', 'other': {}}
                            ],
                            [
                                {'command': 'delay', 'parameters': 5},
                                {'command': 'noise', 'parameters': 0.1},
                                {'command': 'instantiate', 'parameters': ('B', {'bg': 'green', 'fg': 'green'}, {'mu': 10, 'sigma': 2}), 'variable_name': 'e1'},
                                {'command': 'instantiate', 'parameters': ('B', {'bg': 'orange', 'fg': 'blue'}, {'mu': 5, 'sigma': 2}), 'variable_name': 'e2'},
                                {'command': 'instantiate', 'parameters': ('D', {'bg': 'orange', 'fg': 'blue'}, {'mu': 4, 'sigma': 2}), 'variable_name': 'e3'},
                                {'command': 'instantiate', 'parameters': ('A', {'bg': 'green', 'fg': 'green'}, {'mu': 15, 'sigma': 2}), 'variable_name': 'e4'},
                                {'command': 'during', 'parameters': ['e2', 'e1'], 'variable_name': 'e2', 'other': {}},
                                {'command': 'met_by', 'parameters': ['e3', 'e2'], 'variable_name': 'e3', 'other': {}},
                                {'command': 'after', 'parameters': ['e4', 'e3'], 'variable_name': 'e4', 'other': {'gap_dist': {'mu': 2, 'sigma': 1}}}
                            ],
                            [
                                {'command': 'delay', 'parameters': 7},
                                {'command': 'noise', 'parameters': 0.1},
                                {'command': 'instantiate', 'parameters': ('C', {'bg': 'orange', 'fg': 'pink'}, {'mu': 5, 'sigma': 2}), 'variable_name': 'e1'},
                                {'command': 'instantiate', 'parameters': ('H', {'bg': 'blue', 'fg': 'blue'}, {'mu': 5, 'sigma': 2}), 'variable_name': 'e2'},
                                {'command': 'instantiate', 'parameters': ('F', {'bg': 'blue', 'fg': 'blue'}, {'mu': 5, 'sigma': 2}), 'variable_name': 'e3'},
                                {'command': 'instantiate', 'parameters': ('A', {'bg': 'orange', 'fg': 'pink'}, {'mu': 5, 'sigma': 2}), 'variable_name': 'e4'},
                                {'command': 'after', 'parameters': ['e2', 'e1'], 'variable_name': 'e2', 'other': {'gap_dist': {'mu': 2, 'sigma': 1}}},
                                {'command': 'after', 'parameters': ['e3', 'e2'], 'variable_name': 'e3', 'other': {'gap_dist': {'mu': 2, 'sigma': 1}}},
                                {'command': 'after', 'parameters': ['e4', 'e3'], 'variable_name': 'e4', 'other': {'gap_dist': {'mu': 2, 'sigma': 1}}}
                            ]

                      ]


instructions_hard = [
                          [
                              {'command': 'delay', 'parameters': 5},
                              {'command': 'noise', 'parameters': 0.1},
                              {'command': 'instantiate', 'parameters': ('E', {'bg': 'green', 'fg': 'orange'}, {'mu': 6, 'sigma': 2}), 'variable_name': 'e1'},
                              {'command': 'instantiate', 'parameters': ('E', {'bg': 'green', 'fg': 'orange'}, {'mu': 3, 'sigma': 1}), 'variable_name': 'e2'},
                              {'command': 'instantiate', 'parameters': ('A', {'bg': 'red', 'fg': 'green'}, {'mu': 7, 'sigma': 3}), 'variable_name': 'e3'},
                              {'command': 'instantiate', 'parameters': ('G', {'bg': 'orange', 'fg': 'blue'}, {'mu': 8, 'sigma': 2}), 'variable_name': 'e4'},
                              {'command': 'instantiate', 'parameters': ('B', {'bg': 'pink', 'fg': 'red'}, {'mu': 6, 'sigma': 1}), 'variable_name': 'e5'},
                              {'command': 'instantiate', 'parameters': ('H', {'bg': 'green', 'fg': 'blue'}, {'mu': 20, 'sigma': 2}), 'variable_name': 'e6'},
                              {'command': 'instantiate', 'parameters': ('F', {'bg': 'orange', 'fg': 'pink'}, {'mu': 7, 'sigma': 1}), 'variable_name': 'e7'},
                              {'command': 'instantiate', 'parameters': ('I', {'bg': 'blue', 'fg': 'red'}, {'mu': 5, 'sigma': 2}), 'variable_name': 'e8'},
                              {'command': 'instantiate', 'parameters': ('D', {'bg': 'pink', 'fg': 'green'}, {'mu': 20, 'sigma': 3}), 'variable_name': 'e9'},
                              {'command': 'instantiate', 'parameters': ('J', {'bg': 'red', 'fg': 'orange'}, {'mu': 7, 'sigma': 1}), 'variable_name': 'e10'},
                              {'command': 'instantiate', 'parameters': ('E', {'bg': 'blue', 'fg': 'pink'}, {'mu': 6, 'sigma': 2}), 'variable_name': 'e11'},
                              {'command': 'instantiate', 'parameters': ('C', {'bg': 'green', 'fg': 'red'}, {'mu': 9, 'sigma': 1}), 'variable_name': 'e12'},
                              {'command': 'instantiate', 'parameters': ('A', {'bg': 'orange', 'fg': 'blue'}, {'mu': 5, 'sigma': 3}), 'variable_name': 'e13'},
                              {'command': 'instantiate', 'parameters': ('G', {'bg': 'pink', 'fg': 'green'}, {'mu': 7, 'sigma': 2}), 'variable_name': 'e14'},
                              {'command': 'instantiate', 'parameters': ('B', {'bg': 'red', 'fg': 'blue'}, {'mu': 24, 'sigma': 5}), 'variable_name': 'e15'},
                              {'command': 'instantiate', 'parameters': ('H', {'bg': 'blue', 'fg': 'orange'}, {'mu': 8, 'sigma': 2}), 'variable_name': 'e16'},
                              {'command': 'after', 'parameters': ['e1', 'e2'], 'variable_name': 'e1', 'other': {'gap_dist': {'mu': 5, 'sigma': 1}}},
                              {'command': 'during', 'parameters': ['e2', 'e3'], 'variable_name': 'e2', 'other': {}},
                              {'command': 'met_by', 'parameters': ['e3', 'e4'], 'variable_name': 'e3', 'other': {}},
                              {'command': 'after', 'parameters': ['e4', 'e5'], 'variable_name': 'e4', 'other': {'gap_dist': {'mu': 4, 'sigma': 2}}},
                              {'command': 'during', 'parameters': ['e5', 'e6'], 'variable_name': 'e5', 'other': {}},
                              {'command': 'met_by', 'parameters': ['e6', 'e7'], 'variable_name': 'e6', 'other': {}},
                              {'command': 'after', 'parameters': ['e7', 'e8'], 'variable_name': 'e7', 'other': {'gap_dist': {'mu': 3, 'sigma': 1}}},
                              {'command': 'during', 'parameters': ['e8', 'e9'], 'variable_name': 'e8', 'other': {}},
                              {'command': 'met_by', 'parameters': ['e9', 'e10'], 'variable_name': 'e9', 'other': {}},
                              {'command': 'after', 'parameters': ['e10', 'e11'], 'variable_name': 'e10', 'other': {'gap_dist': {'mu': 6, 'sigma': 2}}},
                              {'command': 'during', 'parameters': ['e11', 'e12'], 'variable_name': 'e11', 'other': {}},
                              {'command': 'met_by', 'parameters': ['e12', 'e13'], 'variable_name': 'e12', 'other': {}},
                              {'command': 'after', 'parameters': ['e13', 'e14'], 'variable_name': 'e13', 'other': {'gap_dist': {'mu': 4, 'sigma': 1}}},
                              {'command': 'during', 'parameters': ['e14', 'e15'], 'variable_name': 'e14', 'other': {}},
                              {'command': 'met_by', 'parameters': ['e15', 'e16'], 'variable_name': 'e15', 'other': {}}
                          ]
,
                            [
                              {'command': 'delay', 'parameters': 0},
                              {'command': 'noise', 'parameters': 0.1},
                              {'command': 'instantiate', 'parameters': ('I', {'bg': 'red', 'fg': 'pink'}, {'mu': 8, 'sigma': 1}), 'variable_name': 'e1'},
                              {'command': 'instantiate', 'parameters': ('B', {'bg': 'orange', 'fg': 'red'}, {'mu': 4, 'sigma': 2}), 'variable_name': 'e2'},
                              {'command': 'instantiate', 'parameters': ('C', {'bg': 'pink', 'fg': 'green'}, {'mu': 10, 'sigma': 2}), 'variable_name': 'e3'},
                              {'command': 'instantiate', 'parameters': ('C', {'bg': 'pink', 'fg': 'green'}, {'mu': 9, 'sigma': 3}), 'variable_name': 'e4'},
                              {'command': 'instantiate', 'parameters': ('D', {'bg': 'blue', 'fg': 'orange'}, {'mu': 5, 'sigma': 2}), 'variable_name': 'e5'},
                              {'command': 'instantiate', 'parameters': ('G', {'bg': 'red', 'fg': 'blue'}, {'mu': 15, 'sigma': 1}), 'variable_name': 'e7'},
                              {'command': 'instantiate', 'parameters': ('J', {'bg': 'orange', 'fg': 'pink'}, {'mu': 8, 'sigma': 2}), 'variable_name': 'e6'},
                              {'command': 'instantiate', 'parameters': ('E', {'bg': 'green', 'fg': 'red'}, {'mu': 6, 'sigma': 2}), 'variable_name': 'e8'},
                              {'command': 'instantiate', 'parameters': ('H', {'bg': 'pink', 'fg': 'blue'}, {'mu': 30, 'sigma': 1}), 'variable_name': 'e10'},
                              {'command': 'instantiate', 'parameters': ('A', {'bg': 'blue', 'fg': 'red'}, {'mu': 9, 'sigma': 3}), 'variable_name': 'e9'},
                              {'command': 'instantiate', 'parameters': ('I', {'bg': 'green', 'fg': 'orange'}, {'mu': 8, 'sigma': 1}), 'variable_name': 'e11'},
                              {'command': 'instantiate', 'parameters': ('F', {'bg': 'orange', 'fg': 'blue'}, {'mu': 15, 'sigma': 2}), 'variable_name': 'e13'},
                              {'command': 'instantiate', 'parameters': ('B', {'bg': 'red', 'fg': 'green'}, {'mu': 7, 'sigma': 3}), 'variable_name': 'e12'},
                              {'command': 'instantiate', 'parameters': ('C', {'bg': 'pink', 'fg': 'orange'}, {'mu': 5, 'sigma': 2}), 'variable_name': 'e14'},
                              {'command': 'instantiate', 'parameters': ('D', {'bg': 'blue', 'fg': 'green'}, {'mu': 18, 'sigma': 1}), 'variable_name': 'e16'},
                              {'command': 'instantiate', 'parameters': ('G', {'bg': 'orange', 'fg': 'pink'}, {'mu': 7, 'sigma': 2}), 'variable_name': 'e15'},
                              {'command': 'met_by', 'parameters': ['e1', 'e2'], 'variable_name': 'e1', 'other': {}},
                              {'command': 'during', 'parameters': ['e2', 'e3'], 'variable_name': 'e2', 'other': {}},
                              {'command': 'after', 'parameters': ['e3', 'e4'], 'variable_name': 'e3', 'other': {'gap_dist': {'mu': 4, 'sigma': 1}}},
                              {'command': 'met_by', 'parameters': ['e4', 'e5'], 'variable_name': 'e4', 'other': {}},
                              {'command': 'after', 'parameters': ['e5', 'e6'], 'variable_name': 'e5', 'other': {'gap_dist': {'mu': 5, 'sigma': 2}}},
                              {'command': 'during', 'parameters': ['e6', 'e7'], 'variable_name': 'e6', 'other': {}},
                              {'command': 'met_by', 'parameters': ['e7', 'e8'], 'variable_name': 'e7', 'other': {}},
                              {'command': 'after', 'parameters': ['e8', 'e9'], 'variable_name': 'e8', 'other': {'gap_dist': {'mu': 3, 'sigma': 1}}},
                              {'command': 'during', 'parameters': ['e9', 'e10'], 'variable_name': 'e9', 'other': {}},
                              {'command': 'met_by', 'parameters': ['e10', 'e11'], 'variable_name': 'e10', 'other': {}},
                              {'command': 'after', 'parameters': ['e11', 'e12'], 'variable_name': 'e11', 'other': {'gap_dist': {'mu': 6, 'sigma': 2}}},
                              {'command': 'during', 'parameters': ['e12', 'e13'], 'variable_name': 'e12', 'other': {}},
                              {'command': 'met_by', 'parameters': ['e13', 'e14'], 'variable_name': 'e13', 'other': {}},
                              {'command': 'after', 'parameters': ['e14', 'e15'], 'variable_name': 'e14', 'other': {'gap_dist': {'mu': 4, 'sigma': 1}}},
                              {'command': 'during', 'parameters': ['e15', 'e16'], 'variable_name': 'e15', 'other': {}}
                          ],
                            [
                              {'command': 'delay', 'parameters': 20},
                              {'command': 'noise', 'parameters': 0},
                              {'command': 'instantiate', 'parameters': ('J', {'bg': 'orange', 'fg': 'green'}, {'mu': 9, 'sigma': 2}), 'variable_name': 'e1'},
                              {'command': 'instantiate', 'parameters': ('G', {'bg': 'pink', 'fg': 'blue'}, {'mu': 8, 'sigma': 1}), 'variable_name': 'e2'},
                              {'command': 'instantiate', 'parameters': ('A', {'bg': 'red', 'fg': 'orange'}, {'mu': 9, 'sigma': 3}), 'variable_name': 'e4'},
                              {'command': 'instantiate', 'parameters': ('H', {'bg': 'blue', 'fg': 'pink'}, {'mu': 4, 'sigma': 2}), 'variable_name': 'e3'},
                              {'command': 'instantiate', 'parameters': ('C', {'bg': 'green', 'fg': 'red'}, {'mu': 9, 'sigma': 1}), 'variable_name': 'e5'},
                              {'command': 'instantiate', 'parameters': ('E', {'bg': 'orange', 'fg': 'blue'}, {'mu': 8, 'sigma': 2}), 'variable_name': 'e6'},
                              {'command': 'instantiate', 'parameters': ('F', {'bg': 'pink', 'fg': 'green'}, {'mu': 6, 'sigma': 3}), 'variable_name': 'e7'},
                              {'command': 'instantiate', 'parameters': ('B', {'bg': 'blue', 'fg': 'orange'}, {'mu': 15, 'sigma': 1}), 'variable_name': 'e9'},
                              {'command': 'instantiate', 'parameters': ('D', {'bg': 'red', 'fg': 'blue'}, {'mu': 8, 'sigma': 2}), 'variable_name': 'e8'},
                              {'command': 'instantiate', 'parameters': ('I', {'bg': 'green', 'fg': 'pink'}, {'mu': 9, 'sigma': 3}), 'variable_name': 'e10'},
                              {'command': 'instantiate', 'parameters': ('J', {'bg': 'orange', 'fg': 'green'}, {'mu': 14, 'sigma': 1}), 'variable_name': 'e12'},
                              {'command': 'instantiate', 'parameters': ('G', {'bg': 'pink', 'fg': 'blue'}, {'mu': 8, 'sigma': 2}), 'variable_name': 'e11'},
                              {'command': 'instantiate', 'parameters': ('A', {'bg': 'red', 'fg': 'orange'}, {'mu': 7, 'sigma': 3}), 'variable_name': 'e13'},
                              {'command': 'instantiate', 'parameters': ('H', {'bg': 'blue', 'fg': 'pink'}, {'mu': 20, 'sigma': 1}), 'variable_name': 'e15'},
                              {'command': 'instantiate', 'parameters': ('C', {'bg': 'green', 'fg': 'red'}, {'mu': 9, 'sigma': 2}), 'variable_name': 'e14'},
                              {'command': 'instantiate', 'parameters': ('E', {'bg': 'orange', 'fg': 'blue'}, {'mu': 7, 'sigma': 1}), 'variable_name': 'e16'},
                              {'command': 'met_by', 'parameters': ['e1', 'e2'], 'variable_name': 'e1', 'other': {}},
                              {'command': 'during', 'parameters': ['e3', 'e4'], 'variable_name': 'e3', 'other': {}},
                              {'command': 'after', 'parameters': ['e4', 'e5'], 'variable_name': 'e4', 'other': {'gap_dist': {'mu': 5, 'sigma': 1}}},
                              {'command': 'met_by', 'parameters': ['e5', 'e6'], 'variable_name': 'e5', 'other': {}},
                              {'command': 'after', 'parameters': ['e6', 'e7'], 'variable_name': 'e6', 'other': {'gap_dist': {'mu': 4, 'sigma': 2}}},
                              {'command': 'during', 'parameters': ['e8', 'e9'], 'variable_name': 'e8', 'other': {}},
                              {'command': 'met_by', 'parameters': ['e9', 'e10'], 'variable_name': 'e9', 'other': {}},
                              {'command': 'after', 'parameters': ['e10', 'e11'], 'variable_name': 'e10', 'other': {'gap_dist': {'mu': 3, 'sigma': 1}}},
                              {'command': 'during', 'parameters': ['e11', 'e12'], 'variable_name': 'e11', 'other': {}},
                              {'command': 'met_by', 'parameters': ['e12', 'e13'], 'variable_name': 'e12', 'other': {}},
                              {'command': 'after', 'parameters': ['e13', 'e14'], 'variable_name': 'e13', 'other': {'gap_dist': {'mu': 6, 'sigma': 2}}},
                              {'command': 'during', 'parameters': ['e14', 'e15'], 'variable_name': 'e14', 'other': {}},
                              {'command': 'met_by', 'parameters': ['e15', 'e16'], 'variable_name': 'e15', 'other': {}},
                              {'command': 'after', 'parameters': ['e16', 'e1'], 'variable_name': 'e16', 'other': {'gap_dist': {'mu': 4, 'sigma': 1}}}
                            ]
                      ]


def register_custom_envs():
  register(
    id="OpenTheChests-v0",
    entry_point="openthechests.src.OpenTheChestsGym:OpenTheChestsGym",
    kwargs={
        "instructions": instructions_easy,
        "all_event_types": all_event_types,
        "all_event_attributes": all_event_attributes,
        "all_noise_types": all_noise_types,
        "all_noise_attributes": all_noise_attributes,
        "verbose": False,
        "discrete": False,
        "stb3": True
    }
  )

  register(
    id="OpenTheChests-v1",
    entry_point="openthechests.src.OpenTheChestsGym:OpenTheChestsGym",
    kwargs={
        "instructions": instructions_medium,
        "all_event_types": all_event_types,
        "all_event_attributes": all_event_attributes,
        "all_noise_types": all_noise_types,
        "all_noise_attributes": all_noise_attributes,
        "verbose": False,
        "discrete": False,
        "stb3": True
    }
  )

  register(
    id="OpenTheChests-v2",
    entry_point="openthechests.src.OpenTheChestsGym:OpenTheChestsGym",
    kwargs={
        "instructions": instructions_hard,
        "all_event_types": all_event_types,
        "all_event_attributes": all_event_attributes,
        "all_noise_types": all_noise_types,
        "all_noise_attributes": all_noise_attributes,
        "verbose": False,
        "discrete": False,
        "stb3": True
    }
  )

