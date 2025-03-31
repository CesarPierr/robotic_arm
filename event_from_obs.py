def event_from_obs_gym(obs, types, attributes):
        """
        Extracts event information from the environment observation.
        Converts indices into human-readable symbols and colors.
        """
        return {
            "symbol": types[obs["e_type"]],  # Convert event index to actual event type
            "bg_color": attributes["bg"][obs["bg"]],  # Convert bg index to color
            "symbol_color": attributes["fg"][obs["fg"]],  # Convert fg index to color
            "start_time": obs["start"][0],
            "end_time": obs["end"][0]
        }