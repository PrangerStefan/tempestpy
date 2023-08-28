import minigrid
from minigrid.core.actions import Actions

   
def extract_keys(env):
    env.reset()
    keys = []
    print(env.grid)
    for j in range(env.grid.height):
        for i in range(env.grid.width):
            obj = env.grid.get(i,j)
            
            if obj and obj.type == "key":
                keys.append(obj.color)
    
    return keys


def get_action_index_mapping(actions):
    for action_str in actions:
        if "left" in action_str:
            return Actions.left
        elif "right" in action_str:
            return Actions.right
        elif "east" in action_str: 
            return Actions.forward
        elif "south" in action_str:
            return Actions.forward
        elif "west" in action_str:
            return Actions.forward
        elif "north" in action_str:
            return Actions.forward
        elif "pickup" in action_str:
            return Actions.pickup
        elif "done" in action_str:
            return Actions.done
    
    
    raise ValueError(F"Action string {action_str} not supported")