import stormpy.shields
from . import shields
from .shields import *

def create_action_lookup(model, scheduler):
    ret = {}

    for state_id in model.states:
        choice = scheduler.get_choice(state_id)
        action = choice.get_deterministic_choice()
        state_valuation = model.state_valuations.get_string(state_id)

        action_to_be_executed = model.choice_labeling.get_labels_of_choice(model.get_choice_index(state_id, action))
        ret[state_valuation] = action_to_be_executed

    return ret

def create_shield_action_lookup(model, shield):
    ret = {}

    for state_id in model.states:
        choices = shield.construct().get_choice(state_id)
        state_valuation = model.state_valuations.get_string(state_id)

        l = []
        for choice in choices.choice_map:
            action = choice[1]
            action_to_be_executed = model.choice_labeling.get_labels_of_choice(model.get_choice_index(state_id, action))
            l.append(action_to_be_executed)
        
        ret[state_valuation] = l

    return ret
