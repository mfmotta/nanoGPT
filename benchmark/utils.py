from contextlib import contextmanager

######################  helpers to manipulate dictionary of parameters ###################

def expand_key_values_to_list(d, k):
    # expands a dictionary of lists d = {... k:[v0,...vn]  ... } on key k
    # into a list of n dictionaries [{k:v0}, ...{k:vn}]
    list_of_key_value_dicts = []
    for v in d[k]:
        list_of_key_value_dicts.append({k:v})
    return list_of_key_value_dicts

def expand_all_keys(d):
    #creates a list of expanded_dicts for all keys of d
    return [expand_key_values_to_list(d, k) for k in d.keys()]

def combine_all_keys_and_values(d):
    # creates a list of dictionaries that has all 
    # possible combinations of expanded key-values
    expanded_dicts_list = expand_all_keys(d)

    dict_list = []
    d = {}
    level=0

    def nested_loop(d, level):
        if level==len(expanded_dicts_list):
            return d.copy()
        for dn in expanded_dicts_list[level]:
            d.update(dn)
            loop = nested_loop(d, level+1)
            if loop is not None:
                dict_list.append(loop) 


    nested_loop(d, level)

    return dict_list

##########################################################################################################


class ParameterUpdateError(Exception):
    """Custom exception for errors during parameter updates."""
    def __init__(self, message):
        super().__init__(message)


@contextmanager
def set_params(original_params, new_params):
    #original_params is a module 
    original_values = {}
    for param, value in new_params.items():
        try:
            original_values[param] = getattr(original_params, param)
            setattr(original_params, param, value)#set new value to param attribute
        except Exception as e:
            raise ParameterUpdateError(f"Failed to update parameter {param}: {e}")
    
    yield new_params

    # Restore parameters
    for param, value in new_params.items():
        try:
            setattr(original_params, param, original_values[param])
        except Exception as e:
            raise ParameterUpdateError(f"Failed to restore parameter {param} to its original value: {e}")
        


###################################### explor tensorboard logs ################################
#these methods are quite quick and dirty, but have been checked for correctness

def collect_traces(attention='flash'):
    #returns a list of traceEvents dictionaries for all runs with selected 
    #pt.trace.json keys:['schemaVersion', 'deviceProperties', 'distributedInfo', 'with_flops', 'with_modules', 'with_stack', 'traceEvents', 'traceName']
    data = []
    for run in glob.glob(f"{params.out_dir}/**/*.pt.trace.json", recursive=True):
        if run.split('logs_')[1].split('-')[0]==attention:
            with open(run) as jsonFile:
                data.append(json.load(jsonFile)['traceEvents'])
    return data
    
def kernel_mean_duration(runTrace, name_filter = None):
    #aggregates call duration for all calls of runTrace=traceEvents
    #computes the average call duration for each run

    #TODO add lambda for regex match with name_filer

    keys = {'name', 'dur'}
    kernel_events = [{k: trace[k] for k in keys} for trace in runTrace if 'cat' in trace.keys() if trace['cat']=='kernel']

    if name_filter is not None:
        kernel_events = [event for event in kernel_events if name_filter == event['name']]
    kernel_summary = {event['name']: {'dur': 0, 'calls':0} for event in kernel_events}

    for event in kernel_events:
        kernel_summary[event['name']]['dur'] += event['dur']
        kernel_summary[event['name']]['calls'] += 1
        avg =  np.round(kernel_summary[event['name']]['dur']/ kernel_summary[event['name']]['calls'])
        kernel_summary[event['name']]['mean_duration'] = int(avg)

    return kernel_summary


def kernel_mean_occupancy(runTrace, name_filter = None):
    #aggregates 'est. achieved occupancy %' and call duration for all calls of runTrace=traceEvents
    #computes the weighted average of occupancy with weights=call duration for each run

    #TODO add lambda for regex match with name_filer

    keys = {'name', 'args', 'dur'}
    kernel_events = [{k: trace[k] for k in keys} for trace in runTrace if 'cat' in trace.keys() if trace['cat']=='kernel']

    if name_filter is not None:
        kernel_events = [event for event in kernel_events if name_filter == event['name']]
    kernel_aggregate = {event['name']: {'occupancy': 0,  'calls':0, 'dur':0} for event in kernel_events}
    kernel_summary = {event['name']: {'mean_occupancy': 0} for event in kernel_events}

    for event in kernel_events:
        kernel_aggregate[event['name']]['occupancy'] += event['args']['est. achieved occupancy %']*event['dur']
        kernel_aggregate[event['name']]['calls'] += 1
        kernel_aggregate[event['name']]['dur'] += event['dur']
        weighted_avg = np.round(kernel_aggregate[event['name']]['occupancy']/ kernel_aggregate[event['name']]['dur'],2)
        kernel_summary[event['name']]['mean_occupancy'] = weighted_avg
    return kernel_summary