import glob
import json
import pandas as pd
import numpy as np
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
        


###################################### explore tensorboard logs ################################
#these methods are quite quick and dirty, but have been checked for correctness

def collect_traces(out_dir, attention):
    # returns a dictionary of traceEvents for all runs in out_dir with selected attention
    # traceEvents is a list of events dictionaries
    # pt.trace.json keys:['schemaVersion', 'deviceProperties', 'distributedInfo', 'with_flops', 
    # 'with_modules', 'with_stack', 'traceEvents', 'traceName']
    data = {}
    for run in glob.glob(f"{out_dir}/**/*.pt.trace.json", recursive=True):
        if run.split('logs_')[1].split('-')[0]==attention:
            with open(run) as jsonFile:
                events_dict = json.load(jsonFile)
                run_name = events_dict['traceName'].split('logs_')[1].split('/')[0]
                events = events_dict['traceEvents']
                data[run_name] =  events
    return data


def kernel_mean_duration(runTrace, kernel_name):
    #aggregates call duration for all calls of runTrace=traceEvents
    #computes the average call duration for each run

    #TODO add lambda for regex match with kernel_name

    keys = {'name', 'dur'}
    kernel_events = [{k: trace[k] for k in keys} for trace in runTrace if 'cat' in trace.keys() if trace['cat']=='kernel']

    kernel_events = [event for event in kernel_events if kernel_name == event['name']]
    kernel_summary = {event['name']: {'dur': 0, 'calls':0} for event in kernel_events}

    for event in kernel_events:
        kernel_summary[event['name']]['dur'] += event['dur']
        kernel_summary[event['name']]['calls'] += 1
        avg =  np.round(kernel_summary[event['name']]['dur']/ kernel_summary[event['name']]['calls'])
        kernel_summary[event['name']]['mean_duration'] = int(avg)

    return kernel_summary


def kernel_mean_occupancy(runTrace, kernel_name):
    #aggregates 'est. achieved occupancy %' and call duration for all calls of runTrace=traceEvents
    #computes the weighted average of occupancy with weights=call duration for each run

    #TODO add lambda for regex match with kernel_name

    keys = {'name', 'args', 'dur'}
    #select subset of events: kernel kernels:
    kernel_events = [{k: trace[k] for k in keys} for trace in runTrace if 'cat' in trace.keys() if trace['cat']=='kernel']
    #select events for a kernel with name = kerne_name:
    kernel_events = [event for event in kernel_events if kernel_name == event['name']]
    kernel_aggregate = {event['name']: {'occupancy': 0,  'calls':0, 'dur':0} for event in kernel_events}
    kernel_summary = {event['name']: {'mean_occupancy': 0} for event in kernel_events}

    for event in kernel_events:
        kernel_aggregate[event['name']]['occupancy'] += event['args']['est. achieved occupancy %']*event['dur']
        kernel_aggregate[event['name']]['calls'] += 1
        kernel_aggregate[event['name']]['dur'] += event['dur']
        weighted_avg = np.round(kernel_aggregate[event['name']]['occupancy']/ kernel_aggregate[event['name']]['dur'],2)
        kernel_summary[event['name']]['mean_occupancy'] = weighted_avg
    return kernel_summary


def aggregate_metrics(kernel_name, metric, runs):
    if metric == 'mean_duration':
        return [kernel_mean_duration(run, kernel_name)[kernel_name]['mean_duration'] for run in runs.values()]
    elif metric == 'mean_occupancy':
        return [kernel_mean_occupancy(run, kernel_name)[kernel_name]['mean_occupancy'] for run in runs.values()]
    else:
        raise ValueError('valid metrics are mean_duration and mean_occupancy')
    

def compare_runs(out_dir, attention, kernel_name):
    #outputs the metrics = mean_duration, mean_occupancy for kernel=kernel_name 
    # for different runs
    #this recovers the exact same values in the tensorboard

    runs = collect_traces(out_dir, attention=attention) #index
    cols = ['mean_duration', 'mean_occupancy']
    aggregated_metrics = {metric : aggregate_metrics(kernel_name, metric, runs) for metric in cols}
    
    df =  pd.DataFrame(aggregated_metrics, index = runs.keys())
    df.index.name = attention
    
    return df