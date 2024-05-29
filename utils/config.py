from enum import Enum

def LOGTYPE(injection,task):
    if (injection != INJM.CLN.value )and(task != TASK.OP.value):
        settings = {
        'Activity'              : str,
        'Case'               : str,
        'Timestamp'    : 'timestamp',
        'Resource'              : str, 
        'Injection'              : str
        }
    elif (injection != INJM.CLN.value )and(task == TASK.OP.value):
        settings = {
        'Activity'              : str,
        'Case'               : str,
        'Timestamp'    : 'timestamp',
        'Resource'              : str, 
        'Injection'              : str,
        'label'             :str
        }
    elif (injection == INJM.CLN.value )and(task == TASK.OP.value):
        settings = {
        'Activity'              : str,
        'Case'               : str,
        'Timestamp'    : 'timestamp',
        'Resource'              : str, 
        'label'             :str
        }
        
    else:
        settings = {
        'Activity'              : str,
        'Case'               : str,
        'Timestamp'    : 'timestamp',
        'Resource'              : str, 
        }
    return settings


class COL(Enum):
    ACT = 'Activity'
    CASE = 'Case'
    TIME = 'Timestamp'
    RES = 'Resource'
    INJ = 'Injection'
    LABEL = 'Label'
    TSSC = 'Timesincecasestart'
    TSP = 'Timesincelastevent'
    OC = 'label'

class META(Enum):
    MAXLEN = 'max_length'
    ATTRSZ = 'attr_size'
    OUTDIM = 'output_dim'
    NUMACT = 'num_act'
    SCALER = 'minmax_scaler'

class INJM(Enum):
    DIST = 'DISTORTED'
    POLN = 'POLLUTED.NORND'
    POLR = 'POLLUTED.RANDOM'
    HOM = 'HOMONYM'
    CLN = "CLEAN"
    SYN = "SYNONYM"
    
class TASK(Enum):
    NAP = "next_activity"
    OP = "outcome"
    ERTP = "event_remaining_time"
    CRTP = "case_remaining_time"



