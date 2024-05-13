from enum import Enum

def LOGTYPE(log_name):
    if log_name == 'BPIC15_1.csv':
        settings = {
        'Activity'              : str,
        'caseStatus'            : str,
        'last_phase'            : str,
        'parts'                 : str,
        'termName'              : str, 
        'requestComplete'       : bool,
        'Includes_subCases'     : str, 
        'Resource'              : str,
        'monitoringResource'    : str, 
        'Responsible_actor'     : str,
        'Case ID'               : str,
        'Complete Timestamp'    : 'timestamp',
        'SUMleges'              : float, 
        }
    elif log_name == 'BPIC11.csv':
        settings = {
        'Case ID'               : str,
        'Activity'              : str, # nunique of Acctivity and Activity code is different....how???
        'Complete Timestamp'    : 'timestamp',
        'org:group'             : str,
        'Number of executions'  : float,
        'Specialism code.1'     : str,
       # 'Producer code'         : str,  excluded dueto memory issue
        'Section'               : str   
        }
    elif log_name == 'credit-card-new.csv':
        settings = {
        'Case ID'               : str,
        'Activity'              : str, # nunique of Acctivity and Activity code is different....how???
        'Complete Timestamp'    : 'timestamp',
        'Variant'             : str,
        'elementId'     : str,
        'Resource'               : str,
        'resourceCost'  : float,
        'resourceId'    :str
        }        
    elif log_name == 'pub-new.csv':
        settings = {
        'Case ID'               : str,
        'Activity'              : str, # nunique of Acctivity and Activity code is different....how???
        'Complete Timestamp'    : 'timestamp',
        'Variant'             : str,
        'elementId'     : str,
        'Resource'               : str,
        'resourceId'    :str
        }                
    return settings

def LOGCOLUMN(log_name):
    if log_name == 'BPIC15_1.csv':
        class COLUMN(Enum):
            CASE = 'Case ID'
            ACT  = 'Activity'
            TIME = 'Complete Timestamp'
            OUTCOME = 'requestComplete'
            ATTRLIST = ['caseStatus','last_phase','parts',
                    'termName','Includes_subCases',
                    'Resource','monitoringResource',
                    'Responsible_actor','SUMleges']
            
    elif log_name == 'BPIC11.csv':
        class COLUMN(Enum):
            CASE = 'Case ID'
            ACT  = 'Activity'
            TIME = 'Complete Timestamp'
            OUTCOME = None
            ATTRLIST = ['org:group','Number of executions',
                        'Specialism code.1','Section']
                       # 'Producer code',
    elif log_name == 'credit-card-new.csv':
        class COLUMN(Enum):
            CASE = 'Case ID'
            ACT  = 'Activity'
            TIME = 'Complete Timestamp'
            OUTCOME = None
            ATTRLIST = ['Variant','elementId','Resource',
                        'resourceCost','resourceId']
            
    elif log_name == 'pub-new.csv':
        class COLUMN(Enum):
            CASE = 'Case ID'
            ACT  = 'Activity'
            TIME = 'Complete Timestamp'
            OUTCOME = None
            ATTRLIST = ['Variant','elementId','Resource',
                        'resourceId']
    return COLUMN

class NAME(Enum):
    ATTR = 'Attributes'
    LABEL = 'Label'
    TSSC = 'timesincecasestart'
    TSP = 'timesincelastevent'
    MAXLEN = 'max_length'
    ATTRSZ = 'attr_size'
    OUTDIM = 'output_dim'
    NUMACT = 'num_act'
    ACT = 'Activity'

class TASK(Enum):
    NAP = "next_activity"
    OP = "outcome"
    ERTP = "event_remaining_time"
    CRTP = "case_remaining_time"



