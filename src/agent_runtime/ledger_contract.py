"""Public integer receipt rules and strict ledger proposals; no hidden repair oracle."""
from copy import deepcopy
from .records import fingerprint


class MissingFact(ValueError):
    pass


def expression(expr, facts):
    if type(expr) is int:
        return expr
    if not isinstance(expr, dict):
        raise ValueError('Unknown expression')
    if set(expr) == {'ref'}:
        value = facts
        for key in expr['ref']:
            if not isinstance(value, (dict, list)) or (isinstance(value, dict) and key not in value):
                raise MissingFact('.'.join(map(str, expr['ref'])))
            value = value[key]
        if value is None:
            raise MissingFact('.'.join(map(str, expr['ref'])))
        return value
    op = expr.get('op')
    if op == 'sum' and set(expr) == {'op', 'items'}:
        values = [expression(x, facts) for x in expr['items']]
        if not all(type(x) is int for x in values): raise ValueError('Integer sum required')
        return sum(values)
    if op in {'multiply', 'subtract', 'floor_div', 'min', 'max'} and set(expr) == {'op','left','right'}:
        a, b = expression(expr['left'], facts), expression(expr['right'], facts)
        if type(a) is not int or type(b) is not int: raise ValueError('Integer operands required')
        if op == 'multiply': return a*b
        if op == 'subtract': return a-b
        if op == 'floor_div':
            if b <= 0: raise ValueError('Positive divisor required')
            return a//b
        return min(a,b) if op=='min' else max(a,b)
    if op == 'if_ge' and set(expr) == {'op','left','right','then','else'}:
        a,b = expression(expr['left'],facts),expression(expr['right'],facts)
        if type(a) is not int or type(b) is not int: raise ValueError('Integer comparison required')
        return expression(expr['then'] if a>=b else expr['else'],facts)
    if op == 'sum_where' and set(expr) == {'op','records','field','where'}:
        records=expression(expr['records'],facts)
        if not isinstance(records,list) or set(expr['where'])!={'field','equals'}: raise ValueError('Invalid filtered receipts')
        values=[r[expr['field']] for r in records if r[expr['where']['field']]==expr['where']['equals']]
        if not all(type(x) is int for x in values): raise ValueError('Integer receipts required')
        return sum(values)
    raise ValueError('Unknown public expression structure')


def ledger_changes(case, files, edits, granted):
    if not isinstance(edits,list) or len(edits)>20: raise ValueError('Explicit edit list required')
    result,seen=deepcopy(files),set()
    for edit in edits:
        if not isinstance(edit,dict) or set(edit)!={'file','field','value'}: raise ValueError('Exact edit fields required')
        name,field,value=edit['file'],edit['field'],edit['value']
        if name not in granted or name not in files or field not in case['mutable_fields']:
            raise ValueError('Ledger edit outside explicit grant')
        if type(value) is not int or value<0 or (name,field) in seen: raise ValueError('Unique nonnegative integer edits required')
        seen.add((name,field));result[name][field]=value
    return result


def ledger_check(case, files):
    violations,missing=[],[]
    if set(files)!=set(case['files']): return {'passed':False,'violations':[{'rule':'preserve_file_set'}],'missing_information':[]}
    for name, original in case['files'].items():
        if set(files[name])!=set(original): violations.append({'rule':'preserve_fields','file':name})
        for key,value in original.items():
            actual=files[name].get(key)
            if key not in case['mutable_fields'] and actual!=value:
                violations.append({'rule':'preserve_immutable','file':name,'field':key})
            if key in case['mutable_fields'] and (type(actual) is not int or actual<0):
                violations.append({'rule':'nonnegative_integer','file':name,'field':key})
    if violations: return {'passed':False,'violations':violations,'missing_information':[]}
    for rule in case['rules']:
        if set(rule)!={'id','file','field','expression'}: raise ValueError('Unknown public ledger rule')
        try:
            expected=expression(rule['expression'],case['facts'])
        except MissingFact as exc:
            missing.append({'rule_id':rule['id'],'missing_fact':str(exc)});continue
        if type(expected) is not int or expected<0: raise ValueError('Public rule must produce a nonnegative integer')
        actual=files[rule['file']][rule['field']]
        if actual!=expected:
            # Rule and input are already public. Never return the computed repair value.
            violations.append({'rule_id':rule['id'],'file':rule['file'],'field':rule['field'],'observed':actual})
    return {'passed':not violations and not missing,'violations':violations,'missing_information':missing}


def ledger_outcome(case,task,artifact):
    if artifact is None: return {'semantic_passed':False,'artifact_present':False}
    if fingerprint({k:v for k,v in artifact.items() if k!='sha256'})!=artifact['sha256']:
        raise ValueError('Actual artifact hash mismatch')
    value=artifact['value'];initial=ledger_check(case,task['snapshot']['files'])
    if value['action']=='need_info':
        names=[x['missing_fact'] for x in initial['missing_information']]
        import re
        normalized=' '.join(re.findall(r'[a-z0-9]+',value['summary'].lower()))
        specific=bool(names) and all(' '.join(re.findall(r'[a-z0-9]+',name.lower())) in normalized for name in names)
        return {'semantic_passed':specific,'artifact_present':True,'correct_need_info':specific,
                'missing_fact_names':names,'summary':value['summary'],'files_changed':False}
    candidate=ledger_changes(case,task['snapshot']['files'],value['changes'],task['files'])
    check=ledger_check(case,candidate)
    changed=candidate!=task['snapshot']['files']
    return {'semantic_passed':check['passed'] and (initial['passed'] or changed),
            'artifact_present':True,'correct_need_info':False,'files_changed':changed,
            'candidate_check':check,'candidate_files':candidate,'initial_check':initial}
