"""Validate LogTriage JSONL data."""
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TAX=json.loads((ROOT/'data'/'label_taxonomy.json').read_text())
def validate_record(r):
    issues=[]; lab=r.get('labels',{}); logs=r.get('logs',[])
    for f in ['failure_cause','affected_service','severity','recommended_action']:
        if lab.get(f) not in TAX[f]: issues.append('invalid '+f)
    for i in lab.get('evidence_lines',[]):
        if not isinstance(i,int) or i<1 or i>len(logs): issues.append('bad evidence line')
    return issues
if __name__=='__main__':
    path=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'data'/'logtriage_train.jsonl'
    bad=[]
    for n,l in enumerate(open(path,encoding='utf-8'),1):
        issues=validate_record(json.loads(l))
        if issues: bad.append((n,issues))
    print(json.dumps({'bad_records':len(bad),'sample':bad[:10]},indent=2))
