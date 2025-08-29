"""
Simpler extractor: load checkpoints, if top-level dict contains 'cfg'/'config'/'args'
write it to run/config.json. Print progress and write a report at runs/regeneration_extraction_simple.json
"""
import os, json, traceback
try:
    import torch
except Exception as e:
    print('ERROR: torch import failed:', e)
    raise

ROOT = os.path.dirname(os.path.dirname(__file__))
RUNS = os.path.join(ROOT, 'runs')
REPORT = os.path.join(RUNS, 'regeneration_extraction_simple.json')

results = []
count=0
for d in sorted(os.listdir(RUNS)):
    path = os.path.join(RUNS, d)
    if not os.path.isdir(path):
        continue
    # look for pt/pth in path (non-recursive)
    files = os.listdir(path)
    pt_files = [f for f in files if f.endswith('.pt') or f.endswith('.pth')]
    # also allow nested one-level in archive subdir
    if not pt_files:
        # check subdirs
        for sub in sorted(os.listdir(path)):
            subp = os.path.join(path, sub)
            if os.path.isdir(subp):
                for f in os.listdir(subp):
                    if f.endswith('.pt') or f.endswith('.pth'):
                        pt_files.append(os.path.join(sub, f))
                if pt_files:
                    break
    if not pt_files:
        continue
    # prefer best.pt then model.pt
    prefer = None
    for p in ('best.pt','model.pt'):
        if p in pt_files:
            prefer = p
            break
    if prefer is None:
        # choose first
        prefer = pt_files[0]
    ckpt_path = os.path.join(path, prefer) if os.path.isabs(prefer)==False else os.path.join(path, prefer)
    # if prefer includes subdir path like 'subdir\best.pt'
    if not os.path.exists(ckpt_path):
        # try join differently
        ckpt_path = os.path.join(path, os.path.basename(prefer))
    entry = {'run': path, 'ckpt': ckpt_path, 'wrote': False, 'note': ''}
    try:
        data = torch.load(ckpt_path, map_location='cpu')
    except Exception as e:
        entry['note'] = 'load failed: '+repr(e)
        entry['traceback'] = traceback.format_exc()
        results.append(entry)
        continue
    cfg = None
    if isinstance(data, dict):
        for key in ('cfg','config','args','hparams'):
            if key in data:
                cfg = data[key]
                entry['found_key']=key
                break
    if cfg is None:
        entry['note']='no config-like key'
        results.append(entry)
        continue
    # convert to plain json-serializable
    try:
        json.dumps(cfg)
        cfg_plain = cfg
    except TypeError:
        # try to convert simple objects
        try:
            cfg_plain = {k: (v if isinstance(v,(int,float,str,bool,list,dict)) else str(v)) for k,v in (cfg.items() if isinstance(cfg,dict) else [])}
        except Exception:
            cfg_plain = str(cfg)
    try:
        with open(os.path.join(path,'config.json'),'w',encoding='utf-8') as f:
            json.dump(cfg_plain,f,indent=2,ensure_ascii=False)
        entry['wrote']=True
    except Exception as e:
        entry['note']='write failed: '+repr(e)
        entry['traceback']=traceback.format_exc()
    results.append(entry)
    count+=1

with open(REPORT,'w',encoding='utf-8') as f:
    json.dump(results,f,indent=2,ensure_ascii=False)
print('Done. scanned',len(results),'entries, wrote configs:', sum(1 for r in results if r.get('wrote')))
