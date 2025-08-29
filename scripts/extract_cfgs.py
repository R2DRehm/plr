"""
Scan run directories for checkpoints, extract embedded config metadata (cfg/config/args)
and write a config.json file into the run dir when found. Produce a JSON report
at runs/regeneration_extraction.json.

Usage: python scripts\extract_cfgs.py
"""
import os
import json
import glob
import traceback

try:
    import torch
except Exception:
    print("ERROR: torch not available in this environment. Install PyTorch before running this script.")
    raise

ROOT = os.path.dirname(os.path.dirname(__file__))
RUNS = os.path.join(ROOT, 'runs')
REPORT = os.path.join(RUNS, 'regeneration_extraction.json')
print('DEBUG: ROOT=', ROOT)
print('DEBUG: RUNS=', RUNS)

def try_convert(obj):
    # Try common conversions to plain python dict
    try:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj
        # argparse.Namespace
        if hasattr(obj, '__dict__'):
            return vars(obj)
        # OmegaConf
        try:
            from omegaconf import OmegaConf
            if isinstance(obj, OmegaConf):
                return OmegaConf.to_container(obj, resolve=True)
        except Exception:
            pass
        # dataclass
        try:
            from dataclasses import is_dataclass, asdict
            if is_dataclass(obj):
                return asdict(obj)
        except Exception:
            pass
        # fallback: try json serialization
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)

results = []

print('DEBUG: starting os.walk of', RUNS)

for root, dirs, files in os.walk(RUNS):
    # skip the top-level runs/ itself if there are files
    # find checkpoint files in this dir (not recursively deeper)
    pt_files = [f for f in files if f.endswith('.pt') or f.endswith('.pth')]
    if not pt_files:
        continue
    # prefer best.pt, then model.pt, then first
    ckpt_name = None
    for prefer in ('best.pt', 'model.pt'):
        if prefer in pt_files:
            ckpt_name = prefer
            break
    if ckpt_name is None:
        ckpt_name = pt_files[0]
    ckpt_path = os.path.join(root, ckpt_name)

    entry = {'run': root, 'ckpt': ckpt_name, 'wrote_config': False, 'note': None}
    config_path = os.path.join(root, 'config.json')
    if os.path.exists(config_path):
        entry['note'] = 'config.json already exists'
        results.append(entry)
        continue

    try:
        data = torch.load(ckpt_path, map_location='cpu')
    except Exception as e:
        entry['note'] = 'torch.load failed: ' + repr(e)
        entry['traceback'] = traceback.format_exc()
        results.append(entry)
        continue

    cfg_obj = None
    if isinstance(data, dict):
        for key in ('cfg', 'config', 'args', 'hparams'):
            if key in data:
                cfg_obj = data[key]
                entry['found_key'] = key
                break
        # sometimes checkpoints store nested dict under 'model' and have 'cfg' sibling
        if cfg_obj is None:
            # check nested items
            for v in data.values():
                if isinstance(v, dict) and ('cfg' in v or 'config' in v):
                    if 'cfg' in v:
                        cfg_obj = v['cfg']
                        entry['found_key'] = 'nested.cfg'
                        break
                    if 'config' in v:
                        cfg_obj = v['config']
                        entry['found_key'] = 'nested.config'
                        break
    else:
        # checkpoint is not a dict, nothing to extract
        entry['note'] = 'checkpoint not a dict'

    if cfg_obj is None:
        entry['note'] = entry.get('note') or 'no config-like key found'
        results.append(entry)
        continue

    # convert to plain python dict
    cfg_plain = try_convert(cfg_obj)
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(cfg_plain, f, indent=2, ensure_ascii=False)
        entry['wrote_config'] = True
    except Exception as e:
        entry['note'] = 'failed to write config.json: ' + repr(e)
        entry['traceback'] = traceback.format_exc()

    results.append(entry)

# write report
try:
    with open(REPORT, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Wrote report to {REPORT}")
except Exception as e:
    print('Failed to write report:', e)

# summary
n_total = len(results)
n_written = sum(1 for r in results if r.get('wrote_config'))
print(f"Total checkpoints scanned: {n_total}. Configs written: {n_written}.")

for r in results:
    if r.get('wrote_config'):
        print('WROTE:', r['run'])

print('Done.')
