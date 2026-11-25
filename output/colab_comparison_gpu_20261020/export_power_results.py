#@title 3. 発表用：完了したパワーえさ比較35本を取り出す
import csv, hashlib, json, shutil, zipfile
from pathlib import Path
POWER_ROOT=Path('/content/drive/MyDrive/DQN_Pacman_comparisons_gpu_20261020_9e75afec/results/factors')
POWER_OUT=Path('/content/power_reward_presentation_20261027')
POWER_OUT.mkdir(exist_ok=True)
doc=json.loads((POWER_ROOT/'plan.json').read_text()); plan=doc['plan']
members=plan['groups']['reward_power']; ids={m['condition'] for m in members}
conditions=[c for c in plan['conditions'] if c['id'] in ids]
with (POWER_ROOT/'summary.csv').open(encoding='utf-8-sig',newline='') as f:
 summary=[r for r in csv.DictReader(f) if r['condition'] in ids]
package={'plan':plan,'runtime':doc['runtime'],'members':members,'summary':summary,'runs':[]}
for c in sorted(conditions,key=lambda c:c['config']['rewards']['power']):
 count=0
 for seed in plan['seeds']:
  manifest=dict(schema=1,config=c['config'],seed=seed,**doc['runtime'])
  identity=hashlib.sha256(json.dumps(manifest,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()[:20]
  trial=c['config']['algorithm']+'_'+identity+'_s'+str(seed)
  src=POWER_ROOT/'runs'/trial; dst=POWER_OUT/'runs'/trial
  result=json.loads((src/'result.json').read_text())
  assert result['config']==c['config'] and result['train_seed']==seed and result['total_steps']==200000
  final=json.loads((src/'evaluation_final.json').read_text())
  assert final['phase']=='final' and len(final['episodes'])==100
  dst.mkdir(parents=True,exist_ok=True)
  for name in ['result.json','train.csv','learning_curve.csv','evaluation_final.json','evaluation_final.csv']:
   p=src/name
   if p.exists(): shutil.copy2(p,dst/name)
  package['runs'].append(dict(trial=trial,power_reward=c['config']['rewards']['power'],train_seed=seed,learning_episodes=result['episodes_completed'],total_steps=result['total_steps'],final=final))
  count+=1
 print('POWER_VERIFIED',c['config']['rewards']['power'],count,'training seeds; final 100 episodes each',flush=True)
assert len(package['runs'])==35
(POWER_OUT/'power_results.json').write_text(json.dumps(package,ensure_ascii=False,indent=1))
archive=Path(shutil.make_archive(str(POWER_OUT),'zip',POWER_OUT))
remote=POWER_ROOT.parents[1]/'presentation_power_20261027'
remote.mkdir(exist_ok=True)
shutil.copy2(archive,remote/archive.name)
receipt=dict(training_runs=35,final_evaluation_episodes=3500,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size,drive=str(remote/archive.name))
(POWER_OUT/'receipt.json').write_text(json.dumps(receipt,indent=1))
print('POWER_READY',json.dumps(receipt),flush=True)
from google.colab import files
files.download(str(archive))
