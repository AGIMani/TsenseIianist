import csv,json,hashlib,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
d=Path(__file__).resolve().parent
rows=list(csv.DictReader((d/'note_diagnostics.csv').open()));changes=list(csv.DictReader((d/'changes.csv').open()));report=json.loads((d/'fingering_report.json').read_text())
fingers=['R thumb','R index','R middle','R ring','R little','L thumb','L index','L middle','L ring','L little']
fig,ax=plt.subplots(3,1,figsize=(14,10),layout='constrained',gridspec_kw={'height_ratios':[1.1,2.2,1]})
x=np.arange(10);means=[];evals=[]
for f in range(10):
 rr=[r for r in rows if int(r['finger_id'])==f]
 means.append(np.mean([float(r['coverage']) for r in rr]));evals.append(np.mean([float(r['evaluation_hold_coverage']) for r in rr]))
ax[0].bar(x-.18,np.array(means)*100,.36,label='Video replay');ax[0].bar(x+.18,np.array(evals)*100,.36,label='3 evaluation episodes')
ax[0].set(xticks=x,xticklabels=fingers,ylim=(0,100),ylabel='Mean note hold coverage (%)');ax[0].legend(loc='upper right');ax[0].grid(axis='y',alpha=.2)
sc=ax[1].scatter([float(r['start_seconds']) for r in rows],[int(r['pitch']) for r in rows],c=[float(r['coverage']) for r in rows],vmin=0,vmax=1,cmap='RdYlGn',s=15)
ax[1].scatter([float(r['start_seconds']) for r in changes],[int(r['pitch']) for r in changes],facecolors='none',edgecolors='black',s=65,linewidths=.7,label='61 revised assignments')
ax[1].set(xlabel='Score time (s), preparation excluded',ylabel='MIDI pitch',title='Measured original fingering: red = poorly held; circles = revised notes');ax[1].legend(loc='lower left');fig.colorbar(sc,ax=ax[1],label='Recorded hold fraction')
before=[sum(int(r['finger_id'])==f for r in rows) for f in range(10)];after=[report['counts_by_finger'][str(f)] for f in range(10)]
ax[2].bar(x-.18,before,.36,label='Original assignment');ax[2].bar(x+.18,after,.36,label='Revised assignment');ax[2].set(xticks=x,xticklabels=fingers,ylabel='Assigned notes');ax[2].legend()
fig.suptitle('Prelude BWV 847 / exper009 best.pt / 200 Hz physical keys\nEvidence: yaw 0.15 rad, old joint ranges. New fingering performance has not been measured.',fontsize=14)
fig.savefig(d/'diagnosis.png',dpi=160);plt.close(fig)
