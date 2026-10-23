#!/usr/bin/env python3
"""Plot recorded GPU targets/state/constraint balance around key-side contacts."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--checkpoint',default='best');a=p.parse_args()
    m=json.loads((a.audit/'model_indices.json').read_text());d=a.audit/a.checkpoint/'detail-0'
    ph=np.load(d/'physics_200hz.npz');contacts=np.load(d/'contact_geometry.npz')['contacts']
    report=json.loads((d/'motion_contact_report.json').read_text())
    joints={row['name']:row for row in m['joints']}
    t=(np.arange(len(ph['qpos']))+1)*float(ph['physics_dt'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,2,figsize=(14,11),constrained_layout=True)
    for col,(lo,hi,title) in enumerate(((.8,1.8,'Opening: key underside audit'),(5.15,5.85,'Later C4 contact window'))):
        for ax in axes[:,col]:
            ax.set_xlim(lo,hi);ax.grid(alpha=.2)
            for interval in report['contact_categories']['key_underside']['intervals']:
                ax.axvspan(interval['start'],interval['end'],color='#a64343',alpha=.12)
        for row,name in enumerate(('L_Palm_Base_Pitch','L_Thumb_Rot2')):
            j=joints[name]
            axes[row,col].plot(t,ph['ctrl'][:,j['actuator_id']],lw=1,label='Actuator target')
            axes[row,col].plot(t,ph['qpos'][:,j['qpos_adr']],lw=1.2,label='Actual position')
            axes[row,col].set_ylabel(name+' (rad)');axes[row,col].legend(loc='upper right',fontsize=8)
        j=joints['L_Thumb_Rot2']
        axes[2,col].plot(t,ph['actuator_force'][:,j['actuator_id']],label='Actuator torque',lw=1)
        axes[2,col].plot(t,ph['qfrc_constraint'][:,j['qvel_adr']],label='All constraint torque',lw=1)
        axes[2,col].axhline(-.6,color='grey',ls=':',lw=.8);axes[2,col].axhline(.6,color='grey',ls=':',lw=.8)
        axes[2,col].set_ylabel('L_Thumb_Rot2 torque (Nm)');axes[2,col].legend(fontsize=8,loc='upper right')
        points=contacts[(contacts[:,10]>=0)&(contacts[:,16]<.7)]
        keys=np.array([m['key_geom_midi'][str(int(k))] for k in points[:,10]])
        colors=np.where(points[:,16]<=-.7,'#b72e35','#e49524')
        axes[3,col].scatter((points[:,0]+1)*.005,keys,s=5,c=colors)
        axes[3,col].set_ylabel('Key MIDI: orange side, red underside');axes[3,col].set_xlabel('Policy time (s); music starts at 1.0 s')
        axes[0,col].set_title(title)
    fig.suptitle('Recorded GPU trajectory: commands, actual joint state and key contact\nConstraint torque includes all constraints; shading marks underside-contact intervals')
    fig.savefig(d/'key_contact_motion.png',dpi=170)
    print(d/'key_contact_motion.png')


if __name__=='__main__':main()
