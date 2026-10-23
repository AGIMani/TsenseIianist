"""Video/trace export for the current SceneArmFollower component demonstration.

Used by render_cartesian_installation.py --video inside Slurm. No policy,
checkpoint, music target or piano physics is involved in this component test.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import resource
import time


def export_video(root, out, font_path, bundle, model, data, follower, render, report):
    import imageio_ffmpeg
    import mujoco
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont
    from tsensei_v3_2_pianist.preview_backend import current_gl_info

    dt, fps, duration = follower.dt, 25, 14.
    stride = round(1 / fps / dt)
    control_stride = round(bundle.manifest['control_dt'] / dt)
    assert abs(stride * dt - 1 / fps) < 1e-12
    n = round(duration/dt)
    paced = bool(report.get('scene_realtime'))
    if paced and report.get('execution_backend') != 'numba_cpu':
        raise ValueError('Scene realtime benchmark requires native backend')
    physical = torch.as_tensor(bundle.neutral_action[None], dtype=torch.float64, device=follower.device).clone()
    p0 = follower.wrists['right']['rest_pos'].cpu().numpy().copy()
    prep = (follower.preparation or {}).get('duration', 0.)
    sides = ('right', 'left')
    count=round(report.get('realtime_seconds',56)/dt) if paced else n
    if count<n or count%n:raise ValueError('Realtime duration must be a positive multiple of 14 seconds')
    q = np.empty((count+1,2,7)); velocity=np.empty_like(q); acceleration=np.empty_like(q)
    actual = np.empty((count+1,2,3)); target=np.empty_like(actual)
    actual_rot = np.empty((count+1,2,3,3)); target_rot=np.empty_like(actual_rot)
    # Commit recording pages before the real-time clock (not on first write).
    for storage in (q,velocity,acceleration,actual,target,actual_rot,target_rot):storage.fill(0)
    report['recording_pages_prefaulted']=True
    elapsed_tensor=torch.empty(1,device=follower.device,dtype=torch.float64)
    frame_times=np.arange(n+1)*dt
    requested=np.zeros((n+1,3))

    def schedule(t):
        displacement=np.zeros(3)
        if not 1 <= t < 13:
            return displacement, -1, '起始静止' if t<1 else '回到起点'
        axis=min(int((t-1)//4),2); u=(t-1)%4
        negative=-.020 if axis<2 else .010
        knots=[(0.,0.),(.5,0.),(1.,.020),(1.4,.020),(1.9,0.),(2.2,0.),(2.7,negative),(3.1,negative),(3.6,0.),(4.,0.)]
        for (ta,a),(tb,b) in zip(knots[:-1],knots[1:]):
            if ta<=u<=tb:
                s=(u-ta)/(tb-ta); smooth=s**3*(10-15*s+6*s*s)
                displacement[axis]=a+(b-a)*smooth
                break
        return displacement, axis, '世界 '+'XYZ'[axis]+' 轴'

    def save(i, targets):
        for j,s in enumerate(sides):
            a=follower.sides[s]
            if paced:
                bid=follower.wrists[s]['body_id']
                actual[i,j]=data.xpos[bid];actual_rot[i,j]=data.xmat[bid].reshape(3,3)
            else:
                p,r,*_=follower.fk(a,a['q'])
                actual[i,j]=p[0].cpu().numpy();actual_rot[i,j]=r[0].cpu().numpy()
            if paced:
                state=follower._native_state[0,j]
                q[i,j]=state[0];velocity[i,j]=state[1];acceleration[i,j]=state[2]
                target[i,j]=follower._native_targets[0,j,:3]
                target_rot[i,j]=follower._native_targets[0,j,3:].reshape(3,3)
            else:
                q[i,j]=a['q'][0].cpu().numpy();velocity[i,j]=a['v'][0].cpu().numpy()
                acceleration[i,j]=a['acceleration'][0].cpu().numpy()
                target[i,j]=targets[s][0][0].cpu().numpy();target_rot[i,j]=targets[s][1][0].cpu().numpy()

    initial_targets=follower.requests(physical,torch.tensor([prep],device=follower.device,dtype=torch.float64))
    if paced:
        follower.publish_numpy(data.mocap_pos,data.mocap_quat)
        mujoco.mj_forward(model,data)
    save(0,initial_targets)
    for j,s in enumerate(sides):
        target[0,j]=initial_targets[s][0][0].cpu().numpy();target_rot[0,j]=initial_targets[s][1][0].cpu().numpy()
    import gc
    started=time.perf_counter()
    count=round(report.get('realtime_seconds',56)/dt) if paced else n
    if count<n or count%n:raise ValueError('Realtime duration must be a positive multiple of 14 seconds')
    latency=np.empty(count);lateness=np.zeros(count);start_delay=np.zeros(count)
    stage=np.zeros((count,3))
    profiling=bool(report.get('latency_profile'))
    user_system=np.zeros((count,2))
    cpu_stage=np.zeros((count,3));usage_delta=np.zeros((count,4),dtype=np.int64)
    # Commit timing/profiling pages before the paced loop, including zeros.
    for record in (latency,lateness,start_delay,stage,user_system,cpu_stage,usage_delta):record.fill(0)
    # Thread CPU versus wall time separates execution from descheduling.
    def usage():
        u=resource.getrusage(resource.RUSAGE_THREAD)
        return (u.ru_nvcsw,u.ru_nivcsw,u.ru_minflt,u.ru_majflt,u.ru_utime,u.ru_stime)
    def process_snapshot():
        from pathlib import Path
        result={'cgroup':Path('/proc/self/cgroup').read_text(), 'sched':Path('/proc/self/sched').read_text(),'numa_maps':Path('/proc/self/numa_maps').read_text(),'threads':{}}
        for item in Path('/proc/self/task').iterdir():
            try:
                result['threads'][item.name]={line.split(':')[0]:line.split(':',1)[1].strip() for line in (item/'status').read_text().splitlines() if line.startswith(('Name:','Cpus_allowed_list:','voluntary_ctxt_switches:','nonvoluntary_ctxt_switches:'))}
            except FileNotFoundError:pass
        try:
            from threadpoolctl import threadpool_info
            result['threadpools']=threadpool_info()
        except ImportError:result['threadpools']='threadpoolctl unavailable'
        cg=Path('/sys/fs/cgroup')/result['cgroup'].strip().split(':')[-1].lstrip('/')
        result['cgroup_stats']={}
        while cg!=Path('/sys/fs'):
            for key in ('cpu.stat','cpu.max','memory.events'):
                try:result['cgroup_stats'][str(cg/key)]=(cg/key).read_text()
                except OSError:pass
            cg=cg.parent
        return result
    snapshot_before=process_snapshot() if profiling else None
    gc_enabled=gc.isenabled()
    clock_mode=report.get('realtime_clock','sleep')
    affinity_before=sorted(os.sched_getaffinity(0))
    pinned_cpu=None
    if paced and clock_mode=='pinned_busy':
        # Require Slurm to bind this process to its granted CPUs first. Never
        # select a CPU from a machine-wide unrestricted affinity mask.
        granted=int(os.environ['SLURM_CPUS_PER_TASK'])
        if len(affinity_before)>granted:
            raise RuntimeError('Run this benchmark in a Slurm step with --cpu-bind=threads')
        pinned_cpu=affinity_before[report.get("pin_cpu_index",0)]
        os.sched_setaffinity(0,{pinned_cpu})
    if paced:gc.disable()
    clock_start=time.perf_counter()+.05
    try:
        for i in range(1,count+1):
            k=(i-1)%n+1
            deadline=clock_start+(i-1)*dt
            if paced:
                remaining=deadline-time.perf_counter()
                if clock_mode=='pinned_busy':
                    while time.perf_counter()<deadline:pass
                elif remaining>0:time.sleep(remaining)
            tic=time.perf_counter()
            if profiling:u0=usage();c0=time.thread_time()
            start_delay[i-1]=max(0.,tic-deadline) if paced else 0.
            if (k-1)%control_stride==0:
                displacement,_,_=schedule((k-1)*dt)
                physical[:,follower.wrists['right']['indices']]=torch.as_tensor(displacement,device=follower.device)
            elapsed_tensor.fill_(prep+i*dt)
            targets=follower.requests(physical,elapsed_tensor)
            follower.step(targets)
            t1=time.perf_counter()
            if profiling:c1=time.thread_time()
            if paced:
                follower.publish_numpy(data.mocap_pos,data.mocap_quat)
                mujoco.mj_forward(model,data)
            t2=time.perf_counter()
            if profiling:c2=time.thread_time()
            # Identical complete record workload on every measured cycle.
            save(i,targets)
            if i<=n:requested[k]=displacement
            t3=time.perf_counter()
            if profiling:
                c3=time.thread_time();u1=usage()
                cpu_stage[i-1]=(c1-c0,c2-c1,c3-c2)
                usage_delta[i-1]=np.asarray(u1[:4])-u0[:4]
                user_system[i-1]=np.asarray(u1[4:])-u0[4:]
            stage[i-1]=(t1-tic,t2-t1,t3-t2)
            # The deadline also covers profiling and stage bookkeeping. Only
            # publishing the two final timing scalars follows this timestamp.
            completed=time.perf_counter()
            latency[i-1]=completed-tic
            lateness[i-1]=max(0.,completed-(deadline+dt)) if paced else 0.
            if not paced and i%200==0:
                print(f'IK trace {i*dt:.1f}/{duration:.1f} s',flush=True)
    finally:
        if pinned_cpu is not None:os.sched_setaffinity(0,set(affinity_before))
        if gc_enabled:gc.enable()
    if follower.device.type=='cuda':torch.cuda.synchronize()
    if profiling:
        (out/'process_profile.json').write_text(json.dumps({'before':snapshot_before,'after':process_snapshot()},indent=2)+'\n')
        np.savez_compressed(out/'latency_profile.npz',thread_cpu_stage_seconds=cpu_stage,thread_user_system_seconds=user_system,usage_delta=usage_delta,usage_columns=['voluntary_switch','involuntary_switch','minor_fault','major_fault'])
    report['trace_compute_wall_seconds']=time.perf_counter()-started
    if paced:
        def stat(a):
            a=np.asarray(a)*1000
            return dict(mean_ms=float(a.mean()),p99_ms=float(np.percentile(a,99)),max_ms=float(a.max()))
        timing=dict(cycles=count,simulated_seconds=count*dt,cycle=stat(latency),
            clock_mode=clock_mode,allowed_cpus=affinity_before,pinned_cpu=pinned_cpu,
            scheduling_priority='ordinary Linux scheduling, no real-time privilege', 
            requests_ik=stat(stage[:,0]),scene_publish_forward=stat(stage[:,1]),logging=stat(stage[:,2]),
            start_delay=stat(start_delay),deadline_misses=int(np.count_nonzero(lateness)),
            max_finish_late_ms=float(lateness.max()*1000),compute_over_5ms=int(np.sum(latency>dt)),
            passed=bool(not np.any(lateness)),scope='20 Hz prescribed targets + native IK + actual MuJoCo mocap publication + mj_forward + memory record + diagnostic bookkeeping; rendering asynchronous after measurement; no mj_step or policy')
        report['scene_timing']=timing
        np.savez_compressed(out/'scene_timing.npz',cycle_seconds=latency,stage_seconds=stage,start_delay_seconds=start_delay,finish_lateness_seconds=lateness)
        (out/'scene_timing.json').write_text(json.dumps(timing,indent=2)+'\n')
        print('SCENE TIMING',json.dumps(timing),flush=True)
    if paced:
        np.savez_compressed(out/'constraint_trace_full.npz',time_s=np.arange(count+1)*dt,
            q=q,velocity=velocity,acceleration=acceleration,palm_actual_m=actual,palm_target_m=target)
    full_error=np.linalg.norm(target-actual,axis=-1)
    over=(full_error[1:]>.001).any(axis=1)
    report['tracking_error_statistics']={
        'threshold_m':.001,'scope':'prescribed post-control-shaping component target; not raw policy actions',
        'max_error_mm':float(full_error.max()*1000),
        'over_1mm_substeps':int(over.sum()),'sampled_substeps':int(count),
        'over_1mm_entry_events':int(np.sum(over & ~np.r_[False,over[:-1]])),
        'within_1mm_substep_fraction':float(1-over.mean()),
        'per_side_over_1mm_substeps':dict(zip(sides,(full_error[1:]>.001).sum(axis=0).tolist()))}
    report["full_trace_max_error_mm"]=float(full_error.max()*1000)
    # Audit every executed cycle, including those after the displayed segment.
    np.testing.assert_allclose(np.diff(q,axis=0)/dt,velocity[1:],atol=1e-9,rtol=0)
    np.testing.assert_allclose(np.diff(velocity,axis=0)/dt,acceleration[1:],atol=1e-7,rtol=0)
    for j,s in enumerate(sides):
        a=follower.sides[s]
        assert np.all(np.abs(velocity[:,j])<=a['vmax'].cpu().numpy()+1e-8)
        assert np.all(np.abs(acceleration[:,j])<=a['amax'].cpu().numpy()+1e-7)
        assert np.all(q[:,j]>=a['low'].cpu().numpy()-1e-10) and np.all(q[:,j]<=a['high'].cpu().numpy()+1e-10)
        assert np.max(1.025*np.abs(q[:,j,5])+np.abs(q[:,j,6])) <= follower.coupling_sum_limit+1e-10
    q,velocity,acceleration,actual,target,actual_rot,target_rot = [
        x[:n+1] for x in (q,velocity,acceleration,actual,target,actual_rot,target_rot)]
    error=np.linalg.norm(target[:,0]-actual[:,0],axis=1)*1000
    # Verify the recorded rate-limited execution, not just the IK reference.
    np.testing.assert_allclose(np.diff(q,axis=0)/dt,velocity[1:],atol=1e-9,rtol=0)
    np.testing.assert_allclose(np.diff(velocity,axis=0)/dt,acceleration[1:],atol=1e-7,rtol=0)
    for j,s in enumerate(sides):
        a=follower.sides[s]
        assert np.all(np.abs(velocity[:,j])<=a['vmax'].cpu().numpy()+1e-8)
        assert np.all(np.abs(acceleration[:,j])<=a['amax'].cpu().numpy()+1e-7)
        assert np.all(q[:,j]>=a['low'].cpu().numpy()-1e-10) and np.all(q[:,j]<=a['high'].cpu().numpy()+1e-10)
        assert np.max(1.025*np.abs(q[:,j,5])+np.abs(q[:,j,6])) <= follower.coupling_sum_limit+1e-10
    np.savez_compressed(out/'ik_trace_200hz.npz',time_s=frame_times,q=q,velocity=velocity,acceleration=acceleration,
        palm_actual_m=actual,palm_target_m=target,palm_actual_rotation=actual_rot,palm_target_rotation=target_rot,
        requested_displacement_m=requested,sides=np.array(sides))

    if report.get('benchmark_only'):
        report['verification']={'finite_difference_velocity_acceleration':True,'joint_position_velocity_acceleration_limits':True,'j6_j7_coupling_limits':True}
        (out/'benchmark_report.json').write_text(json.dumps(report,indent=2)+'\n')
        print('BENCHMARK COMPLETE',out,flush=True)
        return
    fonts={s:ImageFont.truetype(str(font_path),s) for s in (20,23,26,32,38)}
    colors=[(239,68,68),(35,190,96),(70,140,255)]
    pixel=(1920,1080)
    sampled=np.arange(0,n,stride,dtype=int)
    np.save(out/'video_source_times.npy',frame_times[sampled])

    def dashed(draw,points,color,width=2):
        for start in range(0,len(points)-1,6):
            if len(points[start:start+4])>1: draw.line(points[start:start+4],fill=color,width=width)

    def graph(draw,at,x,y,width=425,height=155):
        # Three same-scale plots, exact requested vs executed world displacement.
        for axis in range(3):
            xx=x+axis*(width+25)
            draw.text((xx,y-28),'XYZ'[axis]+' 位移 / mm',font=fonts[20],fill=colors[axis])
            draw.rectangle((xx,y,xx+width,y+height),fill=(24,37,52),outline=(70,90,110))
            for v in (-20,0,20):
                yy=y+height/2-v/30*(height/2)
                draw.line((xx,yy,xx+width,yy),fill=(55,70,85),width=1)
            ids=np.arange(0,at+1,2)
            points_t=[(xx+frame_times[k]/duration*width,y+height/2-(target[k,0,axis]-p0[axis])*1000/30*height/2) for k in ids]
            points_a=[(xx+frame_times[k]/duration*width,y+height/2-(actual[k,0,axis]-p0[axis])*1000/30*height/2) for k in ids]
            if len(ids)>1:
                draw.line(points_a,fill=colors[axis],width=3); dashed(draw,points_t,(255,125,255),2)
            draw.line((xx+frame_times[at]/duration*width,y,xx+frame_times[at]/duration*width,y+height),fill=(225,235,245),width=1)
            draw.text((xx,y+height+3),'0 s',font=fonts[20],fill=(175,192,210))
            draw.text((xx+width-50,y+height+3),'14 s',font=fonts[20],fill=(175,192,210))

    encoder=imageio_ffmpeg.get_ffmpeg_exe()
    command=[encoder,'-y','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r',str(fps),'-i','-',
             '-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'wuji_xyz_ik.mp4')]
    with (out/'ffmpeg.log').open('w') as log, mujoco.Renderer(model,width=1200,height=820,max_geom=5000) as renderer:
        report['graphics']=current_gl_info()
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=log,stderr=log)
        try:
            for fi,k in enumerate(sampled):
                t=frame_times[k]; _,axis,phase=schedule(t)
                # Re-render the recorded 200 Hz q, never run a different IK path.
                if hasattr(follower,'load_recorded_q'):
                    follower.load_recorded_q(q[k:k+1])
                else:
                    for j,s in enumerate(sides):follower.sides[s]['q'].copy_(torch.as_tensor(q[k,j][None],device=follower.device))
                tar=(target[k,0],target_rot[k,0])
                wide=render(renderer,None,'机械臂全景 · 固定机位',False,target=tar,elapsed=t,fixed_camera=True)
                detail=render(renderer,None,'掌根 / 法兰 / 四孔连接件 · 固定机位',True,target=tar,elapsed=t,fixed_camera=True)
                for j,s in enumerate(sides):
                    np.testing.assert_allclose(data.xpos[follower.wrists[s]['body_id']],actual[k,j],atol=1e-8,rtol=0)
                frame=Image.new('RGB',pixel,(13,23,36)); draw=ImageDraw.Draw(frame)
                frame.paste(wide.resize((960,656),Image.Resampling.LANCZOS),(0,100))
                frame.paste(detail.resize((960,656),Image.Resampling.LANCZOS),(960,100))
                draw.text((24,8),'wuji_cartesian_v1  |  '+phase+' 跟随',font=fonts[38],fill='white')
                draw.text((24,59),'W 世界系  ·  F 法兰  ·  P 实际掌根  ·  T 目标掌根（紫色）  |  1倍速 / 200 Hz IK / 25 fps',font=fonts[23],fill=(195,210,228))
                draw.text((1460,18),f't = {t:05.2f} / {duration:.0f} s',font=fonts[32],fill='white')
                graph(draw,k,25,825)
                draw.text((1390,785),'右手掌根：目标 → 实际',font=fonts[26],fill='white')
                for j in range(3):
                    tv=(target[k,0,j]-p0[j])*1000; av=(actual[k,0,j]-p0[j])*1000
                    draw.text((1390,830+j*39),f'{"XYZ"[j]}   {tv:+7.2f}  →  {av:+7.2f} mm',font=fonts[23],fill=colors[j])
                draw.text((1390,958),f'位置误差  {error[k]:.3f} mm',font=fonts[26],fill=(255,125,255))
                draw.text((24,1010),'紫虚线：目标   彩色实线：实际   |   左手固定作参照；右手依次沿世界 XYZ 运动；掌朝向固定',font=fonts[23],fill=(220,230,240))
                draw.text((24,1047),(f'CPU 编译 IK + 场景刷新；{count*dt:g}秒计时见报告。手指保持初态，渲染独立进行。' if paced else '生产 IK 组件演示；手指保持初始姿态，无策略/琴键物理回放。Z 段仅抬升与返回。'),font=fonts[20],fill=(160,180,200))
                process.stdin.write(frame.tobytes())
                if fi in (0,40,140,240,325):frame.save(out/f'preview_{fi:03d}.png')
                if fi%50==0:print(f'Render {fi}/{len(sampled)} frames',flush=True)
            process.stdin.close()
            if process.wait()!=0:raise RuntimeError('ffmpeg failed; inspect ffmpeg.log')
        except BaseException:
            if process.stdin and not process.stdin.closed:process.stdin.close()
            process.wait()
            raise
    report.update(scope=__doc__,duration_s=duration,playback_speed=1,fps=fps,video_frames=len(sampled),
        control_hz=1/bundle.manifest['control_dt'],ik_hz=1/dt,physics_steps=0,checkpoint=None,midi=None,
        request_delta_m=None,trajectory='20 Hz sampled quintic segments; X +/-20mm, Y +/-20mm, Z +20mm then +10mm, each returns to zero',
        phases=[{'axis':s,'start_s':1+4*i,'end_s':5+4*i} for i,s in enumerate('XYZ')],
        max_position_error_mm=float(error.max()),final_position_error_mm=float(error[-1]),
        per_axis_max_error_mm={s:float(error[(frame_times>=1+4*i)&(frame_times<5+4*i)].max()) for i,s in enumerate('XYZ')},
        verification={'finite_difference_velocity_acceleration':True,'joint_position_velocity_acceleration_limits':True,
                      'j6_j7_coupling_limits':True,'rendered_palm_matches_recorded_fk':True},
        render_wall_seconds=time.perf_counter()-started-report['trace_compute_wall_seconds'])
    # Offline scientific plot of the very same trace (not MIDI/key activity).
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(4,1,figsize=(12,9),sharex=True)
    for i in range(3):
        axs[i].plot(frame_times,(target[:,0,i]-p0[i])*1000,'--',color='magenta',label='Requested palm')
        axs[i].plot(frame_times,(actual[:,0,i]-p0[i])*1000,color=np.array(colors[i])/255,label='Executed palm')
        axs[i].set_ylabel('XYZ'[i]+' (mm)');axs[i].grid(alpha=.25)
    axs[0].legend(loc='upper right')
    axs[3].plot(frame_times,error,color='purple');axs[3].set_ylabel('Error (mm)');axs[3].set_xlabel('Simulation time (s), 1x video')
    axs[3].grid(alpha=.25)
    fig.suptitle('Same-trajectory requested / executed palm positions | 200 Hz IK component demo')
    fig.tight_layout();fig.savefig(out/'xyz_tracking.png',dpi=170);plt.close(fig)
    report['source_hashes']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        [root/'tsensei_v3_2_pianist/mjlab_env/arm_scene.py',root/'tsensei_v3_2_pianist/mjlab_env/arm_follow.py',root/'tsensei_v3_2_pianist/mjlab_env/arm_native_scene.py',root/'tsensei_v3_2_pianist/mjlab_env/arm_realtime.py',root/'configs/actions/wuji_cartesian_v1.json']}
    (out/'video_report.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n'
        for p in sorted(out.iterdir()) if p.is_file() and p.name!='SHA256SUMS'))
    print('VIDEO COMPLETE',out,flush=True)
