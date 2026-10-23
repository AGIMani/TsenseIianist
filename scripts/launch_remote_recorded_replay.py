#!/usr/bin/env python3
"""Local audio relay; all MuJoCo model loading and software rendering run on volc-a32."""
import json,subprocess,shlex,shutil
from pathlib import Path

def main():
    root=Path(__file__).resolve().parent
    if not shutil.which('ffplay'):raise RuntimeError('Local FFmpeg/ffplay is required for sound')
    settings=json.loads((root/'remote_launch.json').read_text())
    command='ulimit -v 4194304; exec '+shlex.join(['env','LIBGL_ALWAYS_SOFTWARE=1','MUJOCO_GL=glfw',settings['python'],settings['replay'],'--external-audio'])
    process=subprocess.Popen(['ssh','-Y',settings['host'],'bash','-lc',shlex.quote(command)],stdout=subprocess.PIPE,text=True)
    audio=None
    def stop():
        nonlocal audio
        if audio:
            if audio.poll() is None:audio.terminate()
            audio.wait(timeout=5);audio=None
    try:
        for line in process.stdout:
            if not line.startswith('REPLAY_AUDIO '):print(line,end='');continue
            event=json.loads(line[len('REPLAY_AUDIO '):]);stop()
            if event['state']=='play':
                audio=subprocess.Popen(['ffplay','-nodisp','-autoexit','-loglevel','error','-ss',str(event['at']),'-af','atempo='+str(event['speed']),str(root/'played.wav')],stdin=subprocess.DEVNULL)
        if process.wait():raise RuntimeError('Remote viewer failed; see SSH/display diagnostics above')
    finally:
        stop()
        if process.poll() is None:process.terminate()

if __name__=='__main__':main()
