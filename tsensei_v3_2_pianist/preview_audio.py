"""Actual-key audio: substep transitions -> MIDI -> sampled acoustic piano.

No score or goal data is used. Velocity=127 matches the existing CPU MidiModule;
TODO(V32-AUDIO-01): physical key-speed/loudness and acoustic response calibration.
"""
import ctypes as C
import ctypes.util
import hashlib
from pathlib import Path
import struct
import wave

import numpy as np
from .key_activation import key_active

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT/'training_mjlab/audio_runtime/root'


def soundfont_path(requested=None):
    choices = [Path(requested)] if requested else [RUNTIME/'usr/share/sounds/sf2/TimGM6mb.sf2',
                                                   Path('/usr/share/sounds/sf2/TimGM6mb.sf2')]
    for path in choices:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError('Piano SoundFont missing. Run preview/scripts/setup_piano_audio.py or pass --soundfont.')


def events_from_states(key_qpos, limits, sustain, dt, threshold, velocity=127):
    key_qpos = np.asarray(key_qpos)
    sustain = np.asarray(sustain)
    if key_qpos.ndim != 2 or key_qpos.shape[1] != 88 or sustain.shape != (len(key_qpos),):
        raise ValueError('Expected [substeps,88] key positions and [substeps] sustain')
    if not np.isfinite(key_qpos).all() or not np.isfinite(sustain).all() or dt <= 0 or not 1 <= velocity <= 127:
        raise ValueError('Invalid piano recording')
    active = key_active(key_qpos, limits, threshold)
    previous = np.zeros(88, dtype=bool)
    pedal = False
    events = []
    for i, keys in enumerate(active):
        t = (i+1)*dt
        new_pedal = bool(sustain[i] >= .5)
        if new_pedal != pedal:
            events.append(dict(time=t, type='sustain', value=127 if new_pedal else 0))
            pedal = new_pedal
        for key in np.flatnonzero(previous & ~keys):
            events.append(dict(time=t, type='note_off', note=int(key)+21))
        for key in np.flatnonzero(keys & ~previous):
            events.append(dict(time=t, type='note_on', note=int(key)+21, velocity=velocity))
        previous = keys
    # Explicit presentation tail: release remaining voices at episode end.
    end = len(active)*dt
    for key in np.flatnonzero(previous):
        events.append(dict(time=end, type='note_off', note=int(key)+21, end_of_replay=True))
    if pedal:
        events.append(dict(time=end, type='sustain', value=0, end_of_replay=True))
    return events


def write_midi(path, events):
    def vlq(value):
        out = [value & 127]
        while value >> 7:
            value >>= 7
            out.insert(0, (value & 127) | 128)
        return bytes(out)
    track = bytearray(b'\x00\xff\x51\x03\x07\xa1\x20\x00\xc0\x00')
    last = 0
    for event in events:
        tick = round(event['time']*2000)  # PPQ1000, tempo500000 -> 0.5 ms.
        track += vlq(tick-last)
        last = tick
        if event['type'] == 'note_on':
            track += bytes((0x90, event['note'], event['velocity']))
        elif event['type'] == 'note_off':
            track += bytes((0x80, event['note'], 0))
        else:
            track += bytes((0xB0, 64, event['value']))
    track += b'\x00\xff\x2f\x00'
    Path(path).write_bytes(b'MThd'+struct.pack('>IHHH',6,0,1,1000)+b'MTrk'+struct.pack('>I',len(track))+track)


class PianoSynth:
    def __init__(self, soundfont=None, sample_rate=48000):
        self.soundfont = soundfont_path(soundfont)
        libraries = RUNTIME/'usr/lib/x86_64-linux-gnu'
        if (libraries/'libinstpatch-1.0.so.2').exists():
            self.instpatch = C.CDLL(str(libraries/'libinstpatch-1.0.so.2'), mode=C.RTLD_GLOBAL)
        library = str(libraries/'libfluidsynth.so.3') if (libraries/'libfluidsynth.so.3').exists() else C.util.find_library('fluidsynth')
        if not library:
            raise RuntimeError('libfluidsynth missing; run preview/scripts/setup_piano_audio.py')
        self.lib = C.CDLL(library)
        signatures = {
            'new_fluid_settings': (C.c_void_p, []),
            'fluid_settings_setnum': (C.c_int,[C.c_void_p,C.c_char_p,C.c_double]),
            'fluid_settings_setint': (C.c_int,[C.c_void_p,C.c_char_p,C.c_int]),
            'new_fluid_synth': (C.c_void_p,[C.c_void_p]),
            'fluid_synth_sfload': (C.c_int,[C.c_void_p,C.c_char_p,C.c_int]),
            'fluid_synth_program_select': (C.c_int,[C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_int]),
            'fluid_synth_noteon': (C.c_int,[C.c_void_p,C.c_int,C.c_int,C.c_int]),
            'fluid_synth_noteoff': (C.c_int,[C.c_void_p,C.c_int,C.c_int]),
            'fluid_synth_cc': (C.c_int,[C.c_void_p,C.c_int,C.c_int,C.c_int]),
            'fluid_synth_write_float': (C.c_int,[C.c_void_p,C.c_int,C.c_void_p,C.c_int,C.c_int,C.c_void_p,C.c_int,C.c_int]),
            'delete_fluid_synth': (None,[C.c_void_p]), 'delete_fluid_settings': (None,[C.c_void_p])}
        for name, (result, arguments) in signatures.items():
            fn = getattr(self.lib,name); fn.restype=result; fn.argtypes=arguments
        self.settings = self.lib.new_fluid_settings()
        self.lib.fluid_settings_setnum(self.settings,b'synth.sample-rate',sample_rate)
        self.lib.fluid_settings_setnum(self.settings,b'synth.gain',.5)
        self.lib.fluid_settings_setint(self.settings,b'synth.reverb.active',0)
        self.lib.fluid_settings_setint(self.settings,b'synth.chorus.active',0)
        self.synth = self.lib.new_fluid_synth(self.settings)
        self.sample_rate = sample_rate
        sfid = self.lib.fluid_synth_sfload(self.synth,str(self.soundfont).encode(),1)
        if sfid < 0 or self.lib.fluid_synth_program_select(self.synth,0,sfid,0,0) < 0:
            self.close()
            raise RuntimeError('Cannot load acoustic piano program from SoundFont')

    def close(self):
        if self.synth:
            self.lib.delete_fluid_synth(self.synth); self.synth=None
        if self.settings:
            self.lib.delete_fluid_settings(self.settings); self.settings=None

    def render(self, events, duration, path):
        samples = np.zeros((round(duration*self.sample_rate),2), dtype=np.float32)
        cursor = 0
        def fill(end):
            nonlocal cursor
            n = end-cursor
            if n > 0:
                ptr = C.c_void_p(samples[cursor:].ctypes.data)
                code = self.lib.fluid_synth_write_float(self.synth,n,ptr,0,2,ptr,1,2)
                if code < 0:
                    raise RuntimeError('FluidSynth audio rendering failed')
                cursor = end
        for event in events:
            end = round(event['time']*self.sample_rate)
            if not cursor <= end <= len(samples):
                raise ValueError('Audio events must be sorted and within video duration')
            fill(end)
            if event['type'] == 'note_on':
                self.lib.fluid_synth_noteon(self.synth,0,event['note'],event['velocity'])
            elif event['type'] == 'note_off':
                self.lib.fluid_synth_noteoff(self.synth,0,event['note'])
            else:
                self.lib.fluid_synth_cc(self.synth,0,64,event['value'])
        fill(len(samples))
        if not np.isfinite(samples).all():
            raise FloatingPointError('Nonfinite audio')
        peak = float(np.abs(samples).max())
        gain = min(1.,.95/max(peak,1e-12))
        samples *= gain
        fade = min(round(.05*self.sample_rate),len(samples))
        samples[-fade:] *= np.linspace(1,0,fade)[:,None]
        with wave.open(str(path),'wb') as stream:
            stream.setnchannels(2);stream.setsampwidth(2);stream.setframerate(self.sample_rate)
            stream.writeframes((samples*32767).astype('<i2').tobytes())
        return dict(engine='FluidSynth', soundfont=str(self.soundfont),
                    soundfont_sha256=hashlib.sha256(self.soundfont.read_bytes()).hexdigest(),
                    sample_rate=self.sample_rate, channels=2, peak_before_gain=peak, gain=gain,
                    rms=float(np.sqrt(np.mean(samples**2))), duration=duration,
                    source='actual key activation edges and actual sustain at every physics substep',
                    note_velocity=127, velocity_calibrated=False)
