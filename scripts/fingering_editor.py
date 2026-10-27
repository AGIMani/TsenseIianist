#!/usr/bin/env python3
"""Local score/timing editor with remote CPU kinematic preview."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import queue
import shlex
import subprocess
import threading
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np

from fingering_editor_data import FINGERS, Score, note_name
from fingering_editor_view import camera_basis, split_key_faces, pale_color, diagonal_hatching
from fingering_editor_config import connection_config

ROOT = Path(__file__).resolve().parents[1]
COLORS = ("#bf65d5", "#dd655f", "#39adad", "#668ddd", "#b59c35",
          "#ae6ac9", "#d77f53", "#37a388", "#6685c6", "#a4a13c")


class RemotePreview:
    def __init__(self, config_path=None, *, backend='remote'):
        self.backend = backend
        self.config_path = Path(config_path) if config_path else connection_config(ROOT, 'remote')
        self.config = {}
        # One replaceable pending request: rapid timeline scrubbing cannot
        # accumulate unbounded Python objects or pipe input.
        self.requests, self.responses = queue.Queue(maxsize=1), queue.Queue(maxsize=2)
        self.process = None
        self.limited = None
        self.closed = False
        self.cancelled = threading.Event()
        self.stderr_offset = 0
        self.log_path = ROOT / "logs/fingering_editor" / f"session_{time.time_ns()}.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        try:
            if self.backend == 'local':
                self.run_local()
                return
            if not self.config_path.is_file():
                self.responses.put(dict(type='status', message='正在恢复远端三维配置并核验源码，请稍候…'))
                with self.log_path.open('a') as log:
                    result = subprocess.run(
                        [sys.executable, str(ROOT/'scripts/prepare_fingering_editor.py'),
                         '--output', str(self.config_path)],
                        stdout=log, stderr=log, text=True, timeout=240)
                if result.returncode:
                    raise RuntimeError(f'远端三维配置初始化失败（exit={result.returncode}）')
            if self.closed:
                return
            self.config = json.loads(self.config_path.read_text())
            c = self.config
            remote = c["remote_root"]
            command = (
                f"cd {shlex.quote(remote)} && "
                f"test \"$(sha256sum scripts/fingering_pose_worker.py | cut -d ' ' -f 1)\" = {shlex.quote(c['worker_sha256'])} && "
                "export MUJOCO_GL=disable CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 && "
                f"exec {shlex.quote(c['python'])} scripts/fingering_pose_worker.py --cache "
                f"{shlex.quote(remote + '/editor_cache')}"
            )
            with self.log_path.open("a") as log:
                log.write(json.dumps(dict(command=command, host=c["host"], source_sha256=c["source_sha256"])) + "\n")
                log.flush()
                ready = self.connect_remote(command, log)
                if ready is None:
                    return
                log.write(json.dumps({k: v for k, v in ready.items() if k not in ("meshes", "frame")}) + "\n")
                log.flush()
                self.responses.put(ready)
                while not self.closed:
                    req = self.requests.get()
                    if req is None:
                        break
                    while not self.requests.empty():
                        req = self.requests.get_nowait()
                    if req is None:
                        break
                    self.process.stdin.write(json.dumps(req) + "\n")
                    self.process.stdin.flush()
                    self.responses.put(self.read())
        except Exception as exc:
            if not self.closed:
                self.responses.put(dict(type="error", error=f"{exc}；日志：{self.log_path}"))
        finally:
            if self.backend == 'remote' and self.process:
                self.dispose_remote()
                with self.log_path.open('a') as log:
                    log.write(f'\nexit_code={self.process.returncode}\n')

    def connect_remote(self, command, log):
        # Retry only transport failures before READY. Replaying requests after
        # an established session fails could display an obsolete pose.
        transient = ('connection reset', 'connection timed out', 'connection refused',
                     'connection closed', 'no route to host', 'network is unreachable',
                     'temporary failure in name resolution')
        permanent = ('permission denied', 'host key verification failed',
                     'remote host identification has changed')
        for attempt in range(1, 4):
            if self.closed:
                return None
            self.stderr_offset = log.tell()
            self.process = subprocess.Popen(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                 "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2",
                 self.config["host"], command],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                text=True, bufsize=1)
            log.write(f"\nlocal_ssh_pid={self.process.pid} attempt={attempt}/3\n")
            log.flush()
            try:
                return self.read()
            except RuntimeError as exc:
                self.dispose_remote()
                if (self.process.returncode != 255 or attempt == 3 or
                        any(reason in str(exc).lower() for reason in permanent) or
                        not any(reason in str(exc).lower() for reason in transient)):
                    raise
                self.responses.put(dict(type='status', message=
                    f'远端 SSH 连接暂时中断，{attempt} 秒后重试（{attempt+1}/3）…'))
                if self.cancelled.wait(attempt):
                    return None

    def dispose_remote(self):
        try:
            self.process.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            self.process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process.stdout.close()

    def submit(self, request):
        while True:
            try:
                self.requests.put_nowait(request)
                return
            except queue.Full:
                try:
                    self.requests.get_nowait()
                except queue.Empty:
                    pass

    def run_local(self):
        from fingering_local_runner import LimitedWorker, prepare_runtime
        import hashlib
        path = connection_config(ROOT, 'local')
        config = json.loads(path.read_text()) if path.is_file() else {}
        current = all(config.get(key) == hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest()
                      for key, name in (('worker_sha256', 'fingering_pose_worker.py'),
                                        ('helper_sha256', 'fingering_editor_limits.py')))
        if not current:
            prepare_runtime(ROOT)
        if self.closed:
            return
        self.limited = LimitedWorker(ROOT)
        self.log_path = self.limited.log_path
        self.process = self.limited.process
        try:
            ready = self.read()
            self.limited.initialized = ready.get('type') == 'ready'
            self.responses.put(ready)
            while not self.closed:
                request = self.requests.get()
                if request is None:
                    break
                self.process.stdin.write(json.dumps(request) + '\n')
                self.process.stdin.flush()
                self.responses.put(self.read())
        finally:
            self.limited.close()

    def read(self):
        line = self.process.stdout.readline()
        if not line:
            try:
                code = self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                code = self.process.poll()
            reason = self.limited.stats.get('reason') if self.limited else None
            if self.backend == 'remote':
                # Bound reads even when a remote traceback is very large; keep
                # the current attempt's stderr out of an earlier session error.
                with self.log_path.open('rb') as diagnostic:
                    diagnostic.seek(0, 2)
                    diagnostic.seek(max(self.stderr_offset, diagnostic.tell()-4096))
                    lines = diagnostic.read(4096).decode('utf-8', errors='replace').splitlines()
                reason = '\n'.join(line for line in lines
                                   if line.strip() and not line.startswith('local_ssh_pid='))[-1200:]
            prefix = '远端 SSH 连接失败' if self.backend == 'remote' and code == 255 else '预览求解进程退出'
            raise RuntimeError(f"{prefix}（exit={code}）" + (f'：{reason}' if reason else ''))
        return json.loads(line)

    def close(self):
        self.closed = True
        self.cancelled.set()
        self.submit(None)
        if self.process and self.process.poll() is None:
            if self.limited:
                self.limited.stop_scope('Editor window closed')
            else:
                self.process.terminate()


class ModelCanvas(tk.Canvas):
    """Software triangle drawing only; no MuJoCo, physics, IK, or OpenGL locally."""
    def __init__(self, parent):
        super().__init__(parent, background="#eef1f5", highlightthickness=0)
        self.meshes = []
        self.frame = None
        self.previous_assignments = []
        self.show_previous = True
        self.next_assignments = []
        self.show_next = True
        self.azimuth, self.elevation, self.zoom = 0., .80, 2.2
        self.offset = np.zeros(2)
        self.view_name = '演奏者视角：左低音，右高音'
        self.bind("<Configure>", lambda e: self.draw())
        self.bind("<ButtonPress-1>", self.press)
        self.bind("<B1-Motion>", self.rotate)
        self.bind("<ButtonPress-3>", self.press)
        self.bind("<B3-Motion>", self.pan)
        self.bind("<Button-4>", lambda e: self.magnify(1.15))
        self.bind("<Button-5>", lambda e: self.magnify(1/1.15))
        self.bind("<MouseWheel>", lambda e: self.magnify(1.15 if e.delta > 0 else 1/1.15))

    def press(self, e):
        self.drag = (e.x, e.y)

    def rotate(self, e):
        dx, dy = e.x-self.drag[0], e.y-self.drag[1]
        self.azimuth += dx*.009
        self.elevation = np.clip(self.elevation+dy*.009, .12, 1.5)
        self.view_name = '自由视角（左右手标识跟随真实模型）'
        self.drag = (e.x, e.y)
        self.draw()

    def pan(self, e):
        self.offset += [e.x-self.drag[0], e.y-self.drag[1]]
        self.drag = (e.x, e.y)
        self.draw()

    def magnify(self, factor):
        self.zoom = min(6., max(.4, self.zoom*factor))
        self.draw()

    def reset(self):
        self.azimuth, self.elevation, self.zoom = 0., .80, 2.2
        self.offset[:] = 0
        self.view_name = '演奏者视角：左低音，右高音'
        self.draw()

    def set_view(self, view):
        self.reset()
        if view == 'opposite':
            self.azimuth = math.pi
            self.view_name = '对面视角：右手在画面左侧'
        elif view == 'top':
            self.elevation = math.pi/2
            self.view_name = '俯视：左低音，右高音'
        self.draw()

    def load(self, meshes, frame):
        self.meshes = [dict(m, vertices=np.array(m["vertices"]), faces=np.array(m["faces"])) for m in meshes]
        self.frame = frame
        self.draw()

    def project(self, points):
        p = np.asarray(points)-np.array([.07, 0., .05])
        mat = camera_basis(self.azimuth, self.elevation)
        out = p @ mat.T
        out[:, :2] *= self.winfo_width() / 1.30*self.zoom
        out[:, :2] += np.array([self.winfo_width()/2, self.winfo_height()*.48])+self.offset
        return out

    def draw(self):
        self.delete("all")
        if not self.meshes or not self.frame:
            self.create_text(self.winfo_width()/2, self.winfo_height()/2, text="三维模型等待受限 CPU 进程加载…", fill="#52627a")
            return
        active_f = {a["finger"] for a in self.frame["active"]}
        key_fingers = {}
        for assignment in self.frame['active']:
            key_fingers.setdefault(assignment['pitch'], set()).add(assignment['finger'])
        next_keys = {}
        if self.show_next:
            for assignment in self.next_assignments:
                next_keys.setdefault(assignment['pitch'], set()).add(assignment['finger'])
        previous_keys = {}
        if self.show_previous:
            for assignment in self.previous_assignments:
                if assignment['pitch'] not in key_fingers and assignment['pitch'] not in next_keys:
                    previous_keys.setdefault(assignment['pitch'], set()).add(assignment['finger'])
        triangles = []
        for mesh, transform in zip(self.meshes, self.frame["transforms"]):
            t = np.array(transform)
            world = mesh["vertices"] @ t[3:].reshape(3, 3).T + t[:3]
            projected = self.project(world)
            if mesh["pitch"] is not None:
                palette = [COLORS[f] for f in sorted(key_fingers.get(mesh['pitch'], ()))]
                if not palette:
                    palette = [pale_color(COLORS[f]) for f in sorted(next_keys.get(mesh['pitch'], ()))]
                if not palette:
                    palette = [pale_color(COLORS[f]) for f in sorted(previous_keys.get(mesh['pitch'], ()))]
                color = palette[0] if palette else "#343b4b" if mesh["black"] else "#ffffff"
            elif mesh["finger"] in active_f:
                color = COLORS[mesh["finger"]]
            else:
                color = "#b6c4d9" if mesh["side"] == 0 else "#d7c1ce"
            if mesh['pitch'] is not None and len(palette) > 1:
                pieces = ((polygon @ t[3:].reshape(3, 3).T + t[:3], hue)
                          for polygon, hue in split_key_faces(mesh['vertices'], mesh['faces'], palette))
                surfaces = ((xyz, self.project(xyz), hue) for xyz, hue in pieces)
            else:
                surfaces = ((world[face], projected[face], color) for face in mesh['faces'])
            for xyz, pts, hue in surfaces:
                if pts[:, 0].max() < 0 or pts[:, 0].min() > self.winfo_width() or pts[:, 1].max() < 0 or pts[:, 1].min() > self.winfo_height():
                    continue
                rgb = np.array([int(hue[i:i+2], 16) for i in (1, 3, 5)])
                normal = np.cross(xyz[1]-xyz[0], xyz[2]-xyz[0])
                light = .70 + .30*abs(float(normal @ np.array([.3, -.3, .9]))) / max(float(np.linalg.norm(normal)), 1e-12)
                tint = "#" + "".join(f"{int(v):02x}" for v in np.clip(rgb*light, 0, 255))
                stripe_colors = [(pale_color(COLORS[f]) if mesh['pitch'] in key_fingers else '#ffffff')
                                 for f in sorted(next_keys.get(mesh['pitch'], ()))]
                triangles.append((pts[:, 2].mean(), pts[:, :2].reshape(-1).tolist(), tint, stripe_colors))
        for _, points, color, stripes in sorted(triangles, key=lambda v: v[0]):
            self.create_polygon(points, fill=color, outline=color)
            if stripes:
                for i, (a, b) in enumerate(diagonal_hatching(np.asarray(points).reshape(-1, 2))):
                    self.create_line(*a, *b, fill=stripes[i % len(stripes)], width=2, tags='next_hatching')
        tips = self.project(self.frame["tips"])
        for f, p in enumerate(tips):
            self.create_oval(p[0]-8, p[1]-8, p[0]+8, p[1]+8, fill=COLORS[f], outline="white")
            self.create_text(p[0], p[1], text=str(f), fill="white", font=("", 9, "bold"))
        for target in self.frame["targets"]:
            tip = tips[target["finger"]]
            point = self.project([target["position"]])[0]
            self.create_line(*tip[:2], *point[:2], fill=COLORS[target['finger']], dash=(3, 2), width=2)
        for mesh, transform in zip(self.meshes, self.frame['transforms']):
            upcoming = next_keys.get(mesh['pitch'])
            fingers = upcoming or previous_keys.get(mesh['pitch'])
            if not fingers:
                continue
            # Put the previous-ID label toward the far end of the key, away
            # from the current hand and its fingertip number overlays.
            x = .95 * mesh['vertices'][:, 0].min() + .05 * mesh['vertices'][:, 0].max()
            local_top = np.array([x, 0., mesh['vertices'][:, 2].max() + .002])
            t = np.asarray(transform)
            point = self.project([local_top @ t[3:].reshape(3, 3).T + t[:3]])[0]
            self.create_text(*point[:2], text=('下' if upcoming else '上') + '/'.join(str(f) for f in sorted(fingers)),
                             fill=COLORS[min(fingers)], font=('', 9, 'bold'),
                             tags='next_fingering' if upcoming else 'previous_fingering')
        for first, label in ((0, '右手 · 0–4'), (5, '左手 · 5–9')):
            position = np.mean(np.asarray(self.frame['tips'])[first:first+5], axis=0)
            position[2] += .075
            point = self.project([position])[0]
            self.create_text(*point[:2], text=label, fill='#253b56', font=('', 12, 'bold'))
        legend = '原色：当前指法 · 浅色：上一组' if self.show_previous else '目标琴键与分配手指同色'
        if self.show_next:
            legend += '\n浅色斜线：下一组（下＋编号）'
        self.create_text(12, 12, anchor="nw", text=f"{self.view_name}\n左拖旋转 · 滚轮缩放 · 右拖平移\n{legend}", fill="#45566e")


class Editor:
    def __init__(self, root, path=None, remote_config=None, *, backend='remote'):
        self.root, self.score = root, None
        self.selected = None
        self.playing, self.ready, self.request_id = False, False, 0
        self.pending_time = None
        self.pending_previous = []
        self.pending_next = []
        self.current_time, self.pixels_per_second = 0., 90.
        self.remote = None
        self.backend = backend
        root.title("A3.2 指法与时间编辑器")
        root.geometry("1530x940")
        root.minsize(1100, 720)
        root.protocol("WM_DELETE_WINDOW", self.close)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", font=("Noto Sans CJK SC", 10))
        style.configure("Treeview", rowheight=26)
        outer = ttk.Frame(root, padding=12)
        outer.pack(fill="both", expand=True)
        bar = ttk.Frame(outer)
        bar.pack(fill="x")
        for label, action in (("导入 proto", self.open), ("另存 proto", self.save),
                              ("撤销", self.undo), ("重做", self.redo)):
            ttk.Button(bar, text=label, command=action).pack(side="left", padx=(0, 6))
        self.filename = ttk.Label(bar, text="导入 NoteSequence 二进制 proto")
        self.filename.pack(side="left", padx=12)
        main = ttk.Panedwindow(outer, orient="horizontal")
        main.pack(fill="both", expand=True, pady=10)
        left, right = ttk.Frame(main), ttk.Frame(main)
        main.add(left, weight=5)
        main.add(right, weight=6)
        toolbar = ttk.Frame(left)
        toolbar.pack(fill="x")
        self.play_button = ttk.Button(toolbar, text="▶ 姿态预览", command=self.play)
        self.play_button.pack(side="left")
        ttk.Button(toolbar, text="回到开头", command=lambda: self.seek(0)).pack(side="left", padx=4)
        ttk.Label(toolbar, text="时间 / 秒").pack(side="left")
        self.time_text = tk.StringVar(value="0")
        entry = ttk.Entry(toolbar, textvariable=self.time_text, width=9)
        entry.pack(side="left", padx=4)
        entry.bind("<Return>", lambda e: self.seek_entry())
        ttk.Button(toolbar, text="定位", command=self.seek_entry).pack(side="left")
        ttk.Button(toolbar, text="＋", width=3, command=lambda: self.zoom(1.4)).pack(side="right")
        ttk.Button(toolbar, text="－", width=3, command=lambda: self.zoom(1/1.4)).pack(side="right")
        self.time_scale = ttk.Scale(left, from_=0, to=1, command=self.scale_seek)
        self.time_scale.pack(fill="x", pady=6)
        self.roll = tk.Canvas(left, height=300, background="#f8fafc", highlightthickness=1, highlightbackground="#cbd5e1")
        self.roll.pack(fill="both", expand=True)
        scroll = ttk.Scrollbar(left, orient="horizontal", command=self.roll.xview)
        scroll.pack(fill="x")
        self.roll.configure(xscrollcommand=scroll.set)
        self.roll.bind("<Button-1>", self.click_roll)
        ttk.Label(left, text="点击音符选择；点击空白定位时间。红边表示同一手指的时间重叠。").pack(anchor="w", pady=4)
        table_frame = ttk.Frame(left)
        table_frame.pack(fill="both", expand=True)
        self.table = ttk.Treeview(table_frame, columns=("id", "pitch", "start", "end", "finger"), show="headings", height=8, selectmode="browse")
        for col, title, width in (("id", "序号", 48), ("pitch", "音高", 60), ("start", "起音 / s", 100),
                                  ("end", "松键 / s", 100), ("finger", "手指 / ID", 150)):
            self.table.heading(col, text=title)
            self.table.column(col, width=width, anchor="w")
        self.table.pack(side="left", fill="both", expand=True)
        sc = ttk.Scrollbar(table_frame, command=self.table.yview)
        sc.pack(side="right", fill="y")
        self.table.configure(yscrollcommand=sc.set)
        self.table.tag_configure("conflict", foreground="#c14327")
        self.table.bind("<<TreeviewSelect>>", self.select_table)
        edit = ttk.LabelFrame(left, text="选中音符", padding=8)
        edit.pack(fill="x", pady=(8, 0))
        self.selection_label = ttk.Label(edit, text="尚未选择")
        self.selection_label.grid(row=0, column=0, columnspan=5, sticky="w")
        self.start_var, self.end_var = tk.StringVar(), tk.StringVar()
        for col, label, var in ((0, "起音 / s", self.start_var), (2, "松键 / s", self.end_var)):
            ttk.Label(edit, text=label).grid(row=1, column=col, sticky="w")
            ttk.Entry(edit, textvariable=var, width=15).grid(row=1, column=col+1, padx=5)
        ttk.Button(edit, text="应用时间", command=self.edit_time).grid(row=1, column=4, padx=4)
        for f, name in enumerate(FINGERS):
            ttk.Button(edit, text=name, command=lambda f=f: self.assign(f)).grid(row=2+f//5, column=f%5, padx=2, pady=(6, 0), sticky="ew")
        ttk.Label(right, text="A3.2 三维按键姿态", font=("Noto Sans CJK SC", 14, "bold")).pack(anchor="w")
        location = '本地 CPU 求解（1.5 GiB 上限）' if backend == 'local' else '远端 CPU 求解'
        ttk.Label(right, text=f"{location} · 本地渲染（512 MiB 上限） · 无声预览").pack(anchor="w", pady=4)
        self.canvas = ModelCanvas(right)
        self.canvas.pack(fill="both", expand=True)
        self.pose_status = ttk.Label(right, text="正在加载模型…", wraplength=650)
        self.pose_status.pack(anchor="w", pady=8)
        views = ttk.Frame(right)
        views.pack(anchor='w')
        for name, view in (('演奏者视角', 'performer'), ('对面视角', 'opposite'), ('俯视', 'top')):
            ttk.Button(views, text=name, command=lambda v=view: self.canvas.set_view(v)).pack(side='left', padx=(0, 4))
        self.previous_visible = tk.BooleanVar(value=True)
        ttk.Checkbutton(views, text='上一组指法残留', variable=self.previous_visible,
                        command=self.toggle_previous).pack(side='left', padx=6)
        self.next_visible = tk.BooleanVar(value=True)
        ttk.Checkbutton(right, text='下一组指法（浅色斜线）', variable=self.next_visible,
                        command=self.toggle_next).pack(anchor='w')
        ttk.Label(right, text="姿态由指法目标求解，不是训练策略或实际触键回放。\n同色虚线显示指尖到目标的偏差；预览不校验动态碰撞或硬件可执行性。",
                  wraplength=650, foreground="#68748a").pack(anchor="w", pady=8)
        self.status = ttk.Label(outer, text="就绪")
        self.status.pack(fill="x")
        root.bind("<Control-z>", lambda e: self.undo())
        root.bind("<Control-y>", lambda e: self.redo())
        root.bind("<Control-s>", lambda e: self.save())
        root.bind("<Control-o>", lambda e: self.open())
        if path:
            self.load(path)
        if backend == 'local':
            self.remote = RemotePreview(backend='local')
        elif remote_config:
            self.remote = RemotePreview(remote_config)
            if not Path(remote_config).is_file():
                self.pose_status.configure(text='正在恢复远端三维配置并核验源码，请稍候…')
        else:
            self.pose_status.configure(text="未配置远端三维。先运行 scripts/prepare_fingering_editor.py，再重启编辑器。")
        root.after(50, self.poll)

    def changed_prompt(self):
        return not self.score or not self.score.dirty or messagebox.askyesno(
            "尚未导出", "存在尚未导出的修改，是否放弃这些修改？")

    def open(self):
        if not self.changed_prompt():
            return
        path = filedialog.askopenfilename(filetypes=[("NoteSequence", "*.proto"), ("全部文件", "*")])
        if path:
            self.load(path)

    def load(self, path):
        try:
            score = Score(path)
        except Exception as exc:
            messagebox.showerror("无法导入", str(exc))
            return
        self.stop()
        self.score, self.selected = score, None
        self.canvas.previous_assignments = []
        self.canvas.next_assignments = []
        self.filename.configure(text=score.path.name)
        self.refresh()
        self.seek(0)
        if all(n.part == 0 for n in score.seq.notes):
            messagebox.showwarning("指法来源待核对", "全部 part=0；它既可能代表右拇指，也可能是缺失字段的默认值。请核对原文件。")

    def refresh(self):
        if not self.score:
            return
        self.conflicts = self.score.conflicts()
        self.table.delete(*self.table.get_children())
        for i in sorted(range(len(self.score.seq.notes)), key=lambda i: (self.score.seq.notes[i].start_time, i)):
            n = self.score.seq.notes[i]
            self.table.insert("", "end", iid=str(i), values=(i+1, note_name(n.pitch),
                              f"{n.start_time:.6f}", f"{n.end_time:.6f}", FINGERS[n.part]),
                              tags=("conflict",) if i in self.conflicts else ())
        self.time_scale.configure(to=self.score.duration)
        self.status.configure(text=f"{len(self.score.seq.notes)} 个音符 · {self.score.duration:.3f} s · "
                              f"{len(self.conflicts)} 个重叠指法音符 · {'有未导出修改' if self.score.dirty else '已保存 / 原始文件'}")
        if self.selected is not None:
            self.table.selection_set(str(self.selected))
            self.update_selection()
        self.draw_roll()

    def draw_roll(self):
        self.roll.delete("all")
        if not self.score:
            return
        notes = self.score.seq.notes
        low, high = min(n.pitch for n in notes)-1, max(n.pitch for n in notes)+1
        height = max(300, self.roll.winfo_height())
        self.row_height = (height-28)/(high-low+1)
        self.high_pitch = high
        width = max(600, 60+self.score.duration*self.pixels_per_second)
        self.roll.configure(scrollregion=(0, 0, width, height))
        for pitch in range(low, high+1):
            y = 28+(high-pitch)*self.row_height
            if pitch % 12 in (1, 3, 6, 8, 10):
                self.roll.create_rectangle(0, y, width, y+self.row_height, fill="#e8edf4", outline="")
            if pitch % 12 == 0:
                self.roll.create_text(4, y, text=note_name(pitch), anchor="nw", fill="#65748b", font=("", 8))
        tick = max(1, math.ceil(65/self.pixels_per_second))
        for t in range(0, math.ceil(self.score.duration)+1, tick):
            x = 48+t*self.pixels_per_second
            self.roll.create_line(x, 24, x, height, fill="#d4dce8")
            self.roll.create_text(x+3, 4, text=f"{t}s", anchor="nw", fill="#64748b")
        for i, n in enumerate(notes):
            x, y = 48+n.start_time*self.pixels_per_second, 28+(high-n.pitch)*self.row_height
            end = max(x+4, 48+n.end_time*self.pixels_per_second)
            border = "#101827" if i == self.selected else "#d02e22" if i in self.conflicts else COLORS[n.part]
            self.roll.create_rectangle(x, y+1, end, y+self.row_height-1, fill=COLORS[n.part],
                                       outline=border, width=3 if i == self.selected or i in self.conflicts else 1, tags=(f"note:{i}",))
            if end-x > 12:
                self.roll.create_text((x+end)/2, y+self.row_height/2, text=str(n.part), fill="white", font=("", 8), tags=(f"note:{i}",))
        self.draw_cursor()

    def draw_cursor(self):
        self.roll.delete("cursor")
        x = 48+self.current_time*self.pixels_per_second
        self.roll.create_line(x, 0, x, max(300, self.roll.winfo_height()), fill="#e35d21", width=2, tags="cursor")

    def zoom(self, factor):
        self.pixels_per_second = min(600, max(8, self.pixels_per_second*factor))
        self.draw_roll()
        self.scroll_to_time()

    def scroll_to_time(self):
        if not self.score:
            return
        x = 48+self.current_time*self.pixels_per_second
        left = self.roll.canvasx(0)
        if x < left+40 or x > left+self.roll.winfo_width()-40:
            width = max(600, 60+self.score.duration*self.pixels_per_second)
            self.roll.xview_moveto(max(0, (x-80)/width))

    def click_roll(self, e):
        if not self.score:
            return
        items = self.roll.find_overlapping(self.roll.canvasx(e.x), e.y, self.roll.canvasx(e.x), e.y)
        for item in reversed(items):
            tags = self.roll.gettags(item)
            for tag in tags:
                if tag.startswith("note:"):
                    i = int(tag.split(":")[1])
                    self.table.selection_set(str(i))
                    self.table.see(str(i))
                    self.selected = i
                    self.update_selection()
                    self.seek(self.score.seq.notes[i].start_time)
                    return
        self.seek((self.roll.canvasx(e.x)-48)/self.pixels_per_second)

    def select_table(self, _event=None):
        values = self.table.selection()
        if values:
            changed = self.selected != int(values[0])
            self.selected = int(values[0])
            self.update_selection()
            self.draw_roll()
            if changed:
                self.seek(self.score.seq.notes[self.selected].start_time)

    def update_selection(self):
        n = self.score.seq.notes[self.selected]
        self.selection_label.configure(text=f"#{self.selected+1} · {note_name(n.pitch)} · {FINGERS[n.part]}")
        self.start_var.set(repr(n.start_time))
        self.end_var.set(repr(n.end_time))

    def assign(self, finger):
        if self.selected is not None:
            self.score.edit([self.selected], part=finger)
            self.refresh()
            self.request_pose()

    def edit_time(self):
        if self.selected is None:
            return
        try:
            self.score.edit([self.selected], start=float(self.start_var.get()), end=float(self.end_var.get()))
            self.refresh()
            self.seek(self.score.seq.notes[self.selected].start_time)
        except Exception as exc:
            messagebox.showerror("时间未修改", str(exc))

    def undo(self):
        if self.score:
            self.score.undo()
            self.refresh()
            self.seek(min(self.current_time, self.score.duration))

    def redo(self):
        if self.score:
            self.score.redo()
            self.refresh()
            self.request_pose()

    def save(self):
        if not self.score:
            return
        path = filedialog.asksaveasfilename(defaultextension=".proto", initialdir=self.score.path.parent,
                                          initialfile=self.score.path.stem+"_edited.proto",
                                          filetypes=[("NoteSequence", "*.proto")])
        if path:
            try:
                digest = self.score.save(path)
                self.refresh()
                self.status.configure(text=f"已导出：{path} · SHA256 {digest[:16]}…")
            except Exception as exc:
                messagebox.showerror("导出失败", str(exc))

    def scale_seek(self, value):
        if abs(float(value)-self.current_time) > 1e-6:
            self.seek(float(value), update_scale=False)

    def seek_entry(self):
        try:
            value = float(self.time_text.get())
            if not math.isfinite(value):
                raise ValueError()
            self.seek(value)
        except ValueError:
            messagebox.showerror("无效时间", "请输入有限秒数")

    def seek(self, value, update_scale=True):
        if not self.score:
            return
        self.current_time = max(0., min(float(value), self.score.duration))
        self.time_text.set(f"{self.current_time:.4f}")
        if update_scale:
            self.time_scale.set(self.current_time)
        self.draw_cursor()
        self.scroll_to_time()
        self.request_pose()

    def request_pose(self):
        self.request_id += 1
        self.pending_previous = ([dict(finger=n.part, pitch=n.pitch)
                                  for _, n in self.score.previous(self.current_time)] if self.score else [])
        self.pending_next = ([dict(finger=n.part, pitch=n.pitch)
                              for _, n in self.score.upcoming(self.current_time)] if self.score else [])
        if self.remote and self.ready and self.score:
            self.pose_status.configure(text=f"正在求解 {self.current_time:.3f} s 的按键姿态…")
            self.remote.submit(dict(request_id=self.request_id, active=[
                dict(finger=n.part, pitch=n.pitch) for _, n in self.score.active(self.current_time)]))

    def toggle_previous(self):
        self.canvas.show_previous = self.previous_visible.get()
        self.canvas.draw()

    def toggle_next(self):
        self.canvas.show_next = self.next_visible.get()
        self.canvas.draw()

    def play(self):
        if self.playing:
            self.stop()
        elif self.score and self.ready:
            if self.current_time >= self.score.duration:
                self.seek(0)
            self.playing = True
            self.play_button.configure(text="Ⅱ 暂停")
            self.request_pose()

    def stop(self):
        self.playing = False
        self.play_button.configure(text="▶ 姿态预览")

    def advance(self):
        if not self.playing or not self.score:
            return
        if self.current_time >= self.score.duration:
            self.stop()
            return
        # Include exact event boundaries so short notes are never skipped.
        future = [t for n in self.score.seq.notes for t in (n.start_time, n.end_time)
                  if t > self.current_time+1e-9]
        next_time = min(self.current_time+.10, min(future, default=self.score.duration))
        self.seek(next_time)

    def poll(self):
        if self.remote:
            while not self.remote.responses.empty():
                result = self.remote.responses.get_nowait()
                kind = result["type"]
                if kind == 'status':
                    self.pose_status.configure(text=result['message'])
                elif kind == "ready":
                    self.ready = True
                    self.canvas.load(result["meshes"], result["frame"])
                    self.request_pose()
                elif kind == "pose" and result["request_id"] == self.request_id:
                    self.canvas.frame = result
                    self.canvas.previous_assignments = self.pending_previous
                    self.canvas.next_assignments = self.pending_next
                    self.canvas.draw()
                    conflict = f" · 同指重叠：{result['conflicts']}" if result["conflicts"] else ""
                    self.pose_status.configure(text=f"{self.current_time:.3f} s · 最大目标偏差 {result['error_mm']:.1f} mm{conflict}\n"
                                               f"本次求解 {result['elapsed_s']:.2f} s；按求解速度逐步预览")
                    if self.playing:
                        self.root.after(100, self.advance)
                elif kind == "error" and result.get("request_id") in (None, self.request_id):
                    self.ready = False
                    self.stop()
                    self.pose_status.configure(text="三维预览失败：" + result["error"])
        self.root.after(50, self.poll)

    def close(self):
        if self.changed_prompt():
            if self.remote:
                self.remote.close()
            self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("score", nargs="?", type=Path,
                        default=ROOT/'results/exper024/input_sources/015-5_fingered.proto')
    parser.add_argument("--remote-config", type=Path, default=connection_config(ROOT, 'remote'))
    parser.add_argument('--backend', choices=('remote', 'local'), default='remote')
    args = parser.parse_args()
    from fingering_editor_limits import verify_limits
    verify_limits(role='ui', check_available=False)
    root = tk.Tk()
    Editor(root, args.score, args.remote_config, backend=args.backend)
    root.mainloop()


if __name__ == "__main__":
    main()
