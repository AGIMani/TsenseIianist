"""Small Tk control panel beside the MuJoCo viewer; main-thread event polling."""
from .physics_settings import GROUPS, SETTINGS_PATH, save


class ContactPanel:
    def __init__(self, controller, *, save_path=SETTINGS_PATH):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.controller, self.save_path = tk, controller, save_path
        self.expected = save_path.read_bytes() if save_path.exists() else None
        self.root = tk.Tk()
        self.root.title('V3.2 solimp — 接触阻抗')
        self.root.geometry('510x590')
        self.closed = False
        self.loading = False
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.group = tk.StringVar(value='piano')
        ttk.Label(self.root, text='对象：piano=琴键/琴箱，hand=双手，key_limit=琴键限位').pack(pady=8)
        box = ttk.Combobox(self.root, textvariable=self.group, values=GROUPS, state='readonly')
        box.pack(); box.bind('<<ComboboxSelected>>', lambda _: self.refresh())
        self.scales=[]
        specs=[('d0 初始阻抗', .0001,.9999,.0001),('dwidth 最终阻抗',.0001,.9999,.0001),
               ('width 过渡宽度 (接触为 m；限位为 rad)',.00001,.01,.00001),
               ('midpoint 过渡位置',.01,.99,.01),('power 曲线指数',1,10,.1)]
        advanced=ttk.LabelFrame(self.root,text='高级 solimp 设置')
        for i,(label, lo, hi, resolution) in enumerate(specs):
            if i == 1:advanced.pack(pady=5)
            scale=tk.Scale(self.root if i == 0 else advanced,label=label,from_=lo,to=hi,resolution=resolution,
                           orient='horizontal',length=465,command=self.changed)
            scale.pack();self.scales.append(scale)
        self.status=tk.StringVar()
        ttk.Label(self.root,textvariable=self.status,wraplength=480).pack(pady=5)
        row=ttk.Frame(self.root);row.pack()
        ttk.Button(row,text='恢复该组初始值',command=self.restore).pack(side='left')
        ttk.Button(row,text='恢复原版 Shadow solimp',command=self.restore_shadow).pack(side='left',padx=8)
        ttk.Button(row,text='保存接触配置',command=self.save).pack(side='left')
        self.refresh()

    def refresh(self):
        self.loading=True
        for s,v in zip(self.scales,self.controller.settings[self.group.get()]['solimp']):s.set(v)
        self.root.update_idletasks()
        self.loading=False
        self.describe()

    def describe(self):
        s=self.controller.settings
        p,h=s['piano'],s['hand']
        mixed=p['solimp'] if p['priority']>h['priority'] else h['solimp'] if h['priority']>p['priority'] else [(a+b)/2 for a,b in zip(p['solimp'],h['solimp'])]
        self.status.set('当前手—键合并 solimp: '+str([round(v,5) for v in mixed])+'\n滑动即生效；保存仅写配置，下次新导出训练包生效。')

    def changed(self, _=None):
        if not self.loading:
            self.controller.set_values(self.group.get(),[s.get() for s in self.scales])
            self.describe()

    def restore(self):
        g=self.group.get();self.controller.set_values(g,self.controller.initial[g]['solimp']);self.refresh()

    def restore_shadow(self):
        self.controller.restore_shadow()
        self.refresh()
        self.status.set('已恢复原版 Shadow 的三组 solimp；滑动/恢复已实时生效，保存后供下次源预览使用。')

    def save(self):
        try:
            self.expected=save(self.controller.settings,self.save_path,expected=self.expected)
            self.status.set(f'已保存并备份：{self.save_path}\n训练须重新导出新的 scene/task。')
            print(f'Saved contact settings: {self.save_path}',flush=True)
        except (OSError,ValueError) as e:
            self.status.set(str(e))

    def poll(self):
        if not self.closed:self.root.update()

    def close(self):
        if not self.closed:
            self.closed=True;self.root.destroy()
