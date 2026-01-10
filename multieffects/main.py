import customtkinter as ctk
import numpy as np
import sounddevice as sd
import json
import os
from scipy.signal import lfilter

# --- Configuration ---
FS = 44100
BLOCK_SIZE = 1024 

class ChocolateMultiFX_Pro(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("SunsetZ MTFX-01")
        self.geometry("800x480")
        self.configure(fg_color="#1e1b18")

        # --- Data & State ---
        self.db_file = "bank_presets_pro.json"
        self.current_bank = 1
        self.active_preset = "A"
        self.is_idle = True
        self.is_live_mode = False 
        self.is_tuner_mode = False 
        self.current_view = "Main"
        self.master_vol = 0.8
        
        self.fx_colors = {
            "Gate": "#ff9500", "Amp": "#ff4d4d", "Mod": "#4dff4d", "Dly": "#4d4dff", "Rev": "#b34dff"
        }

        # Tuner Data
        self.current_note = "-"
        self.tuning_diff = 0
        self.notes = {"E2": 82.4, "A2": 110.0, "D3": 146.8, "G3": 196.0, "B3": 246.9, "E4": 329.6}

        self.all_data = self.load_data()
        self.current_state = self.all_data[str(self.current_bank)][self.active_preset]

        # --- DSP Variables ---
        self.mod_buffer = np.zeros(8192)
        self.mod_ptr = 0
        self.lfo_phase = 0
        self.delay_buffer = np.zeros(FS + BLOCK_SIZE)
        self.rev_buffer = np.zeros(int(FS * 0.15) + BLOCK_SIZE)

        # Filter States
        self.bass_state = np.zeros(2)
        self.mid_state = np.zeros(2)
        self.treble_state = np.zeros(2)

        self.setup_ui()
        self.start_audio_stream()

    def load_data(self):
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r') as f:
                    data = json.load(f)
                    for b in data:
                        for p in data[b]:
                            if "Gate" not in data[b][p]: data[b][p]["Gate"] = False
                            if "Rev_Params" not in data[b][p]:
                                data[b][p]["Rev_Params"] = {"Size": 50, "Damp": 30, "Mix": 20}
                            if "Warmth" not in data[b][p]["Amp_Params"]: data[b][p]["Amp_Params"]["Warmth"] = 0
                            if "Notes" not in data[b][p]: data[b][p]["Notes"] = ""
                    return data
            except: pass
        return {str(b): {l: {
            "Name": "", "Gate": False, "Amp": False, "Mod": False, "Dly": False, "Rev": False,
            "Gate_Threshold": -45, "Notes": "",
            "Amp_Params": {"Vol": 50, "Gain": 30, "Bas": 50, "Mid": 50, "Tre": 50, "Drive_Mode": "Clean", "Warmth": 0},
            "Mod_Params": {"Type": "Chorus", "Rate": 1.5, "Depth": 50},
            "Dly_Params": {"Time": 300, "Feedback": 30, "Mix": 30},
            "Rev_Params": {"Size": 50, "Damp": 30, "Mix": 20}
        } for l in ["A", "B", "C", "D"]} for b in range(1, 10)}

    def save_data(self):
        with open(self.db_file, 'w') as f: json.dump(self.all_data, f)

    # --- THE ENGINE ---
    def audio_callback(self, indata, outdata, frames, time, status):
        try:
            sig = indata[:, 0].copy()

            # --- 1. Tuner & Idle Logic ---
            if self.is_tuner_mode:
                crossings = np.where(np.diff(np.sign(sig)))[0]
                if len(crossings) > 0:
                    freq = (len(crossings) * FS) / (2 * frames)
                    if freq > 60: self.process_tuner(freq)
                outdata.fill(0)
                return

            if self.is_idle:
                outdata[:, 0] = outdata[:, 1] = sig * self.master_vol
                return

            # --- 2. Amplifier (Simplified) ---
            if self.current_state.get("Amp", False):
                p = self.current_state["Amp_Params"]
                mode = p.get("Drive_Mode", "Clean")
                gain_val = (p["Gain"] / 10) + 1.0
                
                if mode == "Clean":
                    sig = sig * (1.0 + gain_val * 1.5)
                    warmth = p.get("Warmth", 0) / 100.0
                    if warmth > 0:
                        # Soft Sign: ช่วยบีบอัดสัญญาณที่ถูกขยายมาให้โค้งมน ไม่ Clip แข็ง
                        sig = sig / (1 + warmth * np.abs(sig))
                    
                    # ชดเชย Volume เพื่อให้เสียง Clean ไม่ดังทะลุเพดาน
                    sig *= 0.8
                        
                elif mode == "Overdrive 1":
                    sig = np.tanh(sig * (gain_val + 2.0))
                elif mode == "Overdrive 2":
                    sig = (2 / np.pi) * np.arctan(sig * (gain_val * 2.0))
                elif mode == "Distortion":
                    sig = np.clip(sig * (gain_val + 2.0), -0.5, 0.5) * 1.5
                elif mode == "Fuzz":
                    sig = np.sign(sig) * (1 - np.exp(-np.abs(sig * (p["Gain"] * 0.8))))

                # --- 3-Band EQ (Biquad) ---
                sig, self.bass_state = self.apply_shelf_filter(sig, 250, p["Bas"], "low", self.bass_state)
                sig, self.mid_state = self.apply_peaking_filter(sig, 1000, p["Mid"], 1.0, self.mid_state)
                sig, self.treble_state = self.apply_shelf_filter(sig, 5000, p["Tre"], "high", self.treble_state)
                sig *= (p["Vol"] / 100)

            # --- 3. Noise Gate (Post-Amp) ---
            if self.current_state.get("Gate", False):
                thresh = 10**(self.current_state.get("Gate_Threshold", -45) / 20)
                sig[np.abs(sig) < thresh] = 0

            # --- 4. Modulation (Original Stable Version) ---
            if self.current_state.get("Mod", False):
                m = self.current_state["Mod_Params"]
                t = (np.arange(frames) + self.lfo_phase) / FS
                lfo = np.sin(2 * np.pi * m["Rate"] * t)
                depth = m["Depth"] / 100.0
                
                is_flg = (m["Type"] == "Flanger")
                base, m_rng, fb = (50, 40, 0.5) if is_flg else (200, 100, 0.0)
                
                for i in range(frames):
                    d_val = base + (lfo[i] * m_rng)
                    r_ptr = int(self.mod_ptr - d_val) % len(self.mod_buffer)
                    d_smp = self.mod_buffer[r_ptr]
                    self.mod_buffer[self.mod_ptr] = sig[i] + (d_smp * fb)
                    sig[i] = (sig[i] * (1 - depth)) + (d_smp * depth)
                    self.mod_ptr = (self.mod_ptr + 1) % len(self.mod_buffer)
                
                self.lfo_phase = (self.lfo_phase + frames) % FS

            # --- 5. Delay & Reverb ---
            if self.current_state.get("Dly", False):
                d = self.current_state["Dly_Params"]
                d_smp = int((d["Time"] / 1000.0) * FS)
                idx = (np.arange(frames) + len(self.delay_buffer) - d_smp - frames) % len(self.delay_buffer)
                delayed_part = self.delay_buffer[idx]
                self.delay_buffer[-frames:] = sig + (delayed_part * (d["Feedback"]/100))
                self.delay_buffer = np.roll(self.delay_buffer, -frames)
                sig = (sig * (1 - d["Mix"]/100)) + (delayed_part * d["Mix"]/100)

            if self.current_state.get("Rev", False):
                rv = self.current_state["Rev_Params"]
                size, mix = rv["Size"]/100.0 * 0.85, rv["Mix"]/100.0
                rev_idx = (np.arange(frames) + len(self.rev_buffer) - 1500 - frames) % len(self.rev_buffer)
                rev_part = self.rev_buffer[rev_idx]
                self.rev_buffer[-frames:] = sig + (rev_part * size)
                self.rev_buffer = np.roll(self.rev_buffer, -frames)
                sig = (sig * (1 - mix)) + (rev_part * mix)

            sig = np.clip(sig * self.master_vol, -1, 1)
            outdata[:, 0] = outdata[:, 1] = sig
            
        except Exception as e:
            outdata.fill(0)

    # --- DSP Helpers (Filters) ---
    def iir_filter(self, x, b, a, state):
        y, next_state = lfilter(b, a, x, zi=state)
        return y, next_state

    def apply_shelf_filter(self, data, freq, gain_pos, shelf_type, state):
        gain_db = (gain_pos - 50) / 50 * 20
        A = 10**(gain_db / 40); omega = 2 * np.pi * freq / FS; sn = np.sin(omega); cs = np.cos(omega)
        beta = np.sqrt((A**2 + 1) / 0.7 - (A - 1)**2) / 2
        if beta < 0: beta = 0
        if shelf_type == "low":
            b = np.array([A*((A+1)-(A-1)*cs+2*beta*sn), 2*A*((A-1)-(A+1)*cs), A*((A+1)-(A-1)*cs-2*beta*sn)])
            a = np.array([(A+1)+(A-1)*cs+2*beta*sn, -2*((A-1)+(A+1)*cs), (A+1)+(A-1)*cs-2*beta*sn])
        else:
            b = np.array([A*((A+1)+(A-1)*cs+2*beta*sn), -2*A*((A-1)+(A+1)*cs), A*((A+1)+(A-1)*cs-2*beta*sn)])
            a = np.array([(A+1)-(A-1)*cs+2*beta*sn, 2*((A-1)-(A+1)*cs), (A+1)-(A-1)*cs-2*beta*sn])
        return self.iir_filter(data, b/a[0], a/a[0], state)

    def apply_peaking_filter(self, data, freq, gain_pos, Q, state):
        gain_db = (gain_pos - 50) / 50 * 18; A = 10**(gain_db / 40); omega = 2 * np.pi * freq / FS; alpha = np.sin(omega)/(2*Q); cs = np.cos(omega)
        b = np.array([1+alpha*A, -2*cs, 1-alpha*A]); a = np.array([1+alpha/A, -2*cs, 1-alpha/A])
        return self.iir_filter(data, b/a[0], a/a[0], state)

    # --- UI Helpers & Screens ---
    def save_notes(self):
        self.current_state["Notes"] = self.notes_text.get("0.0", "end-1c")
        self.save_data()

    def process_tuner(self, freq):
        closest = "-"; min_diff = float('inf')
        for n, t in self.notes.items():
            diff = freq - t
            if abs(diff) < abs(min_diff): min_diff = diff; closest = n
        self.current_note = closest; self.tuning_diff = min_diff

    def toggle_tuner(self):
        self.is_tuner_mode = not self.is_tuner_mode
        self.tuner_btn.configure(text=f"TUNER: {'ON' if self.is_tuner_mode else 'OFF'}", fg_color="#ff4d4d" if self.is_tuner_mode else "#2d2620")
        self.refresh_ui()

    def next_bank(self): self.current_bank = 1 if self.current_bank >= 9 else self.current_bank + 1; self.select_preset(self.active_preset)
    def prev_bank(self): self.current_bank = 9 if self.current_bank <= 1 else self.current_bank - 1; self.select_preset(self.active_preset)
    def on_bank_change(self, c): self.current_bank = int(c); self.select_preset(self.active_preset)
    def on_preset_change(self, c): self.select_preset(c)
    def on_name_submit(self, e): self.current_state["Name"] = self.name_entry.get(); self.save_data(); self.refresh_ui(); self.focus()
    def toggle_mode(self): self.is_live_mode = not self.is_live_mode; self.is_idle = False; self.mode_btn.configure(text=f"MODE: {'LIVE' if self.is_live_mode else 'PRESET'}"); self.refresh_ui()
    def go_back(self): self.current_view = "Main"; self.refresh_ui()
    def go_to_setup(self, k): self.current_view = f"Setup_{k}"; self.refresh_ui()
    def toggle_fx(self, k): self.current_state[k] = not self.current_state.get(k, False); self.save_data(); self.refresh_ui()
    def update_master_vol(self, v): self.master_vol = float(v)
    
    def on_drive_mode_change(self, c):
        self.current_state["Amp_Params"]["Drive_Mode"] = c
        self.save_data()
        self.refresh_ui()
    
    def on_mod_type_change(self, c):
        self.current_state["Mod_Params"]["Type"] = c
        self.save_data()
        self.refresh_ui() # เพิ่ม refresh เพื่อให้ UI อัปเดตชื่อโหมด    
        
    def update_amp_param(self, v, l, k, t):
        self.current_state["Amp_Params"][k] = int(v)
        if l: 
            l.configure(text=f"{t}\n{int(v)}")
        self.save_data()
        
    def update_mod_param(self, v, l, k, n): val = round(float(v), 1) if k=="Rate" else int(v); self.current_state["Mod_Params"][k] = val; l.configure(text=f"{n}: {val}"); self.save_data()
    def update_gate_val(self, v): self.current_state["Gate_Threshold"] = int(v); self.ng_lbl.configure(text=f"{int(v)} dB"); self.save_data()
    def update_dly_param(self, v, lbl, k, n): self.current_state["Dly_Params"][k] = int(v); lbl.configure(text=f"{n}\n{int(v)}"); self.save_data()
    def update_rev_param(self, v, lbl, k, n): self.current_state["Rev_Params"][k] = int(v); lbl.configure(text=f"{n}\n{int(v)}"); self.save_data()
    def select_preset(self, p): self.is_idle = False; self.active_preset = p; self.current_state = self.all_data[str(self.current_bank)][self.active_preset]; self.refresh_ui()
    def toggle_preset_button(self, p):
        if p == self.active_preset and not self.is_idle: self.is_idle = True
        else: self.is_idle = False; self.active_preset = p; self.current_state = self.all_data[str(self.current_bank)][self.active_preset]
        self.refresh_ui()

    def refresh_ui(self):
        for widget in self.main_container.winfo_children(): 
            widget.destroy()
        if self.is_tuner_mode:
            self.draw_tuner_screen()
        else:
            self.bank_dropdown.set(str(self.current_bank))
            self.preset_dropdown.set(self.active_preset)
            self.name_entry.delete(0, 'end')
            self.name_entry.insert(0, self.current_state.get("Name", ""))
            status = "IDLE (BYPASS)" if self.is_idle else f"{self.current_bank}{self.active_preset}: {self.current_state.get('Name','')}"
            self.status_label.configure(text=f"NOW PRESET: {status}")
            
            if self.current_view == "Main":
                if not self.is_live_mode: self.draw_preset_screen()
                else: self.draw_live_navigator_screen()
            else:
                func = getattr(self, f"draw_{self.current_view.lower()}")
                func()

    def setup_ui(self):
        self.top_bar = ctk.CTkFrame(self, fg_color="#3e362e", height=60); self.top_bar.pack(side="top", fill="x", padx=10, pady=5)
        ctk.CTkButton(self.top_bar, text="<", width=40, command=self.prev_bank).pack(side="left", padx=10)
        nav = ctk.CTkFrame(self.top_bar, fg_color="transparent"); nav.pack(side="left", expand=True)
        self.bank_dropdown = ctk.CTkOptionMenu(nav, values=[str(i) for i in range(1, 10)], width=60, command=self.on_bank_change); self.bank_dropdown.pack(side="left", padx=2)
        self.preset_dropdown = ctk.CTkOptionMenu(nav, values=["A", "B", "C", "D"], width=60, command=self.on_preset_change); self.preset_dropdown.pack(side="left", padx=2)
        self.name_entry = ctk.CTkEntry(nav, width=150); self.name_entry.pack(side="left", padx=10); self.name_entry.bind("<Return>", self.on_name_submit)
        self.tuner_btn = ctk.CTkButton(self.top_bar, text="TUNER: OFF", width=100, fg_color="#2d2620", border_width=1, command=self.toggle_tuner); self.tuner_btn.pack(side="right", padx=5)
        self.mode_btn = ctk.CTkButton(self.top_bar, text="MODE: PRESET", width=120, command=self.toggle_mode, fg_color="#5e503f"); self.mode_btn.pack(side="right", padx=10)
        ctk.CTkButton(self.top_bar, text=">", width=40, command=self.next_bank).pack(side="right", padx=10)
        self.status_bar = ctk.CTkFrame(self, fg_color="#d4a373", height=40); self.status_bar.pack(side="top", fill="x", padx=10, pady=(0, 5))
        self.status_label = ctk.CTkLabel(self.status_bar, text="", font=("Arial", 18, "bold"), text_color="#1e1b18"); self.status_label.pack(expand=True)
        bot = ctk.CTkFrame(self, fg_color="#14110f", height=50); bot.pack(side="bottom", fill="x", padx=10, pady=5)
        ctk.CTkLabel(bot, text="MASTER VOLUME", font=("Arial", 12, "bold"), text_color="#d4a373").pack(side="left", padx=20)
        self.master_sld = ctk.CTkSlider(bot, from_=0, to=1, command=self.update_master_vol); self.master_sld.set(self.master_vol); self.master_sld.pack(side="left", expand=True, padx=20)
        self.main_container = ctk.CTkFrame(self, fg_color="transparent"); self.main_container.pack(expand=True, fill="both", padx=10, pady=5)
        self.refresh_ui()

    def draw_preset_screen(self):
        for p in ["A", "B", "C", "D"]:
            p_data = self.all_data[str(self.current_bank)][p]
            f = ctk.CTkFrame(self.main_container, fg_color="transparent"); f.pack(side="left", expand=True, fill="both", padx=5)
            btn = ctk.CTkButton(f, text=f"{self.current_bank}{p}\n{p_data.get('Name','')}", height=220, font=("Arial", 25, "bold"), command=lambda x=p: self.toggle_preset_button(x), fg_color="#d4a373" if (p == self.active_preset and not self.is_idle) else "#3e362e")
            btn.pack(expand=True, fill="both"); ic = ctk.CTkFrame(f, fg_color="#14110f", height=30); ic.pack(fill="x", pady=(2, 0))
            for fx in ["Gate", "Amp", "Mod", "Dly", "Rev"]: ctk.CTkLabel(ic, text="●", text_color=self.fx_colors[fx] if p_data.get(fx) else "#2d2620", font=("Arial", 16), width=18).pack(side="left", expand=True)

    def draw_live_navigator_screen(self):
        t = ctk.CTkFrame(self.main_container, fg_color="transparent"); t.pack(side="top", expand=True, fill="both", pady=5)
        b = ctk.CTkFrame(self.main_container, fg_color="transparent"); b.pack(side="top", expand=True, fill="both", pady=5)
        for key in ["Gate", "Amp", "Mod", "Dly", "Rev"]:
            is_on = self.current_state.get(key, False)
            ctk.CTkButton(t, text=f"{key.upper()}\n{'ON' if is_on else 'OFF'}", fg_color=self.fx_colors[key] if is_on else "#3e362e", command=lambda k=key: self.toggle_fx(k)).pack(side="left", expand=True, padx=3, fill="both")
            ctk.CTkButton(b, text=f"SET {key.upper()}", fg_color="#2d2620", border_width=1, border_color=self.fx_colors[key], command=lambda k=key: self.go_to_setup(k)).pack(side="left", expand=True, padx=3, fill="both")

    def draw_setup_gate(self):
        h = ctk.CTkFrame(self.main_container, fg_color="transparent"); h.pack(fill="x", pady=10); ctk.CTkButton(h, text="← BACK", width=80, command=self.go_back).pack(side="left")
        v = self.current_state.get("Gate_Threshold", -45); body = ctk.CTkFrame(self.main_container, fg_color="#2d2620", corner_radius=15); body.pack(expand=True, fill="both", padx=20, pady=20)
        self.ng_lbl = ctk.CTkLabel(body, text=f"{int(v)} dB", font=("Arial", 40, "bold")); self.ng_lbl.pack(pady=20)
        s = ctk.CTkSlider(body, from_=-60, to=-5, width=400, command=self.update_gate_val); s.set(v); s.pack(pady=10)
    
    def draw_setup_amp(self):
        h = ctk.CTkFrame(self.main_container, fg_color="transparent"); h.pack(fill="x", pady=5)
        ctk.CTkButton(h, text="← BACK", command=self.go_back).pack(side="left")
        modes = ["Clean", "Overdrive 1", "Overdrive 2", "Distortion", "Fuzz"]
        self.drive_dropdown = ctk.CTkOptionMenu(h, values=modes, command=self.on_drive_mode_change)
        self.drive_dropdown.set(self.current_state["Amp_Params"]["Drive_Mode"]); self.drive_dropdown.pack(side="left", padx=20)
        
        f = ctk.CTkFrame(self.main_container, fg_color="#2d2620", corner_radius=15); f.pack(expand=True, fill="both", padx=5, pady=5)
        if self.current_state["Amp_Params"]["Drive_Mode"] == "Clean":
            wf = ctk.CTkFrame(f, fg_color="transparent"); wf.pack(fill="x", padx=20, pady=10)
            ctk.CTkLabel(wf, text="Warmth").pack(side="left")
            ws = ctk.CTkSlider(wf, from_=0, to=100, command=lambda v: self.update_amp_param(v, None, "Warmth", "Warmth"))
            ws.set(self.current_state["Amp_Params"].get("Warmth", 0)); ws.pack(side="left", expand=True, padx=10)

        row = ctk.CTkFrame(f, fg_color="transparent"); row.pack(expand=True, fill="both")
        for lbl, k in [("VOL", "Vol"), ("GAIN", "Gain"), ("BASS", "Bas"), ("MID", "Mid"), ("TRE", "Tre")]:
            fr = ctk.CTkFrame(row, fg_color="transparent"); fr.pack(side="left", expand=True)
            l = ctk.CTkLabel(fr, text=f"{lbl}\n{self.current_state['Amp_Params'][k]}"); l.pack()
            s = ctk.CTkSlider(fr, from_=0, to=100, orientation="vertical", height=120, command=lambda v, obj=l, key=k, t=lbl: self.update_amp_param(v, obj, key, t))
            s.set(self.current_state['Amp_Params'][k]); s.pack(pady=5)

        nf = ctk.CTkFrame(f, fg_color="#14110f"); nf.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(nf, text="PRESET NOTES:", font=("Arial", 11, "bold")).pack(anchor="w", padx=10)
        self.notes_text = ctk.CTkTextbox(nf, height=70, font=("Courier", 12))
        self.notes_text.pack(fill="x", padx=10, pady=5); self.notes_text.insert("0.0", self.current_state.get("Notes", ""))
        self.notes_text.bind("<KeyRelease>", lambda e: self.save_notes())
        
    def draw_setup_mod(self):
        h = ctk.CTkFrame(self.main_container, fg_color="transparent"); h.pack(fill="x", pady=5); ctk.CTkButton(h, text="← BACK", width=80, command=self.go_back).pack(side="left")
        self.mod_dropdown = ctk.CTkOptionMenu(h, values=["Chorus", "Tremolo", "Flanger"], command=self.on_mod_type_change)
        self.mod_dropdown.pack(side="left", padx=20); self.mod_dropdown.set(self.current_state["Mod_Params"]["Type"])
        f = ctk.CTkFrame(self.main_container, fg_color="#2d2620", corner_radius=15); f.pack(expand=True, fill="both", padx=5, pady=5); m = self.current_state["Mod_Params"]; r = ctk.CTkFrame(f, fg_color="transparent"); r.pack(fill="x", pady=20)
        for lbl, k, st, en in [("RATE (Hz)", "Rate", 0.1, 10.0), ("DEPTH (%)", "Depth", 0, 100)]:
            fr = ctk.CTkFrame(r, fg_color="transparent"); fr.pack(side="left", expand=True, fill="x", padx=10); l = ctk.CTkLabel(fr, text=f"{lbl}: {m[k]}"); l.pack()
            s = ctk.CTkSlider(fr, from_=st, to=en, command=lambda v, obj=l, key=k, n=lbl: self.update_mod_param(v, obj, key, n)); s.set(m[k]); s.pack(fill="x")

    def draw_setup_dly(self):
        h = ctk.CTkFrame(self.main_container, fg_color="transparent"); h.pack(fill="x", pady=5); ctk.CTkButton(h, text="← BACK", width=80, command=self.go_back).pack(side="left")
        d = self.current_state["Dly_Params"]; f = ctk.CTkFrame(self.main_container, fg_color="#2d2620", corner_radius=15); f.pack(expand=True, fill="both", padx=5, pady=5); r = ctk.CTkFrame(f, fg_color="transparent"); r.pack(expand=True, fill="both", pady=20)
        for lbl, k, st, en in [("TIME (ms)", "Time", 50, 1000), ("FEEDBACK", "Feedback", 0, 90), ("MIX", "Mix", 0, 100)]:
            fr = ctk.CTkFrame(r, fg_color="transparent"); fr.pack(side="left", expand=True); l = ctk.CTkLabel(fr, text=f"{lbl}\n{d[k]}"); l.pack()
            s = ctk.CTkSlider(fr, from_=st, to=en, orientation="vertical", height=120, command=lambda v, obj=l, key=k, t=lbl: self.update_dly_param(v, obj, key, t)); s.set(d[k]); s.pack(pady=5)

    def draw_setup_rev(self):
        h = ctk.CTkFrame(self.main_container, fg_color="transparent"); h.pack(fill="x", pady=5); ctk.CTkButton(h, text="← BACK", width=80, command=self.go_back).pack(side="left")
        rv = self.current_state["Rev_Params"]; f = ctk.CTkFrame(self.main_container, fg_color="#2d2620", corner_radius=15); f.pack(expand=True, fill="both", padx=5, pady=5); r = ctk.CTkFrame(f, fg_color="transparent"); r.pack(expand=True, fill="both", pady=20)
        for lbl, k, st, en in [("ROOM SIZE", "Size", 10, 95), ("MIX", "Mix", 0, 100)]:
            fr = ctk.CTkFrame(r, fg_color="transparent"); fr.pack(side="left", expand=True); l = ctk.CTkLabel(fr, text=f"{lbl}\n{rv[k]}"); l.pack()
            s = ctk.CTkSlider(fr, from_=st, to=en, orientation="vertical", height=120, command=lambda v, obj=l, key=k, t=lbl: self.update_rev_param(v, obj, key, t)); s.set(rv[k]); s.pack(pady=5)

    def draw_tuner_screen(self):
        f = ctk.CTkFrame(self.main_container, fg_color="#000000", corner_radius=15); f.pack(expand=True, fill="both", padx=20, pady=20)
        ctk.CTkLabel(f, text="CHROMATIC TUNER", font=("Arial", 24, "bold"), text_color="#ff9500").pack(pady=20)
        self.note_lbl = ctk.CTkLabel(f, text="-", font=("Arial", 120, "bold"), text_color="white"); self.note_lbl.pack(pady=10)
        self.t_gauge = ctk.CTkProgressBar(f, width=500, height=20); self.t_gauge.pack(pady=30); ctk.CTkLabel(f, text="MUTE MODE ACTIVE", text_color="#ff4d4d").pack()
        self.update_tuner_ui_loop()

    def update_tuner_ui_loop(self):
        if not self.is_tuner_mode: return
        self.note_lbl.configure(text=self.current_note)
        val = 0.5 + (np.clip(self.tuning_diff, -10, 10) / 40)
        self.t_gauge.set(val); self.t_gauge.configure(progress_color="#4dff4d" if abs(self.tuning_diff) < 1.5 else "#ff4d4d")
        self.after(100, self.update_tuner_ui_loop)

    def start_audio_stream(self):
        try:
            self.stream = sd.Stream(device=2, channels=(1, 2), callback=self.audio_callback, samplerate=FS, blocksize=BLOCK_SIZE)
            self.stream.start()
        except Exception as e: print(f"Audio Error: {e}")

if __name__ == "__main__":
    app = ChocolateMultiFX_Pro()
    app.mainloop()
