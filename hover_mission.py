# -*- coding: utf-8 -*-
"""
VTOL Kademeli DoF Transferi — Hover Kontrolü
=============================================
Hipotez:
  Düşük DoF'lu modelde hover öğrenen PPO politikasını
  yüksek DoF'lu modele transfer etmek, sıfırdan
  eğitimden daha verimlidir.

Görev: Hedef noktada sabit kal (hover).
  Başarı kriteri: Episode boyunca hedefe ortalama mesafe < 1m

Model 1 — Pitch only    : x-z düzlemi
Model 2 — Pitch + Roll  : x-y-z
Model 3 — Pitch+Roll+Yaw: Tam 3D

Yol A: 14 boyutlu sabit gözlem, 4 boyutlu sabit aksiyon.
  Kullanılmayan DoF gözlemde 0, aksiyonda maskelenir.

Gözlem (14 boyut):
  [x, z, θ, vx, vz, q,    ← pitch (6)
   y, vy, φ, p,            ← roll  (4, Model 1'de = 0)
   ψ, r,                   ← yaw   (2, Model 1-2'de = 0)
   dx, dz]                 ← hedefe uzaklık (2)

Aksiyon (4 boyut):
  [thrust, pitch_tork, roll_tork, yaw_tork]
"""

import matplotlib
matplotlib.use("Agg")

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback


# ══════════════════════════════════════════════════════════════
# 1. ORTAM
# ══════════════════════════════════════════════════════════════

TARGET = np.array([0.0, 0.0, 10.0])   # Sabit hedef: (x=0, y=0, z=10)

class VTOLHoverEnv(gym.Env):
    """
    Görev: TARGET noktasında sabit hover yap.
    Başarı: Hedefe mesafe < 1.0m ise o adımda bonus alır.

    dof=1 → Pitch only
    dof=2 → Pitch + Roll
    dof=3 → Pitch + Roll + Yaw
    """

    MAX_STEPS = 500
    OBS_DIM   = 14
    ACT_DIM   = 4
    SUCCESS_R  = 1.0    # Hedefe yakın olunca adım başı bonus

    def __init__(self, dof=1):
        super().__init__()
        assert dof in (1, 2, 3)
        self.dof = dof

        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.ACT_DIM,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.OBS_DIM,), dtype=np.float32
        )
        self.dt = 0.05

    # ── Gözlem ────────────────────────────────────────────────
    def _obs(self):
        x, z, theta, vx, vz, q = self.pitch_state
        y, phi, p               = self.roll_state
        psi, r                  = self.yaw_state

        dx = TARGET[0] - x
        dz = TARGET[2] - z

        return np.array([
            x, z, theta, vx, vz, q,   # pitch (6)
            y, 0.0, phi, p,            # roll  (4)
            psi, r,                    # yaw   (2)
            dx, dz                     # hedef (2)
        ], dtype=np.float32)

    # ── Reset ─────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps        = 0
        self.success_steps = 0   # Hedefe yakın geçirilen adım

        # Hedefe yakın rastgele başlangıç
        self.pitch_state = np.array([
            np.random.uniform(-5, 5),    # x
            np.random.uniform(5, 15),    # z
            np.random.uniform(-0.2, 0.2),# theta
            0.0, 0.0, 0.0
        ], dtype=np.float32)

        self.roll_state = np.zeros(3, dtype=np.float32)
        if self.dof >= 2:
            self.roll_state[0] = np.random.uniform(-5, 5)  # y

        self.yaw_state = np.zeros(2, dtype=np.float32)

        return self._obs(), {}

    # ── Fizik ─────────────────────────────────────────────────
    def _dynamics(self, action):
        thrust     = (action[0] + 1) * 10.0
        pitch_tork =  action[1] * 1.0
        roll_tork  =  action[2] * 1.0 if self.dof >= 2 else 0.0
        yaw_tork   =  action[3] * 1.0 if self.dof >= 3 else 0.0

        m, g           = 1.0, 9.81
        Jp, Jr, Jy     = 0.1, 0.1, 0.15
        rho, S, Cd     = 1.225, 0.5, 0.05

        # ── Pitch ─────────────────────────────────────────────
        x, z, theta, vx, vz, q = self.pitch_state
        v_mag = np.sqrt(vx**2 + vz**2) + 1e-6
        drag_x = 0.5 * rho * vx**2 * S * Cd * np.sign(vx)
        drag_z = 0.5 * rho * vz**2 * S * Cd * np.sign(vz)

        ax = -(thrust/m)*np.sin(theta) - drag_x/m
        az =  (thrust/m)*np.cos(theta) - g - drag_z/m
        aq =  pitch_tork / Jp

        self.pitch_state[3] += ax * self.dt
        self.pitch_state[4] += az * self.dt
        self.pitch_state[5] += aq * self.dt
        self.pitch_state[0] += self.pitch_state[3] * self.dt
        self.pitch_state[1] += self.pitch_state[4] * self.dt
        self.pitch_state[2] += self.pitch_state[5] * self.dt

        # ── Roll ──────────────────────────────────────────────
        if self.dof >= 2:
            y, phi, p = self.roll_state
            ap = roll_tork / Jr
            self.roll_state[2] += ap * self.dt
            self.roll_state[0] += self.roll_state[2] * self.dt * 0.5
            self.roll_state[1] += self.roll_state[2] * self.dt

        # ── Yaw ───────────────────────────────────────────────
        if self.dof >= 3:
            ar = yaw_tork / Jy
            self.yaw_state[1] += ar * self.dt
            self.yaw_state[0] += self.yaw_state[1] * self.dt

    # ── Step ──────────────────────────────────────────────────
    def step(self, action):
        self._dynamics(action)
        self.steps += 1

        x, z  = self.pitch_state[0], self.pitch_state[1]
        theta = self.pitch_state[2]
        y     = self.roll_state[0] if self.dof >= 2 else 0.0

        pos  = np.array([x, y, z])
        dist = np.linalg.norm(pos - TARGET)

        # Ödül: mesafe cezası + eğim cezası + hız cezası
        reward = (
            -1.0  * dist
            - 0.1  * abs(theta)
            - 0.05 * (self.pitch_state[3]**2 + self.pitch_state[4]**2)
        )

        # Hedefe yakınsa adım başı bonus
        if dist < 1.0:
            reward += self.SUCCESS_R
            self.success_steps += 1

        terminated = bool(
            self.steps >= self.MAX_STEPS
            or z < 0
            or abs(x) > 40
            or abs(z) > 50
        )

        info = {
            "dist"         : dist,
            "success_steps": self.success_steps,
            "success_rate" : self.success_steps / self.steps
        }
        return self._obs(), reward, terminated, False, info


# ══════════════════════════════════════════════════════════════
# 2. CALLBACK
# ══════════════════════════════════════════════════════════════

class HoverCallback(BaseCallback):
    """
    Her eval_freq adımda:
      - Ortalama ödül
      - Ortalama hedefe mesafe
      - Başarı oranı (hedefe yakın adım / toplam adım)
    """
    def __init__(self, eval_env, eval_freq=10_000,
                 n_eval=200, verbose=0):
        super().__init__(verbose)
        self.eval_env  = eval_env
        self.eval_freq = eval_freq
        self.n_eval    = n_eval
        self.timesteps    = []
        self.rewards      = []
        self.mean_dists   = []
        self.success_rates = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            ep_rewards, ep_dists, ep_success = [], [], []
            obs, _ = self.eval_env.reset()
            ep_r, ep_d, count = 0.0, [], 0

            while count < self.n_eval:
                action, _ = self.model.predict(
                    obs, deterministic=True
                )
                obs, r, done, trunc, info = \
                    self.eval_env.step(action)
                ep_r += r
                ep_d.append(info["dist"])

                if done or trunc:
                    ep_rewards.append(ep_r)
                    ep_dists.append(np.mean(ep_d))
                    ep_success.append(info["success_rate"])
                    obs, _ = self.eval_env.reset()
                    ep_r, ep_d = 0.0, []
                    count += 1

            self.timesteps.append(self.num_timesteps)
            self.rewards.append(np.mean(ep_rewards))
            self.mean_dists.append(np.mean(ep_dists))
            self.success_rates.append(np.mean(ep_success) * 100)

        return True


# ══════════════════════════════════════════════════════════════
# 3. PARAMETRELER
# ══════════════════════════════════════════════════════════════

N_ENVS    = 8
N_DOF1    = 500_000    # Model 1 (Pitch) eğitim adımı
K_DOF3    = 300_000    # Model 3 (Tam 3D) ince ayar adımı
EVAL_FREQ = 10_000
N_EVAL    = 200

PPO_KWARGS = dict(
    policy        = "MlpPolicy",
    learning_rate = 3e-4,
    n_steps       = 2048,
    batch_size    = 512,
    n_epochs      = 10,
    gamma         = 0.99,
    verbose       = 1,
    device        = "cpu",
)

vec_dof1  = make_vec_env(VTOLHoverEnv, n_envs=N_ENVS,
                          env_kwargs={"dof": 1})
vec_dof3  = make_vec_env(VTOLHoverEnv, n_envs=N_ENVS,
                          env_kwargs={"dof": 3})
eval_dof1 = VTOLHoverEnv(dof=1)
eval_dof3 = VTOLHoverEnv(dof=3)


# ══════════════════════════════════════════════════════════════
# 4. S1 & S2 — Model 1 ön eğitim
# ══════════════════════════════════════════════════════════════

print("═" * 55)
print("ADIM 1 — Ön eğitim: Model 1 (Pitch only)")
print("═" * 55)

cb_dof1 = HoverCallback(eval_dof1, eval_freq=EVAL_FREQ,
                         n_eval=N_EVAL)
model_dof1 = PPO(env=vec_dof1, **PPO_KWARGS)
model_dof1.learn(total_timesteps=N_DOF1, callback=cb_dof1)
model_dof1.save("hover_dof1_pitch")
print("  → Kaydedildi: hover_dof1_pitch.zip")


# ── S1: Zero-shot ────────────────────────────────────────────
print("\nADIM 2 — S1 Zero-shot: Model 3'te direkt test")

cb_zs = HoverCallback(eval_dof3, eval_freq=1, n_eval=N_EVAL)
cb_zs.init_callback(model_dof1)
cb_zs.num_timesteps = 0
cb_zs._on_step()

r_zs   = cb_zs.rewards[0]
d_zs   = cb_zs.mean_dists[0]
sr_zs  = cb_zs.success_rates[0]
print(f"  Ödül: {r_zs:.2f} | Mesafe: {d_zs:.2f}m | Başarı: %{sr_zs:.1f}")


# ══════════════════════════════════════════════════════════════
# 5. S2 — Model 3 ince ayar
# ══════════════════════════════════════════════════════════════

print("\nADIM 3 — S2 Transfer: Model 3'te ince ayar")

model_ft = PPO.load("hover_dof1_pitch", env=vec_dof3,
                     device="cpu")
model_ft.learning_rate = 1e-4

cb_ft = HoverCallback(eval_dof3, eval_freq=EVAL_FREQ,
                       n_eval=N_EVAL)
model_ft.learn(
    total_timesteps=K_DOF3,
    callback=cb_ft,
    reset_num_timesteps=False
)
model_ft.save("hover_dof3_finetuned")

r_ft   = cb_ft.rewards[-1]
d_ft   = cb_ft.mean_dists[-1]
sr_ft  = cb_ft.success_rates[-1]
print(f"  Ödül: {r_ft:.2f} | Mesafe: {d_ft:.2f}m | Başarı: %{sr_ft:.1f}")


# ══════════════════════════════════════════════════════════════
# 6. S3 — Model 3 sıfırdan
# ══════════════════════════════════════════════════════════════

print("\nADIM 4 — S3 Sıfırdan: Model 3'te eğitim")

cb_sc = HoverCallback(eval_dof3, eval_freq=EVAL_FREQ,
                       n_eval=N_EVAL)
model_sc = PPO(env=vec_dof3, **PPO_KWARGS)
model_sc.learn(
    total_timesteps=N_DOF1 + K_DOF3,
    callback=cb_sc
)
model_sc.save("hover_dof3_scratch")

r_sc   = cb_sc.rewards[-1]
d_sc   = cb_sc.mean_dists[-1]
sr_sc  = cb_sc.success_rates[-1]
print(f"  Ödül: {r_sc:.2f} | Mesafe: {d_sc:.2f}m | Başarı: %{sr_sc:.1f}")


# ══════════════════════════════════════════════════════════════
# 7. GÖRSELLEŞTİRME
# ══════════════════════════════════════════════════════════════

SAVE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULT_PATH = os.path.join(SAVE_DIR, "vtol_hover_dof_results.png")
COLORS      = {"zs": "#ff9999", "ft": "#66b3ff", "sc": "#99ff99"}

# S2 eğrisi için doğru x ekseni:
# cb_ft.timesteps değerleri reset_num_timesteps=False ile
# N_DOF1'den devam ediyor — olduğu gibi kullan
ft_ts = cb_ft.timesteps

fig = plt.figure(figsize=(16, 9))
fig.suptitle(
    "VTOL Hover Kontrolü — Kademeli DoF Transferi\n"
    "Model 1 (Pitch) → Model 3 (Pitch+Roll+Yaw)",
    fontsize=13, fontweight="bold"
)
gs = gridspec.GridSpec(2, 3, figure=fig,
                        hspace=0.42, wspace=0.35)

lbls = ["S1\nZero-shot", "S2\nTransfer", "S3\nSıfırdan"]
cols = [COLORS["zs"], COLORS["ft"], COLORS["sc"]]

# ── Sol üst: Final ödül ──────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(lbls, [r_zs, r_ft, r_sc], color=cols,
               edgecolor="gray", linewidth=0.6, width=0.5)
ax1.axhline(0, color="black", linewidth=0.7,
            linestyle="--", alpha=0.4)
ax1.set_title("Final ortalama ödül", fontsize=11)
ax1.set_ylabel("Ortalama ödül")
ax1.grid(axis="y", linestyle="--", alpha=0.4)
ax1.autoscale(axis="y")
for b in bars:
    h = b.get_height()
    ax1.text(b.get_x() + b.get_width()/2,
             h + (1 if h >= 0 else -5), f"{h:.1f}",
             ha="center", va="bottom",
             fontsize=10, fontweight="bold")

# ── Orta üst: Ortalama mesafe (ters eksen — düşük = iyi) ─────
ax2 = fig.add_subplot(gs[0, 1])
bars2 = ax2.bar(lbls, [d_zs, d_ft, d_sc], color=cols,
                edgecolor="gray", linewidth=0.6, width=0.5)
ax2.axhline(1.0, color="green", linewidth=1.0,
            linestyle=":", label="Hedef < 1m")
ax2.set_title("Ortalama hedefe mesafe (m)\ndüşük = iyi", fontsize=11)
ax2.set_ylabel("Mesafe (m)")
ax2.legend(fontsize=8)
ax2.grid(axis="y", linestyle="--", alpha=0.4)
# Otomatik ölçek — veriye göre ayarla
max_d = max(d_zs, d_ft, d_sc) * 1.2
ax2.set_ylim(0, max_d)
for b in bars2:
    h = b.get_height()
    ax2.text(b.get_x() + b.get_width()/2, h + max_d * 0.02,
             f"{h:.2f}m", ha="center", va="bottom",
             fontsize=10, fontweight="bold")

# ── Sağ üst: Başarı oranı (otomatik ölçek) ───────────────────
ax3 = fig.add_subplot(gs[0, 2])
bars3 = ax3.bar(lbls, [sr_zs, sr_ft, sr_sc], color=cols,
                edgecolor="gray", linewidth=0.6, width=0.5)
max_sr = max(sr_zs, sr_ft, sr_sc, 1.0) * 1.3
ax3.set_ylim(0, min(max_sr, 110))
ax3.axhline(100, color="green", linewidth=0.8,
            linestyle=":", alpha=0.5, label="Mükemmel")
ax3.set_title("Başarı oranı (%)\n(hedefe yakın adım / toplam)", fontsize=11)
ax3.set_ylabel("Başarı oranı (%)")
ax3.legend(fontsize=8)
ax3.grid(axis="y", linestyle="--", alpha=0.4)
for b in bars3:
    h = b.get_height()
    ax3.text(b.get_x() + b.get_width()/2,
             h + max_sr * 0.02,
             f"%{h:.1f}", ha="center", va="bottom",
             fontsize=10, fontweight="bold")

# ── Alt sol+orta: Öğrenme eğrisi — ödül ─────────────────────
ax4 = fig.add_subplot(gs[1, 0:2])

# S3: 0'dan N_DOF1+K_DOF3 adıma kadar
ax4.plot(cb_sc.timesteps, cb_sc.rewards,
         color=COLORS["sc"], linewidth=2,
         label="S3 Sıfırdan")

# S2: sadece ince ayar kısmı (N_DOF1'den itibaren)
ax4.plot(ft_ts, cb_ft.rewards,
         color=COLORS["ft"], linewidth=2.5,
         label="S2 Transfer (ince ayar)")

# Zero-shot yatay çizgi
ax4.axhline(r_zs, color=COLORS["zs"], linestyle="--",
            linewidth=1.5,
            label=f"S1 Zero-shot ({r_zs:.1f})")

# İnce ayar başlangıç noktası
ax4.axvline(N_DOF1, color="gray", linestyle=":",
            linewidth=1.2, label=f"İnce ayar başladı ({N_DOF1:,})")

ax4.set_xlabel("Toplam adım sayısı")
ax4.set_ylabel("Ortalama ödül")
ax4.set_title("Öğrenme eğrisi — Model 3 (Pitch+Roll+Yaw)\n"
              "S2 yalnızca ince ayar aşamasında gösteriliyor",
              fontsize=11)
ax4.legend(fontsize=9)
ax4.grid(linestyle="--", alpha=0.4)
ax4.autoscale(axis="y")

# ── Alt sağ: Başarı oranı eğrisi (otomatik ölçek) ────────────
ax5 = fig.add_subplot(gs[1, 2])
ax5.plot(cb_sc.timesteps, cb_sc.success_rates,
         color=COLORS["sc"], linewidth=2,
         label="S3 Sıfırdan")
ax5.plot(ft_ts, cb_ft.success_rates,
         color=COLORS["ft"], linewidth=2.5,
         label="S2 Transfer")
ax5.axhline(sr_zs, color=COLORS["zs"], linestyle="--",
            linewidth=1.5,
            label=f"S1 Zero-shot (%{sr_zs:.1f})")

# Otomatik ölçek — değerlere göre
all_sr = cb_sc.success_rates + cb_ft.success_rates + [sr_zs]
max_sr_curve = max(all_sr) if max(all_sr) > 0 else 1.0
ax5.set_ylim(0, min(max_sr_curve * 1.3, 110))

ax5.set_xlabel("Toplam adım sayısı")
ax5.set_ylabel("Başarı oranı (%)")
ax5.set_title("Başarı oranı eğrisi", fontsize=11)
ax5.legend(fontsize=8)
ax5.grid(linestyle="--", alpha=0.4)

plt.savefig(RESULT_PATH, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  → Grafik kaydedildi: {RESULT_PATH}")


# ══════════════════════════════════════════════════════════════
# 8. SONUÇ RAPORU
# ══════════════════════════════════════════════════════════════

gain_r  = r_ft  - r_sc
gain_d  = d_sc  - d_ft   # Mesafe farkı: sc > ft iyi
gain_sr = sr_ft - sr_sc

print("\n" + "═" * 60)
print("SONUÇ RAPORU")
print("═" * 60)
print(f"{'Strateji':<22} {'Ödül':>8} {'Mesafe':>8} {'Başarı':>8}")
print("-" * 50)
print(f"{'S1 Zero-shot':<22} {r_zs:>8.2f} {d_zs:>7.2f}m {sr_zs:>7.1f}%")
print(f"{'S2 Transfer ★':<22} {r_ft:>8.2f} {d_ft:>7.2f}m {sr_ft:>7.1f}%")
print(f"{'S3 Sıfırdan':<22} {r_sc:>8.2f} {d_sc:>7.2f}m {sr_sc:>7.1f}%")
print("-" * 50)
print(f"\nTransfer avantajı (S2 − S3):")
print(f"  Ödül         : {gain_r:+.2f}")
print(f"  Mesafe farkı : {gain_d:+.2f}m  (+ = S2 daha yakın)")
print(f"  Başarı farkı : {gain_sr:+.1f} pp")
if gain_r > 0:
    pct = abs(gain_r / (abs(r_sc) + 1e-9)) * 100
    print(f"\n  ✓ Hipotez DESTEKLENDI — %{pct:.1f} avantaj")
else:
    print("\n  ✗ Hipotez desteklenmedi")
print("═" * 60)