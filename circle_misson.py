# -*- coding: utf-8 -*-
"""
VTOL Transfer Learning v2
==========================
Hipotez:
  Basit ortamda (gürültüsüz 4 waypoint) ön eğitim +
  karmaşık ortamda (gürültülü 4 waypoint) ince ayar,
  sıfırdan eğitimden daha verimlidir.

Basit  : 4 waypoint, gürültü yok, ideal fizik
Karmaşık: 4 waypoint, türbülans + hava direnci

3 Strateji — toplam bütçe eşit (N + K adım):
  S1  Zero-shot : Basit'te N adım → Karmaşık'ta test
  S2  Transfer  : Basit'te N adım → Karmaşık'ta K adım → test
  S3  Sıfırdan : Karmaşık'ta N+K adım → test
"""

import matplotlib
matplotlib.use("Agg")

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback


# ══════════════════════════════════════════════════════════════
# 1. ORTAM — tek sınıf, iki mod
# ══════════════════════════════════════════════════════════════

class VTOLEnv(gym.Env):
    """
    complex_mode=False → Basit ortam
      - İdeal fizik, gürültü yok, hava direnci yok
      - 4 rastgele waypoint

    complex_mode=True → Karmaşık ortam
      - Türbülans + hava direnci aktif
      - 4 rastgele waypoint (aynı görev, zor fizik)

    Gözlem (8 boyut): [x, y, θ, vx, vy, ω, dx, dy]
      dx, dy → aktif waypoint'e uzaklık

    Aksiyon (2 boyut): [thrust, tork] ∈ [-1, 1]
    """

    N_WAYPOINTS   = 4
    REACH_RADIUS  = 1.5    # metre
    WP_BONUS      = 20.0   # waypoint ödülü
    MAX_STEPS     = 400

    def __init__(self, complex_mode=False):
        super().__init__()
        self.complex_mode = complex_mode

        self.action_space = spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )
        self.dt = 0.05

    # ── Yardımcı: waypoint üret ──────────────────────────────
    def _sample_waypoints(self):
        pts = []
        while len(pts) < self.N_WAYPOINTS:
            p = np.array([
                np.random.uniform(-12, 12),
                np.random.uniform(3, 20)
            ])
            # Birbirine çok yakın olmasın
            if all(np.linalg.norm(p - q) > 4.0 for q in pts):
                pts.append(p)
        return pts

    # ── Yardımcı: gözlem ──────────────────────────────────────
    def _obs(self):
        x, y, th, vx, vy, om = self.state
        target = self.waypoints[self.wp_idx]
        dx = target[0] - x
        dy = target[1] - y
        return np.array([x, y, th, vx, vy, om, dx, dy],
                        dtype=np.float32)

    # ── Yardımcı: fizik adımı ─────────────────────────────────
    def _dynamics(self, action):
        x, y, theta, vx, vy, omega = self.state
        u1    = (action[0] + 1) * 10.0   # İtiş [0, 20] N
        u2    = action[1] * 1.0           # Tork [-1, 1] Nm
        m, J, g = 1.0, 0.1, 9.81

        if not self.complex_mode:
            # ── Basit: ideal PVTOL ──
            ax = -(u1 / m) * np.sin(theta)
            ay =  (u1 / m) * np.cos(theta) - g
        else:
            # ── Karmaşık: hava direnci + türbülans ──
            rho, S, Cd = 1.225, 0.5, 0.1
            v_mag = np.sqrt(vx**2 + vy**2) + 1e-6
            drag  = 0.5 * rho * v_mag**2 * S * Cd
            turb  = np.random.normal(0, 0.4)   # Rastgele rüzgar

            ax = -(u1/m)*np.sin(theta) - (drag*vx/v_mag) + turb
            ay =  (u1/m)*np.cos(theta) - g - (drag*vy/v_mag)

        alpha = u2 / J

        self.state[3] += ax    * self.dt
        self.state[4] += ay    * self.dt
        self.state[5] += alpha * self.dt
        self.state[0] += self.state[3] * self.dt
        self.state[1] += self.state[4] * self.dt
        self.state[2] += self.state[5] * self.dt

    # ── reset ─────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.waypoints  = self._sample_waypoints()
        self.wp_idx     = 0
        self.wp_reached = 0

        x  = np.random.uniform(-10, 10)
        y  = np.random.uniform(2, 18)
        th = np.random.uniform(-0.4, 0.4)
        self.state = np.array(
            [x, y, th, 0.0, 0.0, 0.0], dtype=np.float32
        )
        self.steps = 0
        return self._obs(), {}

    # ── step ──────────────────────────────────────────────────
    def step(self, action):
        self._dynamics(action)
        x, y, theta = self.state[0], self.state[1], self.state[2]
        self.steps += 1

        target = self.waypoints[self.wp_idx]
        dist   = np.linalg.norm(np.array([x, y]) - target)

        # Ödül
        reward = (
            -1.0  * dist
            - 0.05 * abs(theta)
            - 0.01 * (self.state[3]**2 + self.state[4]**2)
        )

        # Waypoint tamamlandı mı?
        if dist < self.REACH_RADIUS:
            reward += self.WP_BONUS
            self.wp_reached += 1
            if self.wp_idx < self.N_WAYPOINTS - 1:
                self.wp_idx += 1   # Sonraki waypoint'e geç

        terminated = bool(
            self.steps >= self.MAX_STEPS
            or y < 0
            or abs(x) > 50
            or abs(y) > 60
        )

        info = {"wp_reached": self.wp_reached}
        return self._obs(), reward, terminated, False, info


# ══════════════════════════════════════════════════════════════
# 2. CALLBACK — öğrenme eğrisi kaydeder
# ══════════════════════════════════════════════════════════════

class TrainCallback(BaseCallback):
    """
    Her eval_freq adımda:
      - Ortalama ödül
      - Waypoint başarı oranı (%)
    kaydeder.
    """
    def __init__(self, eval_env, eval_freq=10_000,
                 n_eval=200, verbose=0):
        super().__init__(verbose)
        self.eval_env  = eval_env
        self.eval_freq = eval_freq
        self.n_eval    = n_eval
        self.timesteps = []
        self.rewards   = []
        self.wp_rates  = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            ep_rewards, wp_totals = [], []
            obs, _ = self.eval_env.reset()
            ep_r, wp_ep, count = 0.0, 0, 0

            while count < self.n_eval:
                action, _ = self.model.predict(
                    obs, deterministic=True
                )
                obs, r, done, trunc, info = self.eval_env.step(action)
                ep_r += r
                if "wp_reached" in info:
                    wp_ep = info["wp_reached"]
                if done or trunc:
                    ep_rewards.append(ep_r)
                    wp_totals.append(wp_ep)
                    obs, _ = self.eval_env.reset()
                    ep_r, wp_ep = 0.0, 0
                    count += 1

            self.timesteps.append(self.num_timesteps)
            self.rewards.append(np.mean(ep_rewards))
            self.wp_rates.append(
                np.mean(wp_totals) / VTOLEnv.N_WAYPOINTS * 100
            )
        return True


# ══════════════════════════════════════════════════════════════
# 3. DENEY PARAMETRELERİ
# ══════════════════════════════════════════════════════════════

N_ENVS    = 4        # Paralel ortam
N_SIMPLE  = 1_000_000  # Basit ortam eğitim adımı
K_COMPLEX =   500_000  # Karmaşık ortam ince ayar adımı
# S3 toplam = N_SIMPLE + K_COMPLEX  → eşit bütçe garantisi

EVAL_FREQ = 20_000
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

# Vektörlü ortamlar (eğitim)
vec_simple  = make_vec_env(
    VTOLEnv, n_envs=N_ENVS,
    env_kwargs={"complex_mode": False}
)
vec_complex = make_vec_env(
    VTOLEnv, n_envs=N_ENVS,
    env_kwargs={"complex_mode": True}
)

# Tekli ortamlar (değerlendirme)
eval_simple  = VTOLEnv(complex_mode=False)
eval_complex = VTOLEnv(complex_mode=True)


# ══════════════════════════════════════════════════════════════
# 4. S1 & S2 — Basit ortamda ön eğitim
# ══════════════════════════════════════════════════════════════

print("═" * 55)
print("ADIM 1 — Ön eğitim: basit ortam (gürültüsüz 4 WP)")
print("═" * 55)

cb_simple = TrainCallback(
    eval_simple, eval_freq=EVAL_FREQ, n_eval=N_EVAL
)
model_simple = PPO(env=vec_simple, **PPO_KWARGS)
model_simple.learn(total_timesteps=N_SIMPLE, callback=cb_simple)
model_simple.save("simple_pretrained")
print("  → Model kaydedildi: simple_pretrained.zip")


# ── S1: Zero-shot değerlendirme ───────────────────────────────
print("\nADIM 2 — S1 Zero-shot: karmaşık ortamda direkt test")

cb_zs = TrainCallback(eval_complex, eval_freq=1, n_eval=N_EVAL)
cb_zs.init_callback(model_simple)
cb_zs.num_timesteps = 0
cb_zs._on_step()

r_zs  = cb_zs.rewards[0]
wp_zs = cb_zs.wp_rates[0]
print(f"  Ödül: {r_zs:.2f}  |  WP başarı: %{wp_zs:.1f}")


# ══════════════════════════════════════════════════════════════
# 5. S2 — Karmaşık ortamda ince ayar
# ══════════════════════════════════════════════════════════════

print("\nADIM 3 — S2 Transfer: karmaşık ortamda ince ayar")

model_ft = PPO.load(
    "simple_pretrained", env=vec_complex, device="cpu"
)
model_ft.learning_rate = 1e-4   # Önceki 5e-5 çok küçüktü, politika güncellenemiyordu

cb_ft = TrainCallback(
    eval_complex, eval_freq=EVAL_FREQ, n_eval=N_EVAL
)
model_ft.learn(
    total_timesteps=K_COMPLEX,
    callback=cb_ft,
    reset_num_timesteps=False
)
model_ft.save("transfer_finetuned")

r_ft  = cb_ft.rewards[-1]
wp_ft = cb_ft.wp_rates[-1]
print(f"  Ödül: {r_ft:.2f}  |  WP başarı: %{wp_ft:.1f}")


# ══════════════════════════════════════════════════════════════
# 6. S3 — Karmaşık ortamda sıfırdan
# ══════════════════════════════════════════════════════════════

print("\nADIM 4 — S3 Sıfırdan: karmaşık ortamda eğitim")

cb_sc = TrainCallback(
    eval_complex, eval_freq=EVAL_FREQ, n_eval=N_EVAL
)
model_sc = PPO(env=vec_complex, **PPO_KWARGS)
model_sc.learn(
    total_timesteps=N_SIMPLE + K_COMPLEX,  # Eşit toplam bütçe
    callback=cb_sc
)
model_sc.save("scratch_complex")

r_sc  = cb_sc.rewards[-1]
wp_sc = cb_sc.wp_rates[-1]
print(f"  Ödül: {r_sc:.2f}  |  WP başarı: %{wp_sc:.1f}")


# ══════════════════════════════════════════════════════════════
# 7. GÖRSELLEŞTİRME
# ══════════════════════════════════════════════════════════════

import os
SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_PATH = os.path.join(SAVE_DIR, "vtol_transfer_v2_results.png")

COLORS = {"zs": "#ff9999", "ft": "#66b3ff", "sc": "#99ff99"}

fig = plt.figure(figsize=(16, 9))
fig.suptitle(
    "VTOL Transfer Öğrenimi — Basit (gürültüsüz) → Karmaşık (gürültülü)\n"
    "Hipotez: Ön eğitim + ince ayar, sıfırdan eğitimden üstündür",
    fontsize=13, fontweight="bold"
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

# ── Sol üst: Final ödül ──────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
lbls    = ["S1\nZero-shot", "S2\nTransfer", "S3\nSıfırdan"]
rewards = [r_zs, r_ft, r_sc]
cols    = [COLORS["zs"], COLORS["ft"], COLORS["sc"]]
bars = ax1.bar(lbls, rewards, color=cols,
               edgecolor="gray", linewidth=0.6, width=0.5)
ax1.axhline(0, color="black", linewidth=0.7,
            linestyle="--", alpha=0.4)
ax1.set_title("Final ortalama ödül", fontsize=11)
ax1.set_ylabel("Ortalama ödül")
ax1.grid(axis="y", linestyle="--", alpha=0.4)
for b in bars:
    h = b.get_height()
    ax1.text(b.get_x() + b.get_width()/2,
             h + (1 if h >= 0 else -5),
             f"{h:.1f}", ha="center", va="bottom",
             fontsize=10, fontweight="bold")

# ── Orta üst: Waypoint başarı ────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
wp_vals = [wp_zs, wp_ft, wp_sc]
bars2 = ax2.bar(lbls, wp_vals, color=cols,
                edgecolor="gray", linewidth=0.6, width=0.5)
ax2.set_ylim(0, 110)
ax2.set_title("Waypoint başarı oranı (%)", fontsize=11)
ax2.set_ylabel("Tamamlanan WP oranı")
ax2.axhline(100, color="green", linewidth=0.8,
            linestyle=":", alpha=0.5, label="Mükemmel")
ax2.grid(axis="y", linestyle="--", alpha=0.4)
ax2.legend(fontsize=8)
for b in bars2:
    h = b.get_height()
    ax2.text(b.get_x() + b.get_width()/2, h + 1,
             f"%{h:.1f}", ha="center", va="bottom",
             fontsize=10, fontweight="bold")

# ── Sağ üst: Özet tablo ──────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.axis("off")
gain_r  = r_ft  - r_sc
gain_wp = wp_ft - wp_sc
rows = [
    ("Strateji",        "Ödül",          "WP %"),
    ("S1 Zero-shot",    f"{r_zs:.1f}",   f"%{wp_zs:.1f}"),
    ("S2 Transfer ★",  f"{r_ft:.1f}",   f"%{wp_ft:.1f}"),
    ("S3 Sıfırdan",     f"{r_sc:.1f}",   f"%{wp_sc:.1f}"),
    ("",                "",              ""),
    ("S2 − S3",         f"{gain_r:+.1f}", f"{gain_wp:+.1f}pp"),
    ("Hipotez",
     "✓ Desteklendi" if gain_r > 0 else "✗ Desteklenmedi",
     ""),
]
ax3.set_title("Özet", fontsize=11)
y = 0.95
for i, row in enumerate(rows):
    w = "bold" if i in (0, 5, 6) else "normal"
    c = "#1a5c8c" if i == 2 else "black"
    ax3.text(0.0,  y, row[0], transform=ax3.transAxes,
             fontsize=9, fontweight=w, color=c)
    ax3.text(0.55, y, row[1], transform=ax3.transAxes,
             fontsize=9, fontweight=w, color=c)
    ax3.text(0.80, y, row[2], transform=ax3.transAxes,
             fontsize=9, fontweight=w, color=c)
    y -= 0.13
    if i == 0:
        ax3.plot([0, 1], [y + 0.06, y + 0.06],
                 color="gray", linewidth=0.6,
                 transform=ax3.transAxes)

# ── Alt sol+orta: Öğrenme eğrisi — ödül ─────────────────────
ax4 = fig.add_subplot(gs[1, 0:2])
ax4.plot(cb_sc.timesteps, cb_sc.rewards,
         color=COLORS["sc"], linewidth=2,
         label="S3 Sıfırdan")
ax4.plot(cb_ft.timesteps, cb_ft.rewards,
         color=COLORS["ft"], linewidth=2.5,
         label="S2 Transfer (ince ayar)")
ax4.axhline(r_zs, color=COLORS["zs"], linestyle="--",
            linewidth=1.5,
            label=f"S1 Zero-shot ({r_zs:.1f})")
ax4.axvline(N_SIMPLE, color="gray", linestyle=":",
            linewidth=1.2, label="İnce ayar başladı")
ax4.set_xlabel("Karmaşık ortamdaki adım sayısı")
ax4.set_ylabel("Ortalama ödül")
ax4.set_title("Öğrenme eğrisi — karmaşık ortam", fontsize=11)
ax4.legend(fontsize=9)
ax4.grid(linestyle="--", alpha=0.4)

# ── Alt sağ: Öğrenme eğrisi — WP başarı ─────────────────────
ax5 = fig.add_subplot(gs[1, 2])
ax5.plot(cb_sc.timesteps, cb_sc.wp_rates,
         color=COLORS["sc"], linewidth=2,
         label="S3 Sıfırdan")
ax5.plot(cb_ft.timesteps, cb_ft.wp_rates,
         color=COLORS["ft"], linewidth=2.5,
         label="S2 Transfer")
ax5.axhline(wp_zs, color=COLORS["zs"], linestyle="--",
            linewidth=1.5,
            label=f"S1 Zero-shot (%{wp_zs:.1f})")
ax5.set_ylim(0, 110)
ax5.set_xlabel("Karmaşık ortamdaki adım")
ax5.set_ylabel("WP başarı oranı (%)")
ax5.set_title("Waypoint başarı eğrisi", fontsize=11)
ax5.legend(fontsize=8)
ax5.grid(linestyle="--", alpha=0.4)

plt.savefig(RESULT_PATH, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  → Grafik kaydedildi: {RESULT_PATH}")


# ══════════════════════════════════════════════════════════════
# 8. SONUÇ RAPORU
# ══════════════════════════════════════════════════════════════

print("\n" + "═" * 55)
print("SONUÇ RAPORU")
print("═" * 55)
print(f"{'Strateji':<22} {'Ödül':>8} {'WP Başarı':>10}")
print("-" * 42)
print(f"{'S1 Zero-shot':<22} {r_zs:>8.2f} {wp_zs:>9.1f}%")
print(f"{'S2 Transfer ★':<22} {r_ft:>8.2f} {wp_ft:>9.1f}%")
print(f"{'S3 Sıfırdan':<22} {r_sc:>8.2f} {wp_sc:>9.1f}%")
print("-" * 42)
print(f"\nTransfer avantajı (S2 − S3):")
print(f"  Ödül     : {gain_r:+.2f}")
print(f"  WP başarı: {gain_wp:+.1f} pp")
if gain_r > 0:
    pct = abs(gain_r / (abs(r_sc) + 1e-9)) * 100
    print(f"\n  ✓ Hipotez DESTEKLENDI — %{pct:.1f} avantaj")
else:
    print("\n  ✗ Hipotez desteklenmedi")
print("═" * 55)