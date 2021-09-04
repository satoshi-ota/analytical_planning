import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from utils import util

def analytical_planning(v0, vt, a0, am, jm):

    t_dec = (am - a0) / jm
    t_acc = (0.0 - am) / np.abs(jm)
    t_min = ((vt - v0) - a0 * t_dec - 0.5 * jm * t_dec**2 - 0.5 * am * t_acc) / am
    print(t_min)

    h_traj = np.empty((3, 0))

    if t_min > 0.0:

        t_total = t_dec + t_min + t_acc
        t = np.linspace(0, t_total, 1000)
        p1 = util.integral(a0, v0, 0.0, jm, t_dec)
        p2 = util.integral(p1[0, 0], p1[1, 0], p1[2, 0], 0.0, t_min)
        p3 = util.integral(p2[0, 0], p2[1, 0], p2[2, 0], np.abs(jm), t_acc)

        for tk in t:
            print(tk)
            if tk < t_dec:
                p_traj = util.integral(a0, v0, 0.0, jm, tk)
            elif tk < t_dec + t_min:
                p_traj = util.integral(p1[0, 0], p1[1, 0], p1[2, 0], 0.0, tk - t_dec)
            elif tk < t_total:
                p_traj = util.integral(p2[0, 0], p2[1, 0], p2[2, 0], np.abs(jm), tk - t_dec -t_min)

            h_traj = np.hstack((h_traj, p_traj))

    return h_traj

a0 = 0.0
am = -1.0
jm = -0.3
v0 = 20.0 / 3.6
vt = 5.0 / 3.6

h_traj = analytical_planning(v0, vt, a0, am, jm)

fig, ax = plt.subplots(figsize=(6,6))
plt.subplots_adjust(left=0.25, bottom=0.4)
plt.grid()
l, = plt.plot(h_traj[2, :], h_traj[1, :], lw=2)

axcolor = 'gold'
ax_vt = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
ax_jm = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)
ax_am = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_a0 = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor=axcolor)
ax_v0 = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)

sli_vt = Slider(ax_vt, 'Target Vel[km/h]', 0, 10.0, valinit=vt,valstep=1.0)
sli_jm = Slider(ax_jm, 'Min Jerk[m/sss]', -3.0, 0.0, valinit=jm, valstep=0.1)
sli_am = Slider(ax_am, 'Min Acc[m/ss]', -3.0, 0.0, valinit=am,valstep=0.1)
sli_a0 = Slider(ax_a0, 'Current Acc[m/ss]', 0.0, 10.0, valinit=a0,valstep=0.1)
sli_v0 = Slider(ax_v0, 'Current Vel[km/h]', 0.0, 10.0, valinit=v0,valstep=1.0)

def update(val):
    svt = sli_vt.val / 3.6
    sjm = sli_jm.val
    sam = sli_am.val
    sa0 = sli_a0.val
    sv0 = sli_v0.val / 3.6
    h_traj = analytical_planning(sv0, svt, sa0, sam, sjm)
    l.set_xdata(h_traj[2, :])
    l.set_ydata(h_traj[1, :])
    fig.canvas.draw_idle()

sli_vt.on_changed(update)
sli_jm.on_changed(update)
sli_am.on_changed(update)
sli_a0.on_changed(update)
sli_v0.on_changed(update)

resetax = plt.axes([0.8, 0.0, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
    sli_vt.reset()
    sli_jm.reset()
    sli_am.reset()
    sli_a0.reset()
    sli_v0.reset()

button.on_clicked(reset)

plt.show()
