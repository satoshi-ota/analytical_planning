import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from utils import util

def analytical_planning(v0, vt, a0, am, jm, vobs):

    t_dec = (am - a0) / jm
    t_acc = (0.0 - am) / np.abs(jm)
    t_min = ((vt - v0) - a0 * t_dec - 0.5 * jm * t_dec**2 - 0.5 * am * t_acc) / am

    h_traj = np.empty((5, 0))
    h_obs = np.empty((3, 0))

    if t_min > 0.0:

        t_total = t_dec + t_min + t_acc
        t = np.linspace(0, t_total, 1000)

        p1 = util.integral(a0, v0, 0.0, jm, t_dec)
        p2 = util.integral(p1[2, 0], p1[3, 0], p1[4, 0], 0.0, t_min)
        p3 = util.integral(p2[2, 0], p2[3, 0], p2[4, 0], np.abs(jm), t_acc)
        x0_obs = p3[4, 0] + 5.0 - vobs * t_total 

        for tk in t:
            if tk < t_dec:
                p_traj = util.integral(a0, v0, 0.0, jm, tk)
            elif tk < t_dec + t_min:
                p_traj = util.integral(p1[2, 0], p1[3, 0], p1[4, 0], 0.0, tk - t_dec)
            elif tk < t_total:
                p_traj = util.integral(p2[2, 0], p2[3, 0], p2[4, 0], np.abs(jm), tk - t_dec -t_min)

            h_traj = np.hstack((h_traj, p_traj))

            x_obs = x0_obs + vobs * tk
            p_obs = np.array([[tk, x_obs, 0.0]]).T
            h_obs = np.hstack((h_obs, p_obs))

        h_obs[2, :] = h_obs[1, :] - h_traj[4, :]

        print(p3[0, -1], p3[1, -1], p3[2, -1], p3[3, -1], p3[4, -1])
            
        return h_traj, h_obs, t

    else:

        a1_square = (vt - v0 - 0.5 * (0.0 - a0) / jm * a0) * (2.0 * np.abs(jm) * jm / (np.abs(jm) - jm));
        a1 = -np.sqrt(a1_square)
        t_dec = (a1 - a0) / jm
        t_acc = (0.0 - a1) / np.abs(jm)

        t_total = t_dec + t_acc
        t = np.linspace(0, t_total, 1000)

        p1 = util.integral(a0, v0, 0.0, jm, t_dec)
        p2 = util.integral(p1[2, 0], p1[3, 0], p1[4, 0], np.abs(jm), t_acc)
        x0_obs = p2[4, 0] + 5.0 - vobs * t_total 

        for tk in t:
            if tk < t_dec:
                p_traj = util.integral(a0, v0, 0.0, jm, tk)
            elif tk < t_total:
                p_traj = util.integral(p1[2, 0], p1[3, 0], p1[4, 0], np.abs(jm), tk - t_dec)

            h_traj = np.hstack((h_traj, p_traj))

            x_obs = x0_obs + vobs * tk
            p_obs = np.array([[tk, x_obs, 0.0]]).T
            h_obs = np.hstack((h_obs, p_obs))

        h_obs[2, :] = h_obs[1, :] - h_traj[4, :]

        print(p2[0, -1], p2[1, -1], p2[2, -1], p2[3, -1], p2[4, -1])

        return h_traj, h_obs, t


ACC_CURRENT = 0.0
ACC_MIN = -1.0
JERK_MIN = -0.3
VEL_CURRENT = 20.0
VEL_TARGET = 5.0
VEL_OBS_CURRENT = 0.0

h_traj, h_obs, t = analytical_planning(VEL_CURRENT / 3.6, VEL_TARGET / 3.6, ACC_CURRENT, ACC_MIN, JERK_MIN, VEL_OBS_CURRENT / 3.6)
fig = plt.figure(figsize=(10, 15))

ax1 = plt.subplot(4, 2, 1)
lj, = plt.plot(h_traj[4, :], h_traj[1, :], '-r', lw=2)
ax1.set_xlim(h_traj[4, :].min() - 0.5, h_traj[4, :].max() + 0.5)
ax1.set_ylim(h_traj[1, :].min() - 0.5, h_traj[1, :].max() + 0.5)
ax1.set_xlabel('arclength [m]')
ax1.set_ylabel('jerk [m/sss]')
ax1.grid()

ax2 = plt.subplot(4, 2, 3)
la, = plt.plot(h_traj[4, :], h_traj[2, :], '-g', lw=2)
ax2.set_xlim(h_traj[4, :].min() - 0.5, h_traj[4, :].max() + 0.5)
ax2.set_ylim(h_traj[2, :].min() - 0.5, h_traj[2, :].max() + 0.5)
ax2.set_xlabel('arclength [m]')
ax2.set_ylabel('acceleration [m/ss]')
ax2.grid()

ax3 = plt.subplot(4, 2, 5)
lv, = plt.plot(h_traj[4, :], h_traj[3, :], '-b', lw=2)
ax3.set_xlim(h_traj[4, :].min() - 0.5, h_traj[4, :].max() + 0.5)
ax3.set_ylim(h_traj[3, :].min() - 0.5, h_traj[3, :].max() + 0.5)
ax3.set_xlabel('arclength [m]')
ax3.set_ylabel('velocity [m/s]')
ax3.grid()

ax4 = plt.subplot(4, 2, 7)
lx, = plt.plot(t, h_traj[4, :], lw=2)
ax4.set_xlim(t.min() - 0.5, t.max() + 0.5)
ax4.set_ylim(h_traj[4, :].min() - 0.5, h_traj[4, :].max() + 0.5)
ax4.set_xlabel('time [s]')
ax4.set_ylabel('arclength [m]')
ax4.grid()

ax5 = plt.subplot(4, 2, 8)
ld, = plt.plot(t, h_obs[2, :], lw=2)
ax5.set_xlim(t.min() - 0.5, t.max() + 0.5)
ax5.set_ylim(h_obs[2, :].min() - 0.5, h_obs[2, :].max() + 0.5)
ax5.set_xlabel('time [s]')
ax5.set_ylabel('arclength [m]')
ax5.grid()

plt.subplots_adjust(bottom=0.2)

axcolor = 'gold'
ax_vt = plt.axes([0.25, 0.02, 0.2, 0.01], facecolor=axcolor)
ax_jm = plt.axes([0.25, 0.04, 0.2, 0.01], facecolor=axcolor)
ax_am = plt.axes([0.25, 0.06, 0.2, 0.01], facecolor=axcolor)
ax_a0 = plt.axes([0.25, 0.08, 0.2, 0.01], facecolor=axcolor)
ax_v0 = plt.axes([0.25, 0.1, 0.2, 0.01], facecolor=axcolor)
ax_obs_v0 = plt.axes([0.25, 0.12, 0.2, 0.01], facecolor=axcolor)

sli_vt = Slider(ax_vt, 'Target Vel[km/h]', 0, 20.0, valinit=VEL_TARGET,valstep=1.0)
sli_jm = Slider(ax_jm, 'Min Jerk[m/sss]', -3.0, 0.0, valinit=JERK_MIN, valstep=0.1)
sli_am = Slider(ax_am, 'Min Acc[m/ss]', -3.0, 0.0, valinit=ACC_MIN,valstep=0.1)
sli_a0 = Slider(ax_a0, 'Current Acc[m/ss]', 0.0, 3.0, valinit=ACC_CURRENT,valstep=0.1)
sli_v0 = Slider(ax_v0, 'Current Vel[km/h]', 0.0, 20.0, valinit=VEL_CURRENT,valstep=1.0)
sli_obs_v0 = Slider(ax_obs_v0, 'Obstacle Vel[km/h]', 0.0, 20.0, valinit=VEL_OBS_CURRENT,valstep=1.0)

def update(val):
    svt = sli_vt.val
    sjm = sli_jm.val
    sam = sli_am.val
    sa0 = sli_a0.val
    sv0 = sli_v0.val
    s_obs_v0 = sli_obs_v0.val
    h_traj, h_obs, t = analytical_planning(sv0 / 3.6, svt / 3.6, sa0, sam, sjm, s_obs_v0 / 3.6)

    ax1.set_xlim(h_traj[4, :].min() - 0.5, h_traj[4, :].max() + 0.5)
    ax1.set_ylim(h_traj[1, :].min() - 0.5, h_traj[1, :].max() + 0.5)
    lj.set_xdata(h_traj[4, :])
    lj.set_ydata(h_traj[1, :])

    ax2.set_xlim(h_traj[4, :].min() - 0.5, h_traj[4, :].max() + 0.5)
    ax2.set_ylim(h_traj[2, :].min() - 0.5, h_traj[2, :].max() + 0.5)
    la.set_xdata(h_traj[4, :])
    la.set_ydata(h_traj[2, :])

    ax3.set_xlim(h_traj[4, :].min() - 0.5, h_traj[4, :].max() + 0.5)
    ax3.set_ylim(h_traj[3, :].min() - 0.5, h_traj[3, :].max() + 0.5)
    lv.set_xdata(h_traj[4, :])
    lv.set_ydata(h_traj[3, :])

    ax4.set_xlim(t.min() - 0.5, t.max() + 0.5)
    ax4.set_ylim(h_traj[4, :].min() - 0.5, h_traj[4, :].max() + 0.5)
    lx.set_xdata(t)
    lx.set_ydata(h_traj[4, :])

    ax5.set_xlim(t.min() - 0.5, t.max() + 0.5)
    ax5.set_ylim(h_obs[2, :].min() - 0.5, h_obs[2, :].max() + 0.5)
    ld.set_xdata(t)
    ld.set_ydata(h_obs[2, :])
    
    fig.canvas.draw_idle()

sli_vt.on_changed(update)
sli_jm.on_changed(update)
sli_am.on_changed(update)
sli_a0.on_changed(update)
sli_v0.on_changed(update)
sli_obs_v0.on_changed(update)

resetax = plt.axes([0.8, 0.0, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
    sli_vt.reset()
    sli_jm.reset()
    sli_am.reset()
    sli_a0.reset()
    sli_v0.reset()
    sli_obs_v0.reset()

button.on_clicked(reset)

plt.show()
