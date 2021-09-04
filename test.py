import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

fig, ax = plt.subplots(figsize=(6,6))
plt.subplots_adjust(left=0.25, bottom=0.25)
plt.axis('equal')
plt.xlim(-16,16)
plt.ylim(-16,16)
plt.grid()
t = np.linspace(0, 2*np.pi, 100)
rad0 = 2
x0 = 0
y0 = 0
delta = 1
x = x0 + rad0 * np.sin(t)
y = y0 + rad0 * np.cos(t)
l, = plt.plot(x, y, lw=2)

axcolor = 'gold'
ax_r = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_x = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)
ax_y = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)

sli_r = Slider(ax_r, 'Radius', 1, 5, valinit=rad0, valstep=delta/2)
sli_x = Slider(ax_x, 'x center', -10, 10, valinit=x0,valstep=delta)
sli_y = Slider(ax_y, 'y center', -10, 10, valinit=y0,valstep=delta)

def update(val):
    sr = sli_r.val
    sx = sli_x.val
    sy = sli_y.val
    l.set_xdata(sr*np.sin(t)+sx)
    l.set_ydata(sr*np.cos(t)+sy)
    fig.canvas.draw_idle()

sli_r.on_changed(update)
sli_x.on_changed(update)
sli_y.on_changed(update)

resetax = plt.axes([0.8, 0.0, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
    sli_r.reset()
    sli_x.reset()
    sli_y.reset()

button.on_clicked(reset)

plt.show()
