import matplotlib as mpl
CAT = ['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300','#4a3aa7','#e34948']
mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,
 'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,
 'axes.edgecolor':'#52514e','axes.labelcolor':'#0b0b0b','xtick.color':'#52514e','ytick.color':'#52514e',
 'grid.color':'#e4e3df','grid.linewidth':0.6,'lines.linewidth':2,'savefig.dpi':600,'savefig.bbox':'tight','figure.dpi':150})
def save(fig,name):
    fig.savefig(f'figs/{name}.png'); fig.savefig(f'figs/{name}.pdf')


def map_furniture(ax, ext, length=10000, fs=6.5, corner='lr', pad_frac=0.0):
    """Segmented scale bar (0-5-10 km) on a white box in an empty corner, north arrow top-left on a white box.
    ext = [xmin, xmax, ymin, ymax]. Optionally pads the bottom of the frame so the bar never sits on data."""
    from matplotlib.patches import Rectangle, FancyBboxPatch
    xmin, xmax, ymin, ymax = ext
    if pad_frac:
        ymin = ymin - pad_frac * (ymax - ymin); ax.set_ylim(ymin, ymax)
    W, H = xmax - xmin, ymax - ymin
    h = 0.012 * H
    x0 = xmax - 0.11 * W - length if corner == 'lr' else xmin + 0.05 * W
    y0 = ymin + 0.08 * H
    ax.add_patch(FancyBboxPatch((x0 - 0.035 * W, y0 - 0.07 * H), length + 0.135 * W, 0.125 * H, boxstyle='round,pad=0,rounding_size=300',
                                fc='white', ec='#9a9893', lw=0.4, alpha=0.92, zorder=20))
    half = length / 2
    for i, c in enumerate(['k', 'white', 'k', 'white']):
        ax.add_patch(Rectangle((x0 + i * length / 4, y0), length / 4, h, fc=c, ec='k', lw=0.5, zorder=21))
    labs = [(x0, '0'), (x0 + half, f'{half/1000:g}'), (x0 + length, f'{length/1000:g}')] if fs >= 7 else [(x0, '0'), (x0 + length, f'{length/1000:g}')]
    for xx, t in labs:
        ax.text(xx, y0 - 0.008 * H, t, ha='center', va='top', fontsize=fs, zorder=22)
    ax.text(x0 + length + 0.02 * W, y0 + h / 2, 'km', ha='left', va='center', fontsize=fs, zorder=22)
    xa, ya = xmin + 0.07 * W, ymax - 0.05 * H
    ax.add_patch(FancyBboxPatch((xa - 0.035 * W, ya - 0.14 * H), 0.07 * W, 0.16 * H, boxstyle='round,pad=0,rounding_size=300',
                                fc='white', ec='#9a9893', lw=0.4, alpha=0.92, zorder=20))
    ax.annotate('', xy=(xa, ya), xytext=(xa, ya - 0.065 * H), arrowprops=dict(arrowstyle='-|>', color='k', lw=1.2, mutation_scale=9), zorder=22)
    ax.text(xa, ya - 0.08 * H, 'N', ha='center', va='top', fontsize=fs + 1.5, weight='bold', zorder=22)
