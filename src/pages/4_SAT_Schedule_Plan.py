import math
import datetime as dt
import yaml
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FixedLocator
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

import streamlit as st

from schedlib.policies.satp1 import SATP1Policy
from schedlib.policies.satp2 import SATP2Policy
from schedlib.policies.satp3 import SATP3Policy
from schedlib.instrument import CalTarget, parse_cal_targets_from_toast_sat
from dataclasses import replace

from matplotlib.backends.backend_agg import RendererAgg
from threading import RLock

_lock = RLock()

""" How to run this in your own directory
streamlit run src/Home.py --server.address=localhost --browser.gatherUsageStats=false --server.fileWatcherType=none --server.port 8075
""";

def plot_colortable(colors, ax, *, ncols=4, sort_colors=True):
    """
    Plot a color table into a given Matplotlib Axes.

    Parameters
    ----------
    colors : dict
        Mapping of color names to color values.
    ax : matplotlib.axes.Axes
        The Axes object where the color table will be drawn.
    ncols : int, optional
        Number of columns in the color table (default: 4).
    sort_colors : bool, optional
        If True, sort colors by HSV value (default: True).
    """

    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12

    # Sort colors if requested
    if sort_colors:
        names = sorted(
            colors, key=lambda c: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(c))))
    else:
        names = list(colors)

    n = len(names)
    nrows = math.ceil(n / ncols)

    # Configure the passed-in axes
    ax.set_xlim(0, cell_width * ncols)
    ax.set_ylim(cell_height * (nrows - 0.5), -cell_height / 2.0)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, name, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')

        ax.add_patch(
            Rectangle(xy=(swatch_start_x, y-9), width=swatch_width,
                      height=18, facecolor=colors[name], edgecolor='0.7')
        )

target_colors = {
    'jupiter':'C0',
    'saturn':'C1',
    'taua':'C2',
    'moon':'C3',
}

def day_end(day):
    day = dt.datetime(day.year,day.month,day.day, tzinfo=dt.timezone.utc)
    return day + dt.timedelta(days=1)
def day_start(day):
    return dt.datetime(day.year,day.month,day.day, tzinfo=dt.timezone.utc)

schedule_base_dir = os.environ.get("SAT_SCHED_IN_DIR", '.')
platform_cfg_dir = os.environ.get("PLATFORM_CFG_DIR", '.')

now = dt.datetime.utcnow()
init_start_date = now.date()
init_end_date = init_start_date + dt.timedelta(days=7)

left_column, right_column = st.columns(2)

with left_column:
    start_date = st.date_input(
        "Start date", value="today", key='start_date',
    )
    end_date = st.date_input("End date", value=init_end_date,
        key='end_date',
    )

    run_config = st.selectbox(
        "Run Config:",
        options=[f for f in os.listdir(platform_cfg_dir) if ".yaml" in f],
    )


with right_column:
    start_time = st.time_input("Start time (UTC)",
        value="now",
        key='start_time'
    )
    end_time = st.time_input("End time (UTC)",
        value="now",
        key='end_time'
    )

if st.button('Plot Plans'):
    t0 = dt.datetime.combine(
        start_date, start_time, tzinfo=dt.timezone.utc
    )
    t1 = dt.datetime.combine(
        end_date, end_time, tzinfo=dt.timezone.utc
    )

    cfgs = yaml.safe_load( open(os.path.join(platform_cfg_dir,run_config), "r"))
    sfile = os.path.expandvars(cfgs['cmb_plan']) 
    cfile = os.path.expandvars(cfgs['cal_plan']) 

    platform = os.path.expandvars(cfgs['platform'])

    print(f"using schedule file {sfile}")
    t0_state_file = None

    match platform:
        case "satp1":
            Policy = SATP1Policy
        case "satp2":
            Policy = SATP2Policy
        case "satp3":
            Policy = SATP3Policy

    cfg = {'apply_boresight_rot': platform != "satp3", }
    policy = Policy.from_defaults(
        master_file=sfile,
        state_file = None,
        **cfg
    )

    seq = policy.init_cmb_seqs(t0, t1)
    seq = policy.init_cal_seqs(None, None, seq, t0, t1)
    seq = policy.apply(seq)

    data = np.zeros( (0,6))
    def block_to_arr(block):
        return np.array([
            block.t0, block.alt,
            block.boresight_angle,
            block.hwp_dir, block.az_speed,
            block.az_accel
        ])


    data = np.vstack( (data, block_to_arr(seq[0])) )
    for block in seq:
        if np.all( data[-1,1:]==block_to_arr(block)[1:]  ):
            continue
        data = np.vstack( (data, block_to_arr(block)) )

    df = pd.DataFrame(
        data, columns=['Datetime', 'Elevation', 'Boresight', 'HWP Direction', 'Scan Speed', 'Scan Accel'],
    )
    st.table(df)

    all_targets = parse_cal_targets_from_toast_sat(cfile)
    targets = []
    for target in all_targets:
        target = replace(
            target, 
            t0=target.t0.astimezone(dt.timezone.utc),
            t1=target.t1.astimezone(dt.timezone.utc),
        )
        if target.t1 <= t0-dt.timedelta(days=1):
            continue
        if target.t0 >= t1+dt.timedelta(days=1):
            continue

        targets.append(target)
    print(f"found {len(targets)} targets this week")

    n_days = int((t1-t0).days)
    date_bins = t0.date() + np.arange(n_days)*dt.timedelta(days=1)

    with _lock:
        fig = plt.figure(figsize=(8,8))
        ax1 = fig.add_subplot(5,1,1)
        ax2 = fig.add_subplot(5,1,2)
        ax3 = fig.add_subplot(5,1,3)
        ax4 = fig.add_subplot(5,1,4)
        ax5 = fig.add_subplot(5,1,5)

        for block in seq:
            tt = (block.t1-block.t0).total_seconds()
            ax1.fill_between(
                [block.t0, block.t1],
                [block.az+block.throw, block.az+block.throw+block.az_drift*tt],
                [block.az, block.az+block.az_drift*tt],
            )
            ax2.plot( [block.t0, block.t1], [block.alt, block.alt] )
            ax3.plot(
                [block.t0, block.t1],
                [block.boresight_angle, block.boresight_angle]
            )
            ax4.plot(
                [block.t0, block.t1],
                [block.az_speed, block.az_speed]
            )
            ax5.plot(
                [block.t0, block.t1],
                [block.hwp_dir, block.hwp_dir] , lw=2
            )

        vlines = []
        l = t0
        while l <= t1:
            vlines.append(l)
            l += dt.timedelta(hours=24)

        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        for ax in [ax1, ax2, ax3, ax4, ax5]:
            y0,y1 = ax.get_ylim()
            ax.vlines(vlines, ymin=y0, ymax=y1, ls='--', color='k')
            ax.set_ylim(y0,y1)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)

        ax1.set_ylabel("Azimuth")
        ax2.set_ylabel("Elevation")
        ax2.set_ylim(30,70)
        ax3.set_ylabel("Boresight")
        ax3.set_ylim(-50,50)
        ax4.set_ylabel("Scan Speed")
        ax4.set_ylim(0.25,1.0)
        ax5.set_ylabel("HWP Dir")
        ax5.set_ylim(-0.1, 1.1)
        fig.suptitle(t0)

        st.pyplot(fig)

        fig, (ax2, ax) = plt.subplots(
            2, 1, 
            figsize=(10, 6),  # Set figure size (width, height)
            gridspec_kw={'height_ratios': [1, 10]} 
        )
        
        for target in targets:
            y = np.where( target.t0.date() == date_bins)[0]
            if len(y) == 0:
                #print("start of target not this week")
                continue
            y=y[0]
            end = np.min( [target.t1, day_end(date_bins[y])])
            ax.barh( 
                y=y, height=[1], 
                width=(end-target.t0).total_seconds()/3600,
                left=target.t0.hour+target.t0.minute/60,
                color=target_colors[target.source],ec='k',
            )

            ax.text( target.t0.hour+(target.t0.minute+5)/60, y, target.array_query)
            
            y2 = np.where( target.t1.date() == date_bins)[0]
            if len(y2)==0:
                ## ends off the week
                continue
            elif y2[0] != y:
                ## ends the next day
                y=y2[0]
                ax.barh( 
                    y=y, height=[1], 
                    width=(target.t1-day_start(date_bins[y])).total_seconds()/3600,
                    left=0,color=target_colors[target.source],ec='k',
                )
                

        ax.set_ylim(6.5, -0.5)
        ax.set_xlim(0,24)
        ax.set_xlabel("Time of Day (UTC)", size='large')

        ax.set_yticks( np.arange(n_days))
        ax.set_yticklabels( date_bins)
        ax.set_yticks( np.linspace(0.5,n_days-0.5,n_days-1) , minor=True)

        ax.xaxis.grid(
            True, which='major', linestyle='-', 
            linewidth=1.0, color='lightgray'
        )
        ax.yaxis.grid(
            True, which='minor', linestyle='-', 
            linewidth=1.0, color='lightgray'
        )

        plot_colortable(target_colors, ax2, ncols=4, sort_colors=False)
        fig.suptitle(f"Calibration Target Plan for Week of {t0.isoformat()}")
        fig.tight_layout()
        st.pyplot(fig)