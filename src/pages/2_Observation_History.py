import numpy as np
import datetime as dt

import streamlit as st
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

from sotodlib import core

colors = {
    'idle': (1,1,1),
    'oper': (0.75, 0.75, 0),
    'cmb': (0, 0.5, 0),
    'jupiter_targeted':  np.array([5, 46, 252])/255., ## jupiter light
    'jupiter': np.array([3, 24, 130])/255., ## jupiter dark
    'moon_targeted': np.array([197, 7, 240])/255., ## moon bright
    'moon': np.array([107, 3, 130])/255., ## moon dark
    'saturn': np.array([252, 186, 3])/255.,
    'tauA': np.array([240, 99, 12])/255.,
    'calibration_other': (0, 0, 0),
    'streaming_other': (1, 0, 0),
}

def get_color_for_obs(ctx, obs, wafer, tube=None):
    """
    only send in tube for the LAT
    """
    target = wafer
    if tube is not None:
        target = f"{tube}_{wafer}"
    if obs['type'] == 'oper':
        my_color = colors['oper']
    elif obs['subtype'] == 'cmb':
        my_color = colors['cmb']
    elif obs['subtype'] == 'cal':
        tags = ctx.obsdb.get(obs['obs_id'], tags=True)['tags']
        if 'jupiter' in tags:
            if target in tags:
                my_color = colors['jupiter_targeted']
            else:
                my_color = colors['jupiter']
        elif 'moon' in tags:
            if target in tags:
                my_color = colors['moon_targeted']
            else:
                my_color = colors['moon']
        elif 'saturn' in tags:
            my_color = colors['saturn']
        elif 'taua' in tags:
            my_color = colors['tauA']
        else:
            my_color = colors['calibration_other']
    else:
        my_color = colors['streaming_other']
    return my_color

def plot_colortable(colors, *, ncols=4, sort_colors=True):
    """taken straight from a matplotlib example"""
    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12

    # Sort colors by hue, saturation, value and name.
    if sort_colors is True:
        names = sorted(
            colors, key=lambda c: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(c))))
    else:
        names = list(colors)

    n = len(names)
    nrows = int(np.ceil(n / ncols))

    width = cell_width * ncols + 2 * margin
    height = cell_height * nrows + 2 * margin
    dpi = 72

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin/width, margin/height,
                        (width-margin)/width, (height-margin)/height)
    ax.set_xlim(0, cell_width * ncols)
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
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

    return fig

def plot_week_sat(ctx, start_dt, stop_dt=None ):
    wafers = ['ws0', 'ws1', 'ws2', 'ws3', 'ws4', 'ws5', 'ws6']

    start = start_dt.timestamp()
    if stop_dt is None:
        stop_dt = start_dt+dt.timedelta(days=7)
    stop = stop_dt.timestamp()
    obs_list = ctx.obsdb.query(
        f"timestamp >= {start} and "
        f"timestamp < {stop}"
    )
    
    times = np.linspace(start, stop, int((stop-start)/300)+1)
    status = np.ones( (len(wafers), len(times), 3 ))
    
    for w, wafer in enumerate(wafers):
        obs_with_wafer = [obs for obs in obs_list if wafer in obs['wafer_slots_list']]
        
        for obs in obs_with_wafer:
            my_color = get_color_for_obs(ctx, obs, wafer)
            tmsk = np.all( [times >= obs['start_time'], times < obs['stop_time']], axis=0)
            status[w][tmsk] = my_color
            
    fig = plt.figure(figsize=(12,2.0))
    plt.imshow(status, origin='lower', aspect='auto', interpolation='nearest',
          extent=[dt.datetime.utcfromtimestamp(start), dt.datetime.utcfromtimestamp(stop), -0.5, 6.5])
    plt.yticks(np.arange(len(wafers)), wafers)
    return fig

def plot_week_lat(ctx, start_dt, stop_dt=None ):
    
    optics_tubes = ['c1', 'i1', 'i3', 'i4', 'i5', 'i6']
    wafers = ['ws0', 'ws1', 'ws2']
    tot_wafers = len(wafers)*len(optics_tubes)
    
    start = start_dt.timestamp()
    if stop_dt is None:
        stop_dt = start_dt+dt.timedelta(days=7)
    stop = stop_dt.timestamp()
    
    times = np.linspace(start, stop, int((stop-start)/300)+1)
    status = np.ones( (tot_wafers, len(times), 3 ))
    labels = []

    for t, tube in enumerate(optics_tubes):
        obs_list = ctx.obsdb.query(
            f"timestamp >= {start} and "
            f"timestamp < {stop} and tube_slot == '{tube}'"
        )
        
        for w, wafer in enumerate(wafers):
            labels.append( f"{tube}_{wafer}")
            obs_with_wafer = [obs for obs in obs_list if wafer in obs['wafer_slots_list']]

            for obs in obs_with_wafer:
                my_color = get_color_for_obs(ctx, obs, wafer, tube)
                
                tmsk = np.all( [times >= obs['start_time'], times < obs['stop_time']], axis=0)
                status[int(3*t+w)][tmsk] = my_color
            
        
    fig = plt.figure(figsize=(12,5.0))
    plt.imshow(status, origin='lower', aspect='auto', interpolation='nearest',
          extent=[dt.datetime.utcfromtimestamp(start), dt.datetime.utcfromtimestamp(stop), -0.5, tot_wafers-0.5])
    for y in np.arange(len(optics_tubes))*3:
        plt.hlines(y-0.5, color='k', 
                   xmin=dt.datetime.utcfromtimestamp(start), 
                   xmax=dt.datetime.utcfromtimestamp(stop)
        )
    plt.yticks(np.arange(tot_wafers), labels, )
    
    return fig

now = dt.datetime.utcnow()
init_start_date = (now - dt.timedelta(days=7)).date() 
init_end_date = now.date()

left_column, right_column = st.columns(2)
all_platforms = ["satp1", "satp2", "satp3", "lat"]

st.title("Observation History")
with left_column:
    start_date = st.date_input(
        "Start date", value=init_start_date, key='start_date',
    )
    end_date = st.date_input("End date", value=init_end_date,
        key='end_date',
    )
    
    platforms = st.multiselect("Platforms", all_platforms, all_platforms)    


with right_column:
    start_time = st.time_input("Start time (UTC)", 
        value="now",       
        key='start_time'
    )
    end_time = st.time_input("End time (UTC)", 
        value="now", 
        key='end_time'
    )

st.write(
    """Note: book binding and obsdb building can lag by up to six hours during
    normal operations. Lags beyond that can indicate issues with data packaging."""
)
if st.button('Plot Observations'):
    t0 = dt.datetime.combine(
        start_date, start_time, tzinfo=dt.timezone.utc
    )
    t1 = dt.datetime.combine(
        end_date, end_time, tzinfo=dt.timezone.utc
    )
    fig = plot_colortable(colors, ncols=4, sort_colors=False)
    st.pyplot(fig)
    for platform in platforms:
        ctx = core.Context(f"/so/metadata/{platform}/contexts/basic.yaml")
        temp_t0 = t0
        temp_t1 = t0 + dt.timedelta(days=7)
        if temp_t1 > t1:
            temp_t1 = t1
        while temp_t0 < t1:
            if "sat" in platform:
                fig = plot_week_sat(ctx, temp_t0, temp_t1 )
            elif "lat" in platform:
                fig = plot_week_lat(ctx, temp_t0, temp_t1)
            fig.suptitle(f"{platform}")
            st.pyplot(fig)
            temp_t0 += dt.timedelta(days=7)
            temp_t1 += dt.timedelta(days=7)
            if temp_t1 > t1:
                temp_t1 = t1