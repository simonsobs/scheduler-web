import datetime as dt
from functools import partial
import yaml
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

import streamlit as st
from streamlit_timeline import st_timeline
from streamlit_ace import st_ace
from streamlit_sortables import sort_items

from schedlib import policies, core, utils
from schedlib import rules as ru, source as src, instrument as inst
from scheduler_server.configs import get_config

from schedlib.policies.satp1 import make_geometry
from schedlib.thirdparty import SunAvoidance

from so3g.proj import quat, CelestialSightLine
from sotodlib import coords, core as todlib_core

import jax.tree_util as tu

"""
streamlit run src/Home.py --server.address=localhost --browser.gatherUsageStats=false --server.fileWatcherType=none --server.port 8075
"""

geometry = make_geometry()


basedir = '/so/home/kmharrin/software/scheduler-scripts/satp1'
schedule_files = {
    50 : os.path.join(basedir, 'master_files/cmb_2024_el50_20240423.txt'),
    60 : os.path.join(basedir, 'master_files/cmb_2024_el60_20240423.txt'),
}

array_focus = {
    0 : {
        'left' : 'ws3,ws2',
        'middle' : 'ws0,ws1,ws4',
        'right' : 'ws5,ws6',
        'bottom': 'ws1,ws2,ws6',
        'all' : 'ws0,ws1,ws2,ws3,ws4,ws5,ws6',
    },
    45 : {
        'left' : 'ws3,ws4',
        'middle' : 'ws2,ws0,ws5',
        'right' : 'ws1,ws6',
        'bottom': 'ws1,ws2,ws3',
        'all' : 'ws0,ws1,ws2,ws3,ws4,ws5,ws6',
    },
    -45 : {
        'left' : 'ws1,ws2',
        'middle' : 'ws6,ws0,ws3',
        'right' : 'ws4,ws5',
        'bottom': 'ws1,ws6,ws5',
        'all' : 'ws0,ws1,ws2,ws3,ws4,ws5,ws6',
    },
}

SOURCES = ['Moon', 'Jupiter', 'Saturn']

def tod_from_block( block, ndet=100 ):
    # pretty sure these are in degrees
    t, az, alt = block.get_az_alt()

    tod = todlib_core.AxisManager(
        todlib_core.LabelAxis('dets', ['a%02i' % i for i in range(ndet)]),
        todlib_core.OffsetAxis('samps', len(t))
    )
    tod.wrap_new('timestamps', ('samps', ))[:] = t

    bs = todlib_core.AxisManager(tod.samps)
    bs.wrap_new('az', ('samps', ))[:] = np.mod(az,360)* coords.DEG
    bs.wrap_new('el', ('samps', ))[:] = az*0 + alt * coords.DEG
    bs.wrap_new('roll', ('samps', ))[:] = az*0+block.boresight_angle*coords.DEG
    tod.wrap('boresight', bs)

    return tod

def get_focal_plane(tod):
    xid_list = []
    etad_list = []

    roll = np.mean(tod.boresight.roll)

    for waf in geometry:
        xi0, eta0 = geometry[waf]['center']
        R = geometry[waf]['radius']
        phi = np.arange(tod.dets.count) * 2*np.pi / tod.dets.count
        qwafer = quat.rotation_xieta(xi0 * coords.DEG, eta0 * coords.DEG)
        qdets = quat.rotation_xieta(R * coords.DEG * np.cos(phi),
                                    R * coords.DEG * np.sin(phi))
        if roll != 0:
            q_bore_rot = quat.euler(2, -roll * coords.DEG)
            qwafer = q_bore_rot * qwafer
            qdets = q_bore_rot * qdets

        xid, etad, _ = quat.decompose_xieta(qwafer * qdets)
        xid_list.append(xid)
        etad_list.append(etad)

    xid = np.concatenate(xid_list)
    etad = np.concatenate(etad_list)
    return xid, etad

now = dt.datetime.utcnow()
start_date = now.date()
start_time = now.time()
end_date = start_date + dt.timedelta(days=1)
end_time = start_time

if 'timing' not in st.session_state:
    st.session_state['timing'] = {}
    st.session_state['timing']['t0'] = dt.datetime.combine(
        start_date, start_time, tzinfo=dt.timezone.utc
    )
    st.session_state['timing']['t1'] = dt.datetime.combine(
        end_date, end_time, tzinfo=dt.timezone.utc
    )
    st.session_state['timing']['sun_avoid'] = 45


left_column, right_column = st.columns(2)

with left_column:
    start_date = st.date_input("Start date", value=start_date)
    end_date = st.date_input("End date", value=end_date)

    sun_avoid_angle = st.number_input(
        "Sun Avoidance Angle (deg):",
        min_value= 0,
        max_value= 90,
        value=41,
        step=1,
    )

    sun_avoid_time = st.number_input(
        "Sun Avoidance Time (min)",
        min_value= 0,
        max_value= 60,
        value=33,
        step=1,
    )

with right_column:
    start_time = st.time_input("Start time (UTC)", value=start_time)
    end_time = st.time_input("End time (UTC)", value=end_time)

sources = st.multiselect("Sources", SOURCES, SOURCES)    

if st.button('Plot Sources'):
    st.session_state['timing']['t0'] = dt.datetime.combine(
        start_date, start_time, tzinfo=dt.timezone.utc
    )
    st.session_state['timing']['t1'] = dt.datetime.combine(
        end_date, end_time, tzinfo=dt.timezone.utc
    )
    st.session_state['timing']['sun_avoid_angle'] = sun_avoid_angle
    st.session_state['timing']['sun_avoid_time'] = sun_avoid_time

    t0=st.session_state['timing']['t0']
    t1=st.session_state['timing']['t1']
    sun_avoid_angle = st.session_state['timing']['sun_avoid_angle']
    sun_avoid_time = st.session_state['timing']['sun_avoid_time']

    st.header("Source Availability")
    st.write("Lighter line indicates source is cut by sun avoidance")

    fig = plt.figure(figsize=(8,3.75))
    ax = fig.add_subplot(111)
    sun = SunAvoidance(
        min_angle=sun_avoid_angle, 
        min_sun_time=sun_avoid_time*60
    )

    for c, source in enumerate(sources):
        src_blocks = src.source_gen_seq(source.lower(), t0, t1)
        for block in src_blocks:
            t, az, alt = block.get_az_alt(time_step=30)
            plt.plot([dt.datetime.utcfromtimestamp(x) for x in t], alt, f'C{c}-', alpha=0.3)

        src_blocks = core.seq_flatten(sun.apply(src_blocks))

        for b,block in enumerate(src_blocks):
            if b == 0:
                lab=source
            else:
                lab=None
            t, az, alt = block.get_az_alt(time_step=30)
            plt.plot([dt.datetime.utcfromtimestamp(x) for x in t], 
                alt, f'C{c}-', label=lab)

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.xticks()
    plt.ylim(0,90)
    plt.legend()

    plt.xlabel("Azimuth (deg)")
    plt.ylabel("Elevation (deg)")
    ax.set_title(
        f"{t0.strftime('%Y-%m-%d %H:%M')} to "
        f"{t1.strftime('%Y-%m-%d %H:%M')}"
    )
    st.pyplot(fig)

with st.form("my data",clear_on_submit=False):

    st.title("Calibration Targets")

    col1, col2, col3 = st.columns(3)
    with col1:
        target = st.radio(
            "Array Target", 
            ["all", "left", "middle", "right", "bottom", "custom"],
            index=0,
        )
        source = st.radio(
            "Source to Scan", 
            SOURCES,
            index=0,
        )
    with col2:
        ws0 = st.checkbox("ws0", value=False)
        ws1 = st.checkbox("ws1", value=False)
        ws2 = st.checkbox("ws2", value=False)
        ws3 = st.checkbox("ws3", value=False)

        elevation = st.number_input(
            "Elevation (deg)",
            min_value = 48,
            max_value = 80,
            value=50,
            step=1,
        )
        boresight = st.number_input(
            "boresight (deg)",
            min_value = -60,
            max_value = 60,
            value=0,
            step=1,
        )

    with col3:
        ws4 = st.checkbox("ws4", value=False)
        ws5 = st.checkbox("ws5", value=False)
        ws6 = st.checkbox("ws6", value=False)

        min_scan_duration=st.number_input(
            "Minimum Scan Duration (min)",
            min_value = 0,
            max_value = 60,
            value=10,
            step=1,
        )

    run_calculation = st.form_submit_button("Calculate")
    if run_calculation:
        if target == "custom":
            arr = ['ws0', 'ws1', 'ws2', 'ws3', 'ws4', 'ws5', 'ws6']
            x = [ws0, ws1, ws2, ws3, ws4, ws5, ws6]
            target_str = ','.join( [arr[i] for i in range(len(arr)) if x[i]])
        elif target == "all":
            target_str = 'ws0,ws1,ws2,ws3,ws4,ws5,ws6'
        else:
            if int(boresight) not in [-45,0,45]:
                st.write(
                    f":red[Boresight must be -45,0, or 45 to use array focus.]"
                )
            target_str = array_focus[boresight][target]

        st.write(f"Target String is {target_str}")

        t0=st.session_state['timing']['t0']
        t1=st.session_state['timing']['t1']
        sun_avoid_angle = st.session_state['timing']['sun_avoid_angle']
        sun_avoid_time = st.session_state['timing']['sun_avoid_time']

        sun = SunAvoidance(
            min_angle=sun_avoid_angle, 
            min_sun_time=sun_avoid_time*60
        )
        min_dur_rule = ru.make_rule(
            'min-duration', **{'min_duration': min_scan_duration*60},
        )
        src_blocks = sun(src.source_gen_seq(source.lower(), t0, t1))
        #st.write(f"Source Blocks {src_blocks}")

        array_info = inst.array_info_from_query(geometry, target_str)
        ces_rule = ru.MakeCESourceScan(
            array_info=array_info,
            el_bore=elevation,
            drift=True,
            boresight_rot=boresight, 
            allow_partial=True,
        )
        scan_blocks = ces_rule(src_blocks)
        st.write(f"Scan Blocks {scan_blocks}")

        scan_blocks = core.seq_flatten(min_dur_rule(sun(scan_blocks)))
        st.write(f"Scan Blocks {scan_blocks}")

        for block in scan_blocks:
            fig = plt.figure(figsize=(5,3.75))
            ax = fig.add_subplot(111)
        
            tod = tod_from_block(block)
            xi_fp, eta_fp = get_focal_plane(tod)
            ax.scatter(xi_fp, eta_fp, c='k', alpha=0.5)

            csl = CelestialSightLine.az_el(
                tod.timestamps, tod.boresight.az, tod.boresight.el, weather='vacuum')
            ra, dec, _ = quat.decompose_lonlat(csl.Q)
            src_path = coords.planets.SlowSource.for_named_source(
                  source, tod.timestamps.mean()
            )
            ra0, dec0 = src_path.ra, src_path.dec
            ra0 = (ra0 - ra[0]) % (2 * np.pi) + ra[0]
            # Un-rotate the planet into boresight coords.
            xip, etap, _ = quat.decompose_xieta(
                ~csl.Q * quat.rotation_lonlat(ra0, dec0)
            )
            plt.plot(xip, etap, alpha=0.5)
            st.pyplot(fig)