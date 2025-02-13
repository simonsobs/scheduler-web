import datetime as dt
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

import pandas as pd
import streamlit as st

from schedlib import core
from schedlib import rules as ru, source as src, instrument as inst

from schedlib.policies.lat import make_geometry
from schedlib.thirdparty import SunAvoidance

from so3g.proj import quat, CelestialSightLine
from sotodlib import coords, core as todlib_core

import jax.tree_util as tu

""" How to run this in your own directory
streamlit run src/Home.py --server.address=localhost --browser.gatherUsageStats=false --server.fileWatcherType=none --server.port 8075
""";

geometry = make_geometry()


SOURCES = [
    'Moon', 'Jupiter', 'Saturn', 'TauA', 'Uranus', 'Neptune', 'Mars', 'Table'
]

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

def plot_focal_plane(ax, tod):
    roll = np.mean(tod.boresight.roll)

    for waf in geometry:
        xi0, eta0 = geometry[waf]['center']
        R = geometry[waf]['radius']
        phi = np.arange(tod.dets.count) * 2*np.pi / tod.dets.count
        qwafer = quat.rotation_xieta(xi0 * coords.DEG, eta0 * coords.DEG)
        qdets = quat.rotation_xieta(R * coords.DEG * np.cos(phi),
                                    R * coords.DEG * np.sin(phi))
        if roll != 0:
            q_bore_rot = quat.euler(2, -roll )#* coords.DEG)
            qwafer = q_bore_rot * qwafer
            qdets = q_bore_rot * qdets

        xi_c, eta_c, _ = quat.decompose_xieta(qwafer )
        xid, etad, _ = quat.decompose_xieta(qwafer * qdets)

        ax.scatter( xid, etad, marker='.')
        ax.text( xi_c, eta_c, waf)


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


left_column, right_column = st.columns(2)

with left_column:
    start_date = st.date_input("Start date", value=start_date)
    end_date = st.date_input("End date", value=end_date)

    sun_avoid_angle = st.number_input(
        "Sun Avoidance Angle (deg):",
        min_value= 0,
        max_value= 90,
        value=30,
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

    filter_elevation = st.number_input(
        "Filter sources by maximum elevation (el max > X)",
        min_value= 0,
        max_value= 90,
        value=40,
        step=1,
    )

sources = st.multiselect("Sources", SOURCES, SOURCES)    
st.write("Tabled Source info will be added it Table is included in the selected sources.")
new_sources = pd.DataFrame(
    [
        {'name': '1', 'ra': -16.509, 'dec': 16.1484, 'add_to_plot': True} ,
        {'name': '2', 'ra': -172.722, 'dec': 2.0521, 'add_to_plot': True} ,
        {'name': '3', 'ra': -81.5836, 'dec': -21.0609, 'add_to_plot': True} ,
        {'name': '4', 'ra': -15.4751, 'dec': -27.9728, 'add_to_plot': True} ,
        {'name': '5', 'ra': 60.9739, 'dec': -36.0838, 'add_to_plot': True} ,
        {'name': '6', 'ra': -94.005, 'dec': -3.8346, 'add_to_plot': True} ,
        {'name': '7', 'ra': 111.461, 'dec': -0.9157, 'add_to_plot': True} ,
        {'name': '8', 'ra': 32.6923, 'dec': -51.0171, 'add_to_plot': True} ,
        {'name': '9', 'ra': 1.5581, 'dec': -6.3932, 'add_to_plot': True} ,
        {'name': '10', 'ra': 133.703, 'dec': 20.1085, 'add_to_plot': True} ,
        {'name': '11', 'ra': -7.6762, 'dec': -47.5054, 'add_to_plot': True} ,
        {'name': '12', 'ra': -30.474, 'dec': -15.019, 'add_to_plot': True} ,
        {'name': '13', 'ra': 67.168, 'dec': -37.9389, 'add_to_plot': True} ,
        {'name': '14', 'ra': -45.9318, 'dec': -47.2465, 'add_to_plot': True} ,
        {'name': '15', 'ra': -72.2095, 'dec': -20.1153, 'add_to_plot': True} ,
        {'name': '16', 'ra': 57.1588, 'dec': -27.8203, 'add_to_plot': True} ,
        {'name': '17', 'ra': 80.7414, 'dec': -36.4585, 'add_to_plot': True} ,
        {'name': '18', 'ra': -35.8393, 'dec': 0.6982, 'add_to_plot': True} ,
        {'name': '19', 'ra': 73.9611, 'dec': -46.2662, 'add_to_plot': True} ,
        {'name': '20', 'ra': -22.583, 'dec': -8.5485, 'add_to_plot': True} ,
        {'name': '21', 'ra': -7.3508, 'dec': -37.4104, 'add_to_plot': True} ,
        {'name': '22', 'ra': 79.9577, 'dec': -45.778, 'add_to_plot': True} ,
        {'name': '23', 'ra': 127.953, 'dec': 4.4941, 'add_to_plot': True} ,
        {'name': '24', 'ra': 75.3033, 'dec': -1.9871, 'add_to_plot': True} ,
        {'name': '25', 'ra': -105.463, 'dec': 7.691, 'add_to_plot': True} ,
    ]
)
added_sources = st.data_editor(new_sources, num_rows="dynamic")

if st.button('Plot Sources'):
    if "Table" in sources:
        sources.pop( sources.index("Table"))
        for i in range(len(added_sources.name)):
            if not added_sources.add_to_plot[i]:
                continue
            src.add_fixed_source(
                added_sources.name[i], 
                added_sources.ra[i], 
                added_sources.dec[i]
            )
            if added_sources.name[i] not in sources:
                sources.append(added_sources.name[i])
            
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
            if np.max(alt) < filter_elevation:
                print(f"Source {source} skipped, max el = {np.max(alt)}")
                continue
            plt.plot(
                [dt.datetime.utcfromtimestamp(x) for x in t], 
                alt, f'C{c%10}-', alpha=0.3
            )
        
        src_blocks = core.seq_flatten(sun.apply(src_blocks))

        for b,block in enumerate(src_blocks):
            if b == 0:
                lab=source
            else:
                lab=None
            t, az, alt = block.get_az_alt(time_step=30)
            if np.max(alt) < filter_elevation:
                print(f"Source {source} skipped, max el = {np.max(alt)}")
                continue
            plt.plot([dt.datetime.utcfromtimestamp(x) for x in t], 
                alt, f'C{c%10}-', label=lab)

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    plt.xticks()
    plt.ylim(0,90)
    plt.legend()

    #plt.xlabel("Time")
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
            ["custom", "all", "c1", "i1", "i3", "i4", "i5", "i6"],
            index=0,
        )
        source = st.radio(
            "Source to Scan", 
            SOURCES,
            index=0,
        )
    with col2:
        c1_ws0 = st.checkbox("c1_ws0", value=False)
        c1_ws1 = st.checkbox("c1_ws1", value=False)
        c1_ws2 = st.checkbox("c1_ws2", value=False)

        i1_ws0 = st.checkbox("i1_ws0", value=False)
        i1_ws1 = st.checkbox("i1_ws1", value=False)
        i1_ws2 = st.checkbox("i1_ws2", value=False)

        i3_ws0 = st.checkbox("i3_ws0", value=False)
        i3_ws1 = st.checkbox("i3_ws1", value=False)
        i3_ws2 = st.checkbox("i3_ws2", value=False)

        elevation = st.number_input(
            "Elevation (deg)",
            min_value = 30,
            max_value = 80,
            value=50,
            step=1,
        )
        corotator = st.number_input(
            "co-rotator (deg)",
            min_value = -45,
            max_value = 45,
            value=0,
            step=1,
        )

    with col3:
        i4_ws0 = st.checkbox("i4_ws0", value=False)
        i4_ws1 = st.checkbox("i4_ws1", value=False)
        i4_ws2 = st.checkbox("i4_ws2", value=False)

        i5_ws0 = st.checkbox("i5_ws0", value=False)
        i5_ws1 = st.checkbox("i5_ws1", value=False)
        i5_ws2 = st.checkbox("i5_ws2", value=False)

        i6_ws0 = st.checkbox("i6_ws0", value=False)
        i6_ws1 = st.checkbox("i6_ws1", value=False)
        i6_ws2 = st.checkbox("i6_ws2", value=False)

        min_scan_duration=st.number_input(
            "Minimum Scan Duration (min)",
            min_value = 0,
            max_value = 60,
            value= 1,
            step=1,
        )

    run_calculation = st.form_submit_button("Calculate")
    if run_calculation:
        arr = [
            'c1_ws0', 'c1_ws1', 'c1_ws2', 
            'i1_ws0', 'i1_ws1', 'i1_ws2', 
            'i3_ws0', 'i3_ws1', 'i3_ws2', 
            'i4_ws0', 'i4_ws1', 'i4_ws2', 
            'i5_ws0', 'i5_ws1', 'i5_ws2', 
            'i6_ws0', 'i6_ws1', 'i6_ws2'
        ]
        x = [
            c1_ws0, c1_ws1, c1_ws2, 
            i1_ws0, i1_ws1, i1_ws2, 
            i3_ws0, i3_ws1, i3_ws2, 
            i4_ws0, i4_ws1, i4_ws2, 
            i5_ws0, i5_ws1, i5_ws2, 
            i6_ws0, i6_ws1, i6_ws2,
        ]
        
        if target == "custom":          
            target_str = ','.join( [arr[i] for i in range(len(arr)) if x[i]])
        elif target == "all":
            target_str = ','.join( [arr[i] for i in range(len(arr))])
        elif target in ['c1', 'i1', 'i3', 'i4', 'i5', 'i6']:
            target_str = ','.join( [
                arr[i] for i in range(len(arr)) if target in arr[i] ])
        else:
            raise ValueError("how did I get here?")

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
            boresight_rot= -1*(elevation-60-corotator), 
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
            #xi_fp, eta_fp = get_focal_plane(tod)
            #ax.scatter(xi_fp, eta_fp, c='k', alpha=0.5)
            plot_focal_plane(ax, tod)

            csl = CelestialSightLine.az_el(
                tod.timestamps, tod.boresight.az, tod.boresight.el, weather='vacuum')
            ra, dec, _ = quat.decompose_lonlat(csl.Q)
            if source.lower() == 'taua':
                x = [
                    x for x in coords.planets.SOURCE_LIST if isinstance(x, tuple) and x[0] =='tauA'
                ][0]
                source = f"J{x[1]}+{x[2]}"

            src_path = coords.planets.SlowSource.for_named_source(
                  source, tod.timestamps.mean()
            )
            ra0, dec0 = src_path.ra, src_path.dec
            ra0 = (ra0 - ra[0]) % (2 * np.pi) + ra[0]
            # Un-rotate the planet into boresight coords.
            xip, etap, _ = quat.decompose_xieta(
                ~csl.Q * quat.rotation_lonlat(ra0, dec0)
            )
            ax.plot(xip, etap, alpha=0.5)
            ax.set_title( block.t0.isoformat() + f'\n{block.az} throw:{block.throw}' )
            st.pyplot(fig)