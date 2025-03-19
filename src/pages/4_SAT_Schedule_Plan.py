import datetime as dt
import yaml
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

import streamlit as st

from schedlib.policies.satp1 import SATP1Policy
from schedlib.policies.satp2 import SATP2Policy
from schedlib.policies.satp3 import SATP3Policy

from matplotlib.backends.backend_agg import RendererAgg
from threading import RLock

_lock = RLock()

""" How to run this in your own directory
streamlit run src/Home.py --server.address=localhost --browser.gatherUsageStats=false --server.fileWatcherType=none --server.port 8075
""";
schedule_base_dir = os.environ.get("SCHEDULE_BASE_DIR", 'master_files/')
# dictionary goes dict[elevation][sun_keepout]
schedule_files = {
    50 : {
        45: os.path.join(
            schedule_base_dir,
            '20250117_d-40,-10_e50_s0.5,0.8_a45.txt'
        ),
        49: os.path.join(
            schedule_base_dir, 
            '20250117_d-40,-10_e50_s0.5,0.8_a49.txt'
        ),
    },
    60 : {
        45: os.path.join(
            schedule_base_dir, 
            '20250117_d-40,-10_e60_s0.5,0.8_a45.txt'
        ),
        49: os.path.join(
            schedule_base_dir, 
            '20250117_d-40,-10_e60_s0.5,0.8_a49.txt'
        ),
    }
}

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

    platform = st.selectbox(
        "Platform:",
       options=["satp1", "satp2", "satp3"],
    )
    elevation = st.selectbox(
        "CMB Scan Elevation:", options=[50,60], index=1
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


if st.button('Plot Plan'):
    t0 = dt.datetime.combine(
        start_date, start_time, tzinfo=dt.timezone.utc
    )
    t1 = dt.datetime.combine(
        end_date, end_time, tzinfo=dt.timezone.utc
    )


    sfile = schedule_files[int(elevation)]
    if platform == 'satp1':
        sfile = sfile[45] # absorptive baffle runs 45 degree keepout
    elif platform in ['satp2', 'satp3']:
        sfile = sfile[49] # reflective baffle runs 49 degree keepout
    if not os.path.exists(sfile):
        raise ValueError(f"Schedule file {sfile} does not exist")
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

    seq = policy.init_seqs(t0, t1)
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