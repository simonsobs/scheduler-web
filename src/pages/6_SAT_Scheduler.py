import os
import yaml

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import argparse
import datetime as dt
from schedlib import utils as u
from schedlib.quality_assurance import SunCrawler
from schedlib.policies.sat import State

import streamlit as st
from matplotlib.backends.backend_agg import RendererAgg
from threading import RLock

logger = u.init_logger(__name__)

_lock = RLock()

def build_table(t0, t1, cfg, seq, cmds, state, platform):

    total_duration = (u.str2datetime(t1) - u.str2datetime(t0)).total_seconds()

    columns = ['#   Start Time UTC', 'Stop Time UTC', 'dur', 'dir', 'rot',  'az', 'el', 'az_speed', 'az_accel', 'name', 'tag']
    df = pd.DataFrame(columns=columns)

    skip = ['sat.preamble', 'start_time', 'move_to', 'wait_until']

    hwp_dir = state.hwp_dir
    boresight_rot_now = state.boresight_rot_now
    az_speed = state.az_speed_now
    az_accel = state.az_accel_now

    total_cmb_time = 0
    total_source_time = 0
    total_wiregrid_time = 0
    total_setup_time = 0

    for ir in cmds:
        if ir.name in skip or (ir.t1 - ir.t0).total_seconds() <= 0.01:
            continue
        if ir.block is not None:
            hwp_dir = ir.block.hwp_dir
            boresight_rot_now = ir.block.boresight_angle
            tag = ir.block.tag
            az_speed = ir.block.az_speed
            az_accel = ir.block.az_accel

            if ir.name == 'sat.cmb_scan':
                total_cmb_time += (ir.t1 - ir.t0).total_seconds()
            elif ir.name == 'sat.source_scan':
                total_source_time += (ir.t1 - ir.t0).total_seconds()
            elif ir.name == 'sat.wiregrid':
                total_wiregrid_time += (ir.t1 - ir.t0).total_seconds()
            else:
                total_setup_time += (ir.t1 - ir.t0).total_seconds()

        row = {'#   Start Time UTC': ir.t0, 'Stop Time UTC': ir.t1, 'dur': (ir.t1 - ir.t0).total_seconds(), 'dir': hwp_dir, 'rot': boresight_rot_now,
                'az': np.round(ir.block.az,2), 'el': np.round(ir.block.alt,2),
                'az_speed': az_speed, 'az_accel': az_accel, 'name': ir.name, 'tag': tag}

        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    names = sorted(df['name'].unique())
    cmap = plt.get_cmap('tab10', len(names))
    name_colors = {name: cmap(i) for i, name in enumerate(names)}

    fig, ax = plt.subplots(figsize=(12, 6))

    for _, row in df.iterrows():
        ax.barh(
            y=0,
            width=row['Stop Time UTC'] - row['#   Start Time UTC'],
            left=row['#   Start Time UTC'],
            height=0.4,
            color=name_colors[row['name']],
            edgecolor=name_colors[row['name']],
            label=row['name']
        )

    ax.axvline(t0, color='black', linestyle='-', linewidth=1.5)
    ax.axvline(t1, color='black', linestyle='-', linewidth=1.5)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), title='Name', loc='upper right')

    ax.set_yticks([])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    plt.xticks(rotation=45)
    plt.title(f"CMB: {np.round(100*total_cmb_time/total_duration,0)}% | Cal: {np.round(100*total_source_time/total_duration,0)}% | WG: "
                f"{np.round(100*total_wiregrid_time/total_duration,0)}% | Setup: {np.round(100*total_setup_time/total_duration,0)}% | Other: "
                f"{np.round(100 - 100*(total_setup_time + total_cmb_time + total_source_time + total_wiregrid_time)/total_duration,0)}%")
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()

    return fig

schedule_base_dir = os.environ.get("SCHEDULE_BASE_DIR", 'master_schedules/')
# dictionary goes dict[elevation][sun_keepout]
schedule_files = {
    50 : {
        45: os.path.join(schedule_base_dir, '20250411_d-40,-10_e50_s0.5,0.8_a45_j2025-02-15T12:00+00:00_n365.txt'),
        49: os.path.join(schedule_base_dir, '20250411_d-40,-10_e50_s0.5,0.8_a49_j2025-02-15T12:00+00:00_n365.txt'),
    },
    60 : {
        45: os.path.join(schedule_base_dir, '20250411_d-40,-10_e60_s0.5,0.8_a45_j2025-02-15T12:00+00:00_n365.txt'),
        49: os.path.join(schedule_base_dir, '20250411_d-40,-10_e60_s0.5,0.8_a49_j2025-02-15T12:00+00:00_n365.txt'),
    }
}

cal_files = {
    50 : {
        45: os.path.join(schedule_base_dir, '20250411_d-40,-10_e50_s0.5,0.8_a45_j2025-02-15T12:00+00_:00_n365_planets.txt'),
        49: os.path.join(schedule_base_dir, '20250411_d-40,-10_e50_s0.5,0.8_a45_j2025-02-15T12:00+00:00_n365_planets.txt'),
    },
    60 : {
        45: os.path.join(schedule_base_dir, '20250411_d-40,-10_e60_s0.5,0.8_a45_j2025-02-15T12:00+00:00_n365_planets.txt'),
        49: os.path.join(schedule_base_dir, '20250411_d-40,-10_e60_s0.5,0.8_a49_j2025-02-15T12:00+00:00_n365_planets.txt'),
    }
}

wiregrid_files = {
    50 : {
        45: os.path.join(schedule_base_dir, '20250411_d-40,-10_e50_s0.5,0.8_a45_j2025-02-15T12:00+00:00_n365_wiregrid.txt'),
        49: os.path.join(schedule_base_dir, '20250411_d-40,-10_e50_s0.5,0.8_a45_j2025-02-15T12:00+00:00_n365_wiregrid.txt'),
    },
    60 : {
        45: os.path.join(schedule_base_dir, '20250411_d-40,-10_e60_s0.5,0.8_a45_j2025-02-15T12:00+00:00_n365_wiregrid.txt'),
        49: os.path.join(schedule_base_dir, '20250411_d-40,-10_e60_s0.5,0.8_a49_j2025-02-15T12:00+00:00_n365_wiregrid.txt'),
    }
}

st.title("SAT Scheduler")

st.subheader("Scheduler Parameters")
left_column, right_column = st.columns(2)

init_end_date = dt.date.today() + dt.timedelta(days=1)

if "start_time" not in st.session_state:
    st.session_state.start_time = dt.datetime.utcnow().time()

if "end_time" not in st.session_state:
    st.session_state.end_time = dt.datetime.utcnow().time()

with left_column:
    start_date = st.date_input("Start date", value=dt.date.today(), key='start_date')
    end_date = st.date_input("End date", value=init_end_date, key='end_date')
    start_time = st.time_input("Start time (UTC)", value=st.session_state.start_time, key='start_time')
    end_time = st.time_input("End time (UTC)", value=st.session_state.end_time, key='end_time')

    platform = st.selectbox("Platform:", options=["satp1", "satp2", "satp3"])
    elevation = st.selectbox("CMB Scan Elevation:", options=[50, 60], index=1)

    iv_cadence = st.number_input("IV Cadence (seconds)", value=14400)
    relock_cadence = st.number_input("Relock Cadence (seconds)", value=86400)
    bias_step_cadence = st.number_input("Bias Step Cadence (seconds)", value=1800)

    az_speed = st.number_input("Azimuth Speed (deg/s)", value=0.5)
    az_accel = st.number_input("Azimuth Acceleration (deg/s²)", value=0.25)
    min_hwp_el = st.number_input("Min HWP Elevation (deg)", value=48.0)
    max_cmb_scan_duration = st.number_input("Max CMB Scan Duration (seconds)", value=3600)

    boresight = st.number_input("Boresight (deg)", value=0.0)
    az_branch_override = st.number_input("Az Branch Override (deg) (Cal Sources)", value=180.0)

with right_column:
    no_cmb = st.checkbox("No CMB", value=False)
    use_cal_file = st.checkbox("Use Calibration File", value=False)
    use_wiregrid_file = st.checkbox("Use Wiregrid File", value=False)

    hwp_override = st.checkbox("HWP Override", value=False)
    az_motion_override = st.checkbox("Az Motion Override", value=False)
    home_at_end = st.checkbox("Home at End", value=False)
    disable_hwp = st.checkbox("Disable HWP", value=False)

    if platform in ['satp1', 'satp2']:
        brake_default = True
        bore_rot = True
    elif platform in ['satp3']:
        brake_default = False
        bore_rot = False

    brake_hwp = st.checkbox("Brake HWP", value=brake_default)
    if platform in ["satp1", "satp2"]:
        apply_boresight_rotation = st.checkbox("Apply Boresight Rotation", value=bore_rot)
    elif platform in ['satp3']:
        apply_boresight_rotation = False

    drift_override = st.checkbox("Drift Override (Cal Sources)", value=True)
    allow_partial_override = st.checkbox("Allow Partial Override (Cal Sources)", value=False)

    wiregrid_az = st.number_input("Wiregrid Azimuth (deg)", value=180.0)
    wiregrid_el = st.number_input("Wiregrid Elevation (deg)", value=48.0, min_value=48.0)
    az_offset = st.number_input("Azimuth Offset (deg)", value=0.0)
    el_offset = st.number_input("Elevation Offset (deg)", value=0.0)
    xi_offset = st.number_input("Xi Offset (deg)", value=0.0)
    eta_offset = st.number_input("Eta Offset (deg)", value=0.0)

    # outfile = st.text_input("Output Filename")
    # cal_anchor_time = st.text_input("Calibration Anchor Time")

left_column_outer, right_column_outer = st.columns(2)

with left_column_outer:
    if "show_cal_target_dropdown" not in st.session_state:
        st.session_state.show_cal_target_dropdown = False
        cal_targets = []

    def toggle_dropdown():
        st.session_state.show_cal_target_dropdown = not st.session_state.show_cal_target_dropdown

    st.button(
        "Add Cal Targets" if not st.session_state.show_cal_target_dropdown else "Adding Cal Targets",
        on_click=toggle_dropdown
    )

    if st.session_state.show_cal_target_dropdown:
        yaml_input = st.text_area("Cal Targets", value="", height=200)
        cal_targets = yaml.safe_load(yaml_input)
    else:
        cal_targets = []

with right_column_outer:
    if "show_state_dropdown" not in st.session_state:
        st.session_state.show_state_dropdown = False

    def toggle_dropdown():
        st.session_state.show_state_dropdown = not st.session_state.show_state_dropdown

    st.button(
        "Custom State" if not st.session_state.show_state_dropdown else "Default State",
        on_click=toggle_dropdown
    )

    if st.session_state.show_state_dropdown:
        left_column_state, right_column_state = st.columns(2)

        with left_column_state:
            az_now = st.number_input("Azimuth Now", value=180.0, format="%.2f", key="az_now")
            el_now = st.number_input("Elevation Now", value=48.0, format="%.2f", key="el_now", min_value=40.0, max_value=90.0)
            az_speed_now = st.number_input("Azimuth Speed Now", value=0.0, format="%.2f", key="az_speed_now")
            az_accel_now = st.number_input("Azimuth Accel Now", value=0.0, format="%.2f", key="az_accel_now")

        with right_column_state:
            if platform in ["satp1", "satp2"]:
                boresight_rot_now = st.number_input("Boresight Rotation Now", value=0.0, format="%.2f", key="boresight_rot_now")
            elif platform in ["satp3"]:
                boresight_rot_now = 0.0
            is_det_setup = st.checkbox("Det setup", value=False)
            has_active_channels = st.checkbox("Active Channels", value=True)
            hwp_spinning = st.checkbox("HWP Spinning", value=False)
            hwp_dir = st.radio("HWP Direction", options=["None", "Forward", "Reverse"], index=0)

        if hwp_dir == "None":
            hwp_dir_val = None
        else:
            hwp_dir_val = hwp_dir.lower() == "forward"

if st.button('Generate Schedule'):
    t0 = dt.datetime.combine(
        start_date, start_time, tzinfo=dt.timezone.utc
    )
    t1 = dt.datetime.combine(
        end_date, end_time, tzinfo=dt.timezone.utc
    )

    #cal_targets = []
    t0_state_file = None
    cal_anchor_time = None

    match platform:
        case "satp1":
            from schedlib.policies.satp1 import SATP1Policy as Policy
        case "satp2":
            from schedlib.policies.satp2 import SATP2Policy as Policy
        case "satp3":
            from schedlib.policies.satp3 import SATP3Policy as Policy

    if no_cmb:
        sfile = os.path.join(schedule_base_dir, "empty_cmb.txt")
    else:
        sfile = schedule_files[int(elevation)]
        if use_cal_file:
            cfile = cal_files[int(elevation)]
        else:
            cfile = None
        if use_wiregrid_file:
            wgfile = wiregrid_files[int(elevation)]
        else:
            wgfile = None
        if platform == 'satp1':
            sfile = sfile[45] # absorptive baffle runs 45 degree keepout
            if use_cal_file:
                cfile = cfile[45]
            if use_wiregrid_file:
                wgfile = wgfile[45]
        elif platform in ['satp2', 'satp3']:
            sfile = sfile[49] # reflective baffle runs 49 degree keepout
            if use_cal_file:
                cfile = cfile[49]
            if use_wiregrid_file:
                wgfile = wgfile[49]
        if not os.path.exists(sfile):
            raise ValueError(f"Schedule file {sfile} does not exist")
        if use_cal_file and not os.path.exists(cfile):
            raise ValueError(f"Cal file {sfile} does not exist")

    if (not t0_state_file is None) and (not os.path.exists(t0_state_file)):
        print(f"Not using state file {t0_state_file} because it doesn't exist")
        t0_state_file = None

    if relock_cadence == "None":
        relock_cadence = None
    cfg = {
        'az_speed': az_speed,
        'az_accel': az_accel,
        'az_offset': az_offset,
        'el_offset': el_offset,
        'xi_offset': xi_offset,
        'eta_offset': eta_offset,
        'iv_cadence': iv_cadence,
        'bias_step_cadence': bias_step_cadence,
        'min_hwp_el': min_hwp_el,
        'max_cmb_scan_duration': max_cmb_scan_duration,
        'disable_hwp': disable_hwp,
        'brake_hwp': brake_hwp,
        'apply_boresight_rot': apply_boresight_rotation,
        'boresight_override': boresight,
        'hwp_override': hwp_override,
        'az_motion_override': az_motion_override,
        'home_at_end': home_at_end,
        'relock_cadence': relock_cadence,
        'az_branch_override': az_branch_override,
        'allow_partial_override': allow_partial_override,
        'drift_override': drift_override,
        'wiregrid_az': wiregrid_az,
        'wiregrid_el': wiregrid_el,
    }

    policy = Policy.from_defaults(
        master_file=sfile,
        state_file = t0_state_file,
        **cfg
    )

    policy.cal_targets = []
    for target in cal_targets:
        tb = target.get('boresight', None)
        if tb is None:
            target['boresight'] = boresight
        policy.add_cal_target(**target)

    if not st.session_state.show_state_dropdown:
        init_state = policy.init_state(t0)
    else:
        init_state = State(
            curr_time=t0,
            az_now=az_now,
            el_now=el_now,
            az_speed_now=az_speed_now,
            az_accel_now=az_accel_now,
            boresight_rot_now=boresight_rot_now,
            hwp_spinning=hwp_spinning,
            hwp_dir=hwp_dir,
            is_det_setup=is_det_setup,
            has_active_channels=has_active_channels
        )

    seq = policy.init_cmb_seqs(t0, t1)
    seq = policy.init_cal_seqs(cfile, wgfile, seq, t0, t1, cal_anchor_time)
    seq = policy.apply(seq)
    cmds, state = policy.seq2cmd(seq, t0, t1, state=init_state, return_state=True)
    schedule = policy.cmd2txt(cmds, t0, t1, state=init_state)

    sun_safe = True
    try:
        ## check sun safety
        sc = SunCrawler(platform, cmd_txt=schedule, az_offset=az_offset, el_offset=el_offset)
        sc.step_thru_schedule()
    except Exception as e:
        print("SCHEDULE NOT SUN SAFE")
        sun_safe=False

    if not sun_safe:
        st.error("SunCrawer found the schedule is not Sun Safe")

    fig = build_table(t0, t1, cfg, seq, cmds, init_state, platform)
    st.pyplot(fig)

    st.code(schedule, language="text", line_numbers=True)
