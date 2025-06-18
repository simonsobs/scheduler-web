import os
import yaml

import argparse
import numpy as np
import datetime as dt
from schedlib import utils as u, source as src
from schedlib.quality_assurance import SunCrawler
from schedlib.policies.lat import LATPolicy as Policy, State
from typing import Union

import streamlit as st
from matplotlib.backends.backend_agg import RendererAgg
from threading import RLock

logger = u.init_logger(__name__)

_lock = RLock()

schedule_base_dir = os.environ.get("LAT_SCHEDULE_BASE_DIR", 'master_schedules/')

st.title("LAT Scheduler")

st.subheader("Scheduler Parameters")
left_column, right_column = st.columns(2)

init_end_date = dt.date.today() + dt.timedelta(days=1)

with left_column:
    platform = "lat"

    start_date = st.date_input("Start date", value=dt.date.today(), key='start_date')
    end_date = st.date_input("End date", value=init_end_date, key='end_date')
    use_cal_file = st.checkbox("Use Calibration File", value=False)
    corotator = st.text_input("Corotator Angle [float, None, or Locked]", value="None")
    try:
        corotator = float(corotator)
    except ValueError:
        pass

    # cal_targets = st.text_input("Calibration Targets (comma-separated)")

    no_cmb = st.checkbox("No CMB", value=False)
    az_speed = st.number_input("Azimuth Speed (deg/s)", value=0.5)
    az_accel = st.number_input("Azimuth Acceleration (deg/s²)", value=0.25)
    iv_cadence = st.number_input("IV Cadence (seconds)", value=14400)
    relock_cadence = st.number_input("Relock Cadence (seconds)", value=86400)
    bias_step_cadence = st.number_input("Bias Step Cadence (seconds)", value=1800)
    max_cmb_scan_duration = st.number_input("Max CMB Scan Duration (seconds)", value=3600)
    cryo_stabilization_time = st.number_input("Cryo Stabilization Time", value=180)

with right_column:
    start_time = st.time_input("Start time (UTC)", value=dt.datetime.utcnow().time(), key='start_time')
    end_time = st.time_input("End time (UTC)", value=dt.datetime.utcnow().time(), key='end_time')

    az_motion_override = st.checkbox("Az Motion Override", value=False)
    apply_corotator_rotation = st.checkbox("Apply Corotator Rotation", value=False)
    elevations_under_90 = st.checkbox("Elevations Under 90", value=True)
    open_shutter = st.checkbox("Open Shutter", value=True)
    close_shutter = st.checkbox("Close Shutter", value=True)

    az_branch_override = st.number_input("Az Branch Override (deg)", value=180.0)
    allow_partial_override = st.checkbox("Allow Partial Override", value=False)
    drift_override = st.checkbox("Drift Override", value=True)
    az_offset = st.number_input("Azimuth Offset (deg)", value=0.0)
    el_offset = st.number_input("Elevation Offset (deg)", value=0.0)
    xi_offset = st.number_input("Xi Offset (deg)", value=0.0)
    eta_offset = st.number_input("Eta Offset (deg)", value=0.0)
    corotator_offset = st.number_input("Corotator Offset (deg)", value=0.0)
    # outfile = st.text_input("Output Filename")
    # cal_anchor_time = st.text_input("Calibration Anchor Time")

if "show_dropdown" not in st.session_state:
    st.session_state.show_dropdown = False

def toggle_dropdown():
    st.session_state.show_dropdown = not st.session_state.show_dropdown

st.button(
    "Custom State" if not st.session_state.show_dropdown else "Default State",
    on_click=toggle_dropdown
)

if st.session_state.show_dropdown:
    left_column_state, right_column_state = st.columns(2)

    with left_column_state:
        az_now = st.number_input("Azimuth Now", value=180.0, format="%.2f", key="az_now")
        el_now = st.number_input("Elevation Now", value=60.0, format="%.2f", key="el_now", min_value=40.0, max_value=90.0)
        az_speed_now = st.number_input("Azimuth Speed Now", value=0.0, format="%.2f", key="az_speed_now")
        az_accel_now = st.number_input("Azimuth Accel Now", value=0.0, format="%.2f", key="az_accel_now")

    with right_column_state:
        corotator_now = st.number_input("Corotator Rotation Now", value=0.0, format="%.2f", key="boresight_rot_now")
        is_det_setup = st.checkbox("Det setup", value=False)
        has_active_channels = st.checkbox("Active Channels", value=True)

if st.button('Generate Schedule'):
    t0 = dt.datetime.combine(
        start_date, start_time, tzinfo=dt.timezone.utc
    )
    t1 = dt.datetime.combine(
        end_date, end_time, tzinfo=dt.timezone.utc
    )
    schedule_file = None
    t0_state_file = None
    cal_anchor_time = None
    remove_targets = []
    cal_targets = []

    assert platform in ['lat'], (f"{platform} is not an "
            "implemented platform, you can only choose lat")

    if relock_cadence == "None":
            relock_cadence = None

    if no_cmb:
        sfile = os.path.join(
            schedule_base_dir,
            "empty_cmb.txt"
        )
    elif schedule_file is not None:
        sfile = schedule_file
    else:
        sfile = os.path.join(
            schedule_base_dir,
            'iso/phase2/2025-05-23T20:46:10+00:00_phase2_cmb_lat_field_schedule.txt'
        )

    if use_cal_file:
        cfile = os.path.join(
            schedule_base_dir,
            'iso/phase2/2025-05-22T17:29:30+00:00_calibration_lat_field_schedule.txt'
        )
    else:
        cfile = None

    if (not t0_state_file is None) and (not os.path.exists(t0_state_file)):
        print(f"Not using state file {t0_state_file} because it doesn't exist")
        t0_state_file = None


    # scheduler runs with the corotator "locked to elevation axis" if
    # corotator_override is None. But I am worried the None might get
    # confusing so we'll enable the passing of "locked" as well

    if type(corotator)==str:
        if corotator.lower() == 'locked':
            corotator = None
        elif corotator.lower() == 'none':
            corotator = None

    cfg = {
        'az_speed': az_speed,
        'az_accel': az_accel,
        'az_offset': az_offset,
        'el_offset': el_offset,
        'xi_offset': xi_offset,
        'eta_offset': eta_offset,
        'iv_cadence': iv_cadence,
        'bias_step_cadence': bias_step_cadence,
        'max_cmb_scan_duration': max_cmb_scan_duration,
        'az_motion_override': az_motion_override,
        'corotator_override': corotator,
        'apply_corotator_rot': apply_corotator_rotation,
        'cryo_stabilization_time': cryo_stabilization_time,
        'corotator_offset': corotator_offset,
        'elevations_under_90' : elevations_under_90,
        'remove_targets': tuple(remove_targets),
        'open_shutter': open_shutter,
        'close_shutter': close_shutter,
        'relock_cadence': relock_cadence,
        'az_branch_override': az_branch_override,
        'allow_partial_override': allow_partial_override,
        'drift_override': drift_override,
        'az_stow' : 180,
        'el_stow' : 60,
    }

    policy = Policy.from_defaults(
        master_file=sfile,
        state_file = t0_state_file,
        **cfg
    )
    policy.cal_targets = []
    for target in cal_targets:
        if target['source'] not in src.get_source_list():
            assert 'ra' in target and 'dec' in target, "need RA and DEC"
            src.add_fixed_source(
                name=target['source'],
                ra=target['ra'], dec=target['dec'],
                ra_units='deg'
            )
        if 'ra' in target:
            target.pop("ra")
        if 'dec' in target:
            target.pop("dec")
        if 'elevation' not in target:
            target['elevation'] = elevation
        if 'corotator' not in target:
            target['corotator'] = corotator
        policy.add_cal_target(**target)

    seq = policy.init_seqs(cfile, t0, t1)
    seq = policy.apply(seq)
    cmds, state = policy.seq2cmd(seq, t0, t1, return_state=True)
    schedule = policy.cmd2txt(cmds, t0, t1)

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

    st.code(schedule, language="text")