import os
import yaml

import argparse
import datetime as dt
from schedlib import utils as u
from schedlib.quality_assurance import SunCrawler

import streamlit as st
from matplotlib.backends.backend_agg import RendererAgg
from threading import RLock

logger = u.init_logger(__name__)

_lock = RLock()

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

left_column, right_column = st.columns(2)
init_end_date = dt.date.today() + dt.timedelta(days=1)

with left_column:
    start_date = st.date_input("Start date", value=dt.date.today(), key='start_date')
    end_date = st.date_input("End date", value=init_end_date, key='end_date')

    platform = st.selectbox("Platform:", options=["satp1", "satp2", "satp3"])
    elevation = st.selectbox("CMB Scan Elevation:", options=[50, 60], index=1)

    use_cal_file = st.checkbox("Use Calibration File", value=False)
    use_wiregrid_file = st.checkbox("Use Wiregrid File", value=False)

    boresight = st.number_input("Boresight (deg)", value=0.0)
    # cal_targets = st.text_input("Calibration Targets (comma-separated)")

    no_cmb = st.checkbox("No CMB", value=False)
    az_speed = st.number_input("Azimuth Speed (deg/s)", value=0.5)
    az_accel = st.number_input("Azimuth Acceleration (deg/s²)", value=0.25)
    az_offset = st.number_input("Azimuth Offset (deg)", value=0.0)
    el_offset = st.number_input("Elevation Offset (deg)", value=0.0)
    xi_offset = st.number_input("Xi Offset (deg)", value=0.0)
    eta_offset = st.number_input("Eta Offset (deg)", value=0.0)
    iv_cadence = st.number_input("IV Cadence (seconds)", value=14400)
    bias_step_cadence = st.number_input("Bias Step Cadence (seconds)", value=1800)
    min_hwp_el = st.number_input("Min HWP Elevation (deg)", value=48.0)
    max_cmb_scan_duration = st.number_input("Max CMB Scan Duration (seconds)", value=3600)

with right_column:
    start_time = st.time_input("Start time (UTC)", value=dt.datetime.utcnow().time(), key='start_time')
    end_time = st.time_input("End time (UTC)", value=dt.datetime.utcnow().time(), key='end_time')

    hwp_override = st.checkbox("HWP Override", value=False)
    az_motion_override = st.checkbox("Az Motion Override", value=False)
    home_at_end = st.checkbox("Home at End", value=False)
    relock_cadence = st.number_input("Relock Cadence (seconds)", value=86400)
    disable_hwp = st.checkbox("Disable HWP", value=False)
    if platform in ['satp1', 'satp2']:
        brake_default = True
        bore_rot = True
    elif platform in ['satp3']:
        brake_default = False
        bore_rot = False
    brake_hwp = st.checkbox("Brake HWP", value=brake_default)
    apply_boresight_rotation = st.checkbox("Apply Boresight Rotation", value=bore_rot)
    az_branch_override = st.number_input("Az Branch Override (deg)", value=180.0)
    allow_partial_override = st.checkbox("Allow Partial Override", value=False)
    drift_override = st.checkbox("Drift Override", value=True)
    wiregrid_az = st.number_input("Wiregrid Azimuth (deg)", value=180.0)
    wiregrid_el = st.number_input("Wiregrid Elevation (deg)", value=48.0)
    # outfile = st.text_input("Output Filename")
    # cal_anchor_time = st.text_input("Calibration Anchor Time")

if st.button('Generate Schedule'):
    t0 = dt.datetime.combine(
        start_date, start_time, tzinfo=dt.timezone.utc
    )
    t1 = dt.datetime.combine(
        end_date, end_time, tzinfo=dt.timezone.utc
    )

    cal_targets = []
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

    seq = policy.init_cmb_seqs(t0, t1)
    seq = policy.init_cal_seqs(cfile, wgfile, seq, t0, t1, cal_anchor_time)
    seq = policy.apply(seq)
    cmds, state = policy.seq2cmd(seq, t0, t1, return_state=True)
    schedule = policy.cmd2txt(cmds, t0, t1)

    sun_safe = False
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
