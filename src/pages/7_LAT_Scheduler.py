import argparse
import datetime as dt
import importlib
import os
from importlib.metadata import version
from threading import RLock
from typing import Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml
from matplotlib.backends.backend_agg import RendererAgg

_lock = RLock()

# Get gen_schedule
script_base = os.environ['SCHED_SCRIPTS_DIR']
module_path = os.path.join(script_base, "lat/gen_schedule.py")
program = "scheduler_scripts/lat/gen_schedule.py"
module_name = "_" + program.replace(".py", "").replace("/", ".")
spec = importlib.util.spec_from_file_location(module_name, module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# scheduler-configs
config_base = os.environ['SCHED_CFG_DIR']

st.title("LAT Scheduler")

try:
    schedlib_version = version("schedlib")
    st.markdown(f"**schedlib version:** `{schedlib_version}`")
except PackageNotFoundError:
    st.error("schedlib is not installed or version metadata is missing.")

st.subheader("Scheduler Parameters")
left_column, right_column = st.columns(2)

init_end_date = dt.date.today() + dt.timedelta(days=1)

cfg = {}

if "start_time" not in st.session_state:
    st.session_state.start_time = dt.datetime.utcnow().time()

if "end_time" not in st.session_state:
    st.session_state.end_time = dt.datetime.utcnow().time()

with left_column:
    platform = "lat"

    start_date = st.date_input("Start date", value=dt.date.today(), key='start_date')
    end_date = st.date_input("End date", value=init_end_date, key='end_date')
    start_time = st.time_input("Start time (UTC)", value=st.session_state.start_time, key='start_time')
    end_time = st.time_input("End time (UTC)", value=st.session_state.end_time, key='end_time')

    cfg["az_speed"] = st.number_input("Azimuth Speed (deg/s)", value=0.5)
    cfg["az_accel"] = st.number_input("Azimuth Acceleration (deg/s²)", value=0.25)
    cfg["az_offset"] = st.number_input("Azimuth Offset (deg)", value=0.0)
    cfg["el_offset"] = st.number_input("Elevation Offset (deg)", value=0.0)
    cfg["xi_offset"] = st.number_input("Xi Offset (deg)", value=0.0)
    cfg["eta_offset"] = st.number_input("Eta Offset (deg)", value=0.0)

    cfg["iv_cadence"] = st.number_input("IV Cadence (seconds)", value=14400)
    cfg["relock_cadence"] = st.number_input("Relock Cadence (seconds)", value=86400)
    cfg["bias_step_cadence"] = st.number_input("Bias Step Cadence (seconds)", value=1800)

with right_column:
    use_cal_file = st.checkbox("Use Calibration File", value=False)
    no_cmb = st.checkbox("No CMB", value=False)
    cfg["az_motion_override"] = st.checkbox("Az Motion Override", value=False)
    cfg["apply_corotator_rot"] = st.checkbox("Apply Corotator Rotation", value=False)
    cfg["elevations_under_90"] = st.checkbox("Elevations Under 90", value=True)
    cfg["open_shutter"] = st.checkbox("Open Shutter", value=True)
    cfg["close_shutter"] = st.checkbox("Close Shutter", value=True)
    cfg["drift_override"] = st.checkbox("Drift Override (Cal Sources)", value=True)
    cfg["allow_partial_override"] = st.checkbox("Allow Partial Override (Cal Sources)", value=False)

    cfg["az_branch_override"] = st.number_input("Az Branch Override (deg) (Cal Sources)", value=180.0)
    cfg["max_cmb_scan_duration"] = st.number_input("Max CMB Scan Duration (seconds)", value=3600)
    cfg["cryo_stabilization_time"] = st.number_input("Cryo Stabilization Time (seconds)", value=180)

    cfg["corotator_override"] = st.text_input("Corotator Angle [float, None, or Locked]", value="None")
    try:
        cfg["corotator_override"] = float(cfg["corotator_override"])
    except ValueError:
        cfg["corotator_override"] = None if cfg["corotator_override"].lower() == "none" else cfg["corotator_override"]

    cfg["corotator_offset"] = st.number_input("Corotator Offset (deg)", value=0.0)

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
            el_now = st.number_input("Elevation Now", value=60.0, format="%.2f", key="el_now", min_value=40.0, max_value=90.0)
            az_speed_now = st.number_input("Azimuth Speed Now", value=0.0, format="%.2f", key="az_speed_now")
            az_accel_now = st.number_input("Azimuth Accel Now", value=0.0, format="%.2f", key="az_accel_now")

        with right_column_state:
            corotator_now = st.number_input("Corotator Rotation Now", value=0.0, format="%.2f", key="boresight_rot_now")
            is_det_setup = st.checkbox("Det setup", value=False)
            has_active_channels = st.checkbox("Active Channels", value=True)

if st.button('Generate Schedule'):

    os.environ["PLATFORM_CFG_DIR"] = os.path.join(config_base, "lat/daq-lat/configs/")
    os.environ["LAT_SCHED_IN_DIR"] = os.path.join(config_base, "lat/daq-lat/ref_plans/")

    try:
        with open(os.path.join(config_base, "lat/daq-lat/configs/run_lat.yaml")) as f:
            f = os.path.expandvars(f.read())
        config = yaml.safe_load(f)
    except:
        raise ValueError(f"Failed to parse yaml config {config_path}: {e}")

    platform_cfg = yaml.safe_load(open(config["platform_cfg_path"], 'r'))

    for k, v in cfg.items():
        platform_cfg["config"][k] = v

    config["platform_cfg_path"] = platform_cfg
    config["state_mode"] = False
    config["t0"] = dt.datetime.combine(
        start_date, start_time, tzinfo=dt.timezone.utc
    )
    config["t1"] = dt.datetime.combine(
        end_date, end_time, tzinfo=dt.timezone.utc
    )

    config["save_schedule"] = False
    config["make_table"] = True
    config["save_table"] = False
    config["show_plot"] = False
    config["save_plot"] = False

    schedule, table, fig = module.main(**config)

    with st.expander("Show Observation Plot"):
        st.plotly_chart(fig, width='stretch')

    with st.expander("Show Ref Table"):
        st.dataframe(table)

    st.code(schedule, language="python", line_numbers=True, height=500)
