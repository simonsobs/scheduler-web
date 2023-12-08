import datetime as dt
from functools import partial
import yaml
import pandas as pd

import streamlit as st
from streamlit_timeline import st_timeline
from streamlit_ace import st_ace
from streamlit_sortables import sort_items

from schedlib import policies, core, utils, source as src
from scheduler_server.configs import get_config

import jax.tree_util as tu

SOURCES = ['moon', 'mercury', 'venus', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune']

# ====================
# utility functions
# ====================

def seq2visdata(seqs):
    # make group
    def block2group(path, block):
        key = utils.path2key(path)
        if key == '': key = 'root'
        return {
            'id': key, 
            'content': key
        }
    groups = tu.tree_leaves(
        tu.tree_map_with_path(
            block2group,
            seqs, 
            is_leaf=lambda x: isinstance(x, list)
        ),
        is_leaf=lambda x: 'id' in x,
    )

    # make items
    def block2item(block, group=""):
        res = {
            'id': hash(block),
            'content': block.name,
            'start': block.t0.isoformat(),
            'end': block.t1.isoformat(),
        }
        if group != "": res['group'] = group
        return res
    items = tu.tree_leaves(
        tu.tree_map_with_path(
            lambda path, x: core.seq_map(
                partial(block2item, group=utils.path2key(path)), 
                core.seq_sort(x, flatten=True)
            ),
            seqs, 
            is_leaf=lambda x: isinstance(x, list)
        ),
        is_leaf=lambda x: 'id' in x
    )
    return items, groups

# ====================
# initialize session state
# ====================

if 'user_config_str' not in st.session_state:
    st.session_state.user_config_str = "{}"
    
if 'commands' not in st.session_state:
    st.session_state.commands = ""

if 'checkpoints' not in st.session_state:
    st.session_state.checkpoints = {}

# ====================
# sidebar UI
# ====================

with st.sidebar:
    st.subheader("Schedule")
    now = dt.datetime.utcnow()

    start_date = now.date()
    start_time = now.time()
    end_date = start_date + dt.timedelta(days=1)
    end_time = start_time
    start_date = st.date_input("Start date", value=start_date)
    start_time = st.time_input("Start time (UTC)", value=start_time)
    end_date = st.date_input("End date", value=end_date)
    end_time = st.time_input("End time (UTC)", value=end_time)

    options = []
    for _src in SOURCES:
        options += [
            [_src, 'left_boresight_0', 50, 0, 'left_focal_plane'],
            [_src, 'middle_boresight_0', 50, 0, 'middle_focal_plane'],
            [_src, 'right_boresight_0', 50, 0, 'right_focal_plane'],
            [_src, 'bottom_boresight_0', 50, 0, 'bottom_focal_plane'],
            [_src, 'left_boresight_p45', 50, 45, 'left_focal_plane'],
            [_src, 'middle_boresight_p45', 50, 45, 'middle_focal_plane'],
            [_src, 'right_boresight_p45', 50, 45, 'right_focal_plane'],
            [_src, 'bottom_boresight_p45', 50, 45, 'bottom_focal_plane'],
            [_src, 'left_boresight_n45', 50, -45, 'left_focal_plane'],
            [_src, 'middle_boresight_n45', 50, -45, 'middle_focal_plane'],
            [_src, 'right_boresight_n45', 50, -45, 'right_focal_plane'],
            [_src, 'bottom_boresight_n45', 50, -45, 'bottom_focal_plane']
        ]
    cal_targets_candidate = yaml.safe_load(st.session_state.user_config_str).get('cal_targets', [])
    cal_targets = st.multiselect("Calibration Sources", options=options)
    user_config = yaml.safe_load(st.session_state.user_config_str)
    user_config['cal_targets'] = cal_targets
    merge_order = list(set([tar[0] for tar in cal_targets])) + ['baseline']
    st.text("Merge order: ")
    merge_order_sorted = sort_items(merge_order)

    user_config = yaml.safe_load(st.session_state.user_config_str)
    user_config['cal_targets'] = cal_targets
    user_config['merge_order'] = merge_order_sorted
    st.session_state.user_config_str = yaml.dump(user_config) 

    with st.expander("Customize Source", expanded=False):
        source_name = st.selectbox("Name", options=SOURCES)
        elevation = st.number_input("Elevation (deg)", value=50.0)
        boresight_angle = st.selectbox("Boresight angle", options=[0, 45, -45])
        query = st.multiselect("Array query", options=[
            'left_boresight_0', 'middle_boresight_0', 'right_boresight_0', 'bottom_boresight_0',
            'left_boresight_p45', 'middle_boresight_p45', 'right_boresight_p45', 'bottom_boresight_p45',
            'left_boresight_n45', 'middle_boresight_n45', 'right_boresight_n45', 'bottom_boresight_n45',
            'ws0', 'ws1', 'ws2', 'ws3', 'ws4', 'ws5', 'ws6',
        ])
        tag = st.text_input("Tag", value="")
        def on_add():
            user_config = yaml.safe_load(st.session_state.user_config_str)
            new_entry = [source_name, ",".join(query), elevation, boresight_angle, tag]
            if 'cal_targets' not in user_config: user_config['sources'] = []
            user_config['cal_targets'].append(new_entry)
            st.session_state.user_config_str = yaml.dump(user_config)
        def on_reset():
            user_config = yaml.safe_load(st.session_state.user_config_str)
            user_config['cal_targets'] = []
            st.session_state.user_config_str = yaml.dump(user_config)
        st.button("Add source", on_click=on_add)
        st.button("Reset sources", on_click=on_reset)
        _sources = yaml.safe_load(st.session_state.user_config_str).get('cal_targets', [])
        st.table(pd.DataFrame(
            _sources,
            columns=['source', 'query', 'elevation', 'boresight', 'tag']))
        
    with st.expander("Advanced"):
        # user_config = st.text_area("Config overwrite:", value=json.dumps(st.session_state.user_config, indent=2), height=300)
        user_config_str = st_ace(value=st.session_state.user_config_str, language='yaml')
        try:
            user_config = yaml.safe_load(user_config_str)
            # save a good config on parsing success
            st.session_state.user_config_str = user_config_str
        except Exception as e:
            st.error('Unable to parse config', icon="🚨")
            user_config = yaml.safe_load(st.session_state.user_config_str)

    def on_load_schedule():
        t0 = dt.datetime.combine(start_date, start_time).astimezone(dt.timezone.utc)
        t1 = dt.datetime.combine(end_date, end_time).astimezone(dt.timezone.utc)

        config = get_config('satp1')
        config = utils.nested_update(config, user_config)
        policy = policies.SATPolicy.from_config(config)

        seqs = policy.apply(policy.init_seqs(t0, t1))
        commands = policy.seq2cmd(seqs, t0, t1)
        st.session_state.checkpoints = policy.checkpoints
        st.session_state.commands = commands
        
    st.button("Generate Schedule", on_click=on_load_schedule)


# ====================
# main page
# ====================

for ckpt_name, ckpt_seqs in st.session_state.checkpoints.items():
    with st.expander(f"Checkpoint: {ckpt_name}", expanded=True):
        data, groups = seq2visdata(ckpt_seqs)
        timeline = st_timeline(data, groups, key=ckpt_name)

with st.expander("Commands", expanded=False):
    st.code(str(st.session_state.commands), language='python')