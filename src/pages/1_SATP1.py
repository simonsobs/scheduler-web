import json
import datetime as dt
from functools import partial

import streamlit as st
from streamlit_timeline import st_timeline

from schedlib import policies, core, utils
from scheduler_server.configs import get_config

import jax.tree_util as tu

# ====================
# utility functions
# ====================

def block2dict(block, group=None):
    res = {
        'id': hash(block),
        'content': block.name,
        'start': block.t0.isoformat(),
        'end': block.t1.isoformat(),
    }
    if group is not None: res['group'] = group
    return res

def seq2visdata(seqs):
    # make group
    is_list = lambda x: isinstance(x, list)
    groups = []
    tu.tree_map_with_path(
        lambda path, x: groups.append({
            'id': utils.path2key(path), 
            'content': utils.path2key(path)
        }), 
        seqs, 
        is_leaf=is_list
    )
    # make items
    seqs = tu.tree_leaves(
        tu.tree_map_with_path(
            lambda path, x: core.seq_map(
                partial(block2dict, group=utils.path2key(path)), 
                core.seq_sort(x, flatten=True)
            ),
            seqs, 
            is_leaf=is_list
        ),
        is_leaf=lambda x: 'id' in x)
    return seqs, groups

# ====================
# initialize session state
# ====================

if 'user_config' not in st.session_state:
    st.session_state.user_config = {}
    
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

    with st.expander("Advanced"):
        user_config = st.text_area("Config overwrite:", value=json.dumps(st.session_state.user_config, indent=2), height=300)
        try:
            user_config = json.loads(user_config)
        except Exception as e:
            st.error('Unable to parse config', icon="🚨")
            user_config = {}
        st.session_state.user_config = user_config

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