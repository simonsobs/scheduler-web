import datetime as dt
import numpy as np
from zoneinfo import ZoneInfo
import so3g.proj as proj
import ephem

import streamlit as st
from matplotlib import pyplot as plt


so3gsite = proj.coords.SITES['so']
site = so3gsite.ephem_observer()
CHILE = ZoneInfo("America/Santiago")
UTC = dt.timezone.utc


def meas_angle(az1, el1, az2, el2):
    
    x1 = np.cos(el1)*np.sin(az1)
    y1 = np.cos(el1)*np.cos(az1)
    z1 = np.sin(el1)
    
    x2 = np.cos(el2)*np.sin(az2)
    y2 = np.cos(el2)*np.cos(az2)
    z2 = np.sin(el2)
    
    dot = x1*x2 + y1*y2 + z1*z2
    return np.rad2deg(np.arccos(dot))


def sun_angles(
    start: dt.datetime, 
    end: dt.datetime, 
    delta: float, 
    Az:float, 
    El:float, 
    site:ephem.Observer=site, 
    zone:ZoneInfo=CHILE):

    delta = dt.timedelta(minutes=delta)
    
    az = np.deg2rad(Az)
    el = np.deg2rad(El)
    
    data = []
    i = 0

    # Calculate sun position every 10 minutes
    current_time = start
    while current_time <= end:
        site.date = ephem.Date(current_time)
        current_time += delta

        sun = ephem.Sun(site)

        az_sun = sun.az
        el_sun = sun.alt
        
        angle = meas_angle(az, el, az_sun, el_sun)
        data.append([current_time, angle])
    
    return np.array(data)

def plot_sun_angles(Az, El, start, end, delta, thre=45, site=site, zone=CHILE):
    
    data = sun_angles(start, end, delta, Az, El, site=site, zone=zone)
    datatime = [data[i][0] for i in range(len(data))]
    angle = [data[i][1] for i in range(len(data))]

    fig, ax = plt.subplots(figsize=(9, 5))  # Adjust as needed
    ax.plot(datatime, angle)
    ax.axhline(y=thre, color='r', linestyle='-')

    # cross point of the line and the curve
    cp = []
    message = ""
    for i in range(len(datatime)-1):
        if angle[i] < thre and angle[i+1] > thre:
            message += "{},{} becomes safe at: {}\n".format(
                Az, El, 
                datatime[i].astimezone(t0.tzinfo).strftime("%Y-%m-%d  %H:%M")
            )
            cp.append(datatime[i])
        elif angle[i] > thre and angle[i+1] < thre:
            x = i-1
            if x < 0:
                x=0
            message += "{},{} becomes UNSAFE at: {}\n".format(
                Az, El,
                datatime[x].astimezone(t0.tzinfo).strftime("%Y-%m-%d  %H:%M")
            )
            cp.append(datatime[x])
    if len(cp) == 0:
        if angle[i] <= thre:
            message += "{},{} is always UNSAFE\n".format(
                Az, El, 
            )
        else:
            message += "{},{} is always safe\n".format(
                Az, El, 
            )
            
    # plot the cross point
    for i in range(len(cp)):
        ax.axvline(x=cp[i], color='black', linestyle='--')
        
    ax.set_ylabel('Sun angle [deg]')
    if t0.tzinfo == UTC:
        x = 'UTC'
    elif t0.tzinfo == CHILE:
        x = 'CLT'
    ax.set_xlabel(f'Time ({x})')
    plt.show()

    st.pyplot(fig)

    st.text(f"{thre} deg threshold: \n" + message) 


with st.form("my data",clear_on_submit=False):
    st.title("Sun Avoidance Angle Calculator")
    st.write(
        "Angle Calculations and Plot Format taken directly from Daichi Sasaki's https://github.com/d1ssk/satp3sky software"
    )


    left_column,  right_column = st.columns(2)

    with left_column:
        tz = st.selectbox("TimeZone", ("CLT", "UTC") )
        if tz == "UTC":
            use_TZ = UTC
        elif tz == "CLT":
            use_TZ = CHILE
        now = dt.datetime.now().astimezone(use_TZ)
        start_date = now.date()
        start_time = now.time()

        end_date = start_date + dt.timedelta(days=1)
        end_time = start_time

        start_date = st.date_input("Start date", value=start_date)
        start_time = st.time_input("Start time", value=start_time)
        end_date = st.date_input("End date", value=end_date)
        end_time = st.time_input("End time", value=end_time)

        sampling = st.number_input(
            "Sampling (min)", min_value=1, max_value=60, value=5
        )

        
    with right_column:
        azimuth = st.number_input(
            "Azimuth (deg)", min_value=0, max_value=360, value=180
        )
        elevation = st.number_input(
            "Elevation (deg)", min_value=0, max_value=90, value=50
        )    

        keep_out = st.number_input(
            "Keep Out Angle (deg)", min_value=0, max_value=90, value=41
        )
    run_calculation = st.form_submit_button("Calculate")

    if run_calculation:
        t0 = dt.datetime.combine(start_date, start_time, tzinfo=use_TZ)
        t1 = dt.datetime.combine(end_date, end_time, tzinfo=use_TZ)

        plot_sun_angles(azimuth, elevation, t0, t1, sampling, thre=keep_out)

