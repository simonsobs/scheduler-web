# scheduler-web

`scheduler-web` is the web front end to the scheduler, providing a streamlit
interface for understanding the schedule configuration.

## Related Packages

* [**scheduler**](https://github.com/simonsobs/scheduler) - The core scheduling
  library, a.k.a. `schedlib`.
* [**scheduler-server**](https://github.com/simonsobs/scheduler-server) - The
  Flask API for fetching schedules.
* [**scheduler-web**](https://github.com/simonsobs/scheduler-web) - This app.
  The web front end for the scheduler.


## Installation
First clone this repository, and then install the dependencies:
```bash
git clone git@github.com:simonsobs/scheduler-web.git
pip install -r requirements.txt
```
## Launch
You can launch scheduler-web locally by running:
```bash
streamlit run src/app.py --server.address=localhost --browser.gatherUsageStats=false --server.fileWatcherType=none
```

You can then navigate to http://localhost:8501 to access the app.

### Docker
Alternatively, you can build and launch the server in a docker container:
```bash
docker build -t scheduler-web .
docker run --rm -p 8501:8501 scheduler-web
```
