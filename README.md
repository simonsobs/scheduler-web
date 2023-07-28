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

## Deployment
Deploying `scheduler-web` is typically done using docker-compose and a reverse
proxy, such as nginx. Here is just one example of how to deploy in this
configuration.

### Docker Compose
Here is an example Docker Compose service block for `scheduler-web` that allows
for proxying to an arbitrary subdirectory, here `anySubPath`:

```yaml
  scheduler-web:
    image: simonsobs/scheduler-web:latest
    container_name: scheduler-web
    restart: always
    ports:
      - 8501:8501
    command: ["--server.baseUrlPath=/anySubPath"]
```

This path can be changed, and must match the path configured in nginx. The port
used on the host (the left 8501) can be changed if needed. If that is changed,
you must match the port number in the nginx config below.

### nginx
Below is a location snippet for nginx to proxy `scheduler-web`.

```
  location /anySubPath/ {
    proxy_pass         http://<ip-address>:8501/anySubPath/;
    proxy_set_header   Host      $host;
    proxy_set_header   X-Real-IP $remote_addr;
    proxy_set_header   X-Forwarded-For $remote_addr;
    proxy_set_header   X-Forwarded-Proto $scheme;
    proxy_http_version 1.1;

    # websockets
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
  }
```

Here you can replace `anySubPath` with the subdirectory you would like to serve
from. Note this must match the path passed to `--server.baseUrlPath` in the
docker-compose file.

You must also replace `<ip-address` in the `proxy_pass` command with the
appropriate address or domain name for the machine running `scheduler-web`.
