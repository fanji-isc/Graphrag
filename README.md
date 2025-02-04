[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat&logo=AdGuard)](LICENSE)
# intersystems-iris-graphrag
This is a basic template (created during the 2025 InterSystems Employee Hackathon), in order to run Graph-Rag natively in IRIS using it's Vector Search capabilities. It generates a working image for Jupyter + InterSystems IRIS. The image comes with all of the necessary python modules installed to use graphrag out of the box.


## Description
This repository provides a ready-to-go development environment for using our graphrag implementation.

## Prerequisites
Make sure you have [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) and [Docker desktop](https://www.docker.com/products/docker-desktop) installed.

## Installation

Clone/git pull the repo into any local directory


Open the terminal in this directory and call the command to build and run the two containes:
*Note: Users running containers on a Linux CLI, should use "docker compose" instead of "docker-compose"*
*See [Install the Compose plugin](https://docs.docker.com/compose/install/linux/)*


```
$ docker-compose up -d
```

The jupyter notebook should be accessible through: [http://localhost:4200](http://localhost:4200)

**Password:** admin

The IRIS management portal should be accessible through: [http://localhost:9092/csp/sys/UtilHome.csp](http://localhost:9092/csp/sys/UtilHome.csp)