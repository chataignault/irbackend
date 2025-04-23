# Interest Rates backend API and analytics

With public rates or macro data, 
the objective is to model various target objectives,
using models of all sorts.

It is also a playground to get the gist around dev concepts and tools :
- REST API best practices
- Containerisation
- Managing dependencies with uv
- Github workflow for CI

## Data
Because the idea is to gather data from different sources,
with, say, different ticking frequencies, qualities and types, 
the long run objective is to have a clean ingestion pipeline
that can gather each endpoint, 
manage exceptions without failing,
and run in a separate service.

Options :
- `pandas` / `polars` dataframe for a few datasources only (short term),
- https://datafusion.apache.org/
- https://iceberg.apache.org/


## Targets

**FR yiels**



## Models

**Tools overview :**

| Task | Package |
| --- | --- |
| Data preprocessing | polars |
| Linear algebra | numpy |
| Unit tests | pytest |
| API Routing | uvicorn, fastapi |

## Run API

To start the app, run :
```bash
uvicorn src.irbackend.main:app --reload
```

To build the image, run :
```bash
source run.sh
```

## References :
- to display full logging in the uvicorn app :
    https://gist.github.com/liviaerxin/d320e33cbcddcc5df76dd92948e5be3b
- https://github.com/astral-sh/uv-docker-example/tree/main
- https://github.com/actions/starter-workflows
- https://github.com/astral-sh/uv-docker-example
- https://docs.github.com/en/actions/writing-workflows/about-workflows

***

Data sources :
- https://www.kaggle.com/datasets/everget/government-bonds/data
- https://api.energy-charts.info/