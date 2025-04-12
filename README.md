# Fast API backend

Playground to get the gist around :
- REST API best practices
- Containerisation
- Managing dependencies with uv
- Github workflow for CI

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
***

Sources :
- https://github.com/astral-sh/uv-docker-example
- https://docs.github.com/en/actions/writing-workflows/about-workflows
- https://www.kaggle.com/datasets/everget/government-bonds/data
