from fastapi import FastAPI

from car5_autolabel import __version__


app = FastAPI(
    title="car5 2D Auto Labeling",
    version=__version__,
    description="Detector and Label Studio integration service.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
