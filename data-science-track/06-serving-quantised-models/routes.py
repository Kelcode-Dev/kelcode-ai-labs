from fastapi import APIRouter, Depends, Body, HTTPException
from starlette.concurrency import run_in_threadpool
from schemas import PredictRequest, PredictResponse, LabelScore
from main import _drivers, resolve_model

router = APIRouter()

@router.get("/health")
def health(): return {"status": "ok"}

@router.post("/predict", response_model=PredictResponse)
async def predict(
    text: str = Body(..., embed=True),
    model_name: str = Depends(resolve_model)
):
    driver = _drivers[model_name]
    raw = await run_in_threadpool(driver.predict, text)
    id2label = driver.config.id2label
    items = [
        LabelScore(code=i, label=id2label[str(i)], score=prob)
        for i, prob in enumerate(raw)
    ]
    items.sort(key=lambda x: x.score, reverse=True)
    return PredictResponse(model=model_name, predictions=items)
