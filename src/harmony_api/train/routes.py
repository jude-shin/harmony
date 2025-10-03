from fastapi import APIRouter, HTTPException

router: APIRouter = APIRouter(
    prefix="/train",
)

@router.post('/start')
async def start():
    raise HTTPException(status_code=500, detail='[start] route not implemented yet... Sorry :(')

@router.post('/stop')
async def stop():
    raise HTTPException(status_code=500, detail='[stop] route not implemented yet... Sorry :(')
