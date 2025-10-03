from fastapi import APIRouter, HTTPException

router: APIRouter = APIRouter(
    prefix="/manage",
)

# enable a model (and version) from bein serveable
@router.post('/deploy')
async def deploy():
    raise HTTPException(status_code=500, detail='[deploy] route not implemented yet... Sorry :(')

# disable a model (and version) from being servable
@router.post('/recall')
async def recall():
    raise HTTPException(status_code=500, detail='[recall] route not implemented yet... Sorry :(')

# removes the model.keras from the tensroflow serving location alltogether
# please try not to use this
@router.post('/purge')
async def purge():
    raise HTTPException(status_code=500, detail='[purge] route not implemented yet... Sorry :(')

# gets all of the processes that are training, and that have been trained?
@router.get('/monitor')
async def monitor():
    raise HTTPException(status_code=500, detail='[monitor] route not implemented yet... Sorry :(')
