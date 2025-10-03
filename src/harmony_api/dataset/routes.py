from fastapi import APIRouter, HTTPException

router: APIRouter = APIRouter(
    prefix='/dataset',
)

# dequeues and processes the data from the mongodb
@router.post('/dequeue')
async def dequeue():
    raise HTTPException(status_code=500, detail='[dequeue] route not implemented yet... Sorry :(')

# merges stray records in the records volume together
@router.post('/merge-records')
async def merge_records():
    raise HTTPException(status_code=500, detail='[merge-records] route not implemented yet... Sorry :(')

# returns stats on all of the records
@router.get('/record-info')
async def record_info():
    raise HTTPException(status_code=500, detail='[record-info] route not implemented yet... Sorry :(')
