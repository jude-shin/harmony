import logging
from typing import List, Optional

from fastapi import FastAPI

from processing.image_processing import get_tensor_from_image
from utils.product_lines import string_to_product_line
from utils.data_conversion import label_to_id
from utils.file_handler.dir import get_config_path
from utils.file_handler.toml import * 

from harmony_api.tfs_models import identify
from harmony_api.tfs_models import load_image_from_upload
from harmony_api.tfs_models import load_image_from_url

# ---------------------------------------------------------------------------
# Logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
app = FastAPI(title="Harmony ML API", version="1.0.0")
class User(BaseModel):
    username: str
    password: str
        

# ---------------------------------------------------------------------------
###############
### GENERAL ###
###############

@app.get("/ping", summary="testing")
async def ping() -> dict[str, str]:
    return {"ping": "pong"}
