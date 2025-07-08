import os
import asyncio
import aiohttp
import aiofiles
import logging
import json 

from urllib.parse import urlparse
from tqdm.asyncio import tqdm
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

from utils.product_lines import PRODUCTLINES as PLS

async def _ensure_session():
    timeout = aiohttp.ClientTimeout(total=60)
    connector = aiohttp.TCPConnector(limit=50)
    return aiohttp.ClientSession(timeout=timeout, connector=connector)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(2),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
)
async def _download_image(session, item, pl: PLS):
    try:
        url = item['images']['large']
        img_id = str(item['_id'])


        data_dir= os.getenv('DATA_DIR')
        if data_dir is None:
            logging.error(' [generate_keys] DATA_DIR env var not set. Returning an empty dict.')
            raise FileNotFoundError
    
        images_dir = os.path.join(data_dir, pl.value, 'images')

        filename = os.path.join(images_dir, f'{img_id}.jpg')

        async with session.get(url) as response:
            if response.status == 200:
                async with aiofiles.open(filename, 'wb') as f:
                    await f.write(await response.read())
                logging.info(f'Downloaded {img_id} from {url}')
            else:
                msg = f'Failed to download {img_id} (HTTP {response.status})'
                logging.warning(msg)
                raise aiohttp.ClientError(msg)
    except Exception as e:
        logging.error(f'Error downloading {item["_id"]} from {item["images"]["large"]}: {e}')
        raise

async def _run_downloads(items, pl: PLS):
    async with await _ensure_session() as session:
        tasks = [_download_image(session, item, pl) for item in items]
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc='Downloading images'):
            try:
                await f
            except Exception:
                continue  # error already logged via retry

async def download_images(pl: PLS):
    '''
    Download images concurrently using aiohttp.
    
    Args:
        items (list[dict]): List of dicts with keys ['_id'] and ['images']['large']
    '''
    
    data_dir= os.getenv('DATA_DIR')
    if data_dir is None:
        logging.error(' [generate_keys] DATA_DIR env var not set. Returning an empty dict.')
    
    images_dir = os.path.join(data_dir, pl.value, 'images')

    json_path = 'deckdrafterprod.json'
    
    deckdrafterprod_path = os.path.join(data_dir, pl.value, json_path)
    
    with open(deckdrafterprod_path, 'r') as f:
            deckdrafterprod = json.load(f)


    await _run_downloads(deckdrafterprod, pl)

