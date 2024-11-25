# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
'''download_tools_multithread'''
import os
import time
from threading import Thread, Lock
import requests
import urllib3
from tqdm import tqdm

from logger import logger
try:
    import fcntl
except ImportError:
    fcntl = None
    logger.warning("The library fcntl is not found. This may cause the reading file failed "
                   "when call the from_pretrained for different process.")

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def download_chunk(url, start, end, file, pbar, lock, chunk_size=1024):
    '''Download a chunk of a file from the given URL and write it to the provided file object.

    Args:
        url (str): The URL to download from.
        start (int): The starting byte position of the chunk to download.
        end (int): The ending byte position of the chunk to download.
        file (file): The file object to write the downloaded chunk to.
        pbar (tqdm.tqdm): The progress bar object to update the download progress.
        lock (threading.Lock): The lock object to synchronize writing to the file.
        chunk_size (int, optional): The size of each chunk to download in bytes. Defaults to 1024.

    Returns:
        None.

    Raises:
        None.
    '''
    headers = {'Range': 'bytes=%d-%d' % (start, end)}
    response = requests.get(url, headers=headers, verify=False, stream=True)
    if response.status_code == requests.codes.partial_content:
        with lock:
            file.seek(start)
            for data in response.iter_content(chunk_size=chunk_size):
                file.write(data)
                pbar.update(chunk_size)
    else:
        logger.error("%s is unconnected!", url)


def download_with_progress_bar(url, filepath, num_threads=1, timeout=4):
    """Downloads a file from the given URL with multi-threading support, resuming from breakpoints,
    and displays a progress bar to show the progress of the download.

    Args:
        url (str): The URL to download the file from.
        filepath (str): The path where the downloaded file will be saved.
        num_threads (int, optional): The number of threads to use for the download. Default is 8.
        timeout (int, optional): The connection timeout in seconds. Default is 4 seconds.

    Returns:
        bool: Returns True if the download is successful; otherwise, False.
    """
    if not os.path.exists(filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    start_time = time.time()

    try:
        response = requests.head(url, verify=False, stream=True, timeout=timeout)
        content_size = int(response.headers['content-length'])
        etag = response.headers.get('etag')
    except (TimeoutError,
            urllib3.exceptions.MaxRetryError,
            requests.exceptions.ProxyError,
            requests.exceptions.ConnectionError):
        logger.error("Connect error, please download %s to %s.", url, filepath)
        return False

    if content_size is None:
        response_json = response.json()
        download_url = response_json.get("data").get("download_url")
        if download_url:
            header = {
                "Accept-Encoding": "identity",
                "User-agent": 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.12; rv:65.0) Gecko/20100101 Firefox/65.0'
            }
            try:
                response = requests.head(download_url, verify=False, stream=True, timeout=timeout, headers=header)
                content_size = int(response.headers.get('content-length'))
                etag = response.headers.get('etag')
            except (TimeoutError,
                    urllib3.exceptions.MaxRetryError,
                    requests.exceptions.ProxyError,
                    requests.exceptions.ConnectionError):
                logger.error("Connect error, please download %s to %s.", url, filepath)
                return False
        else:
            logger.error("Download url parsing failed from json file, please download %s to %s.", url, filepath)
            return False

    start = 0
    if os.path.exists(filepath):
        # check file size and etag
        file_size = os.path.getsize(filepath)
        if file_size == content_size and etag == response.headers.get('etag'):
            logger.info('File already exists: %s', filepath)
            return True

        logger.info('File is not complete, resuming download: %s', filepath)
        start = file_size
    flags_ = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    with os.fdopen(os.open(filepath, flags_, 0o750), 'ab') as file:
        if fcntl:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX)

        chunk_size = (content_size - start) // num_threads
        with tqdm(total=content_size, initial=start, unit='B', unit_scale=True, desc=filepath.split('/')[-1]) as pbar:
            threads = []
            for i in range(num_threads):
                if i < num_threads - 1:
                    end = start + chunk_size - 1
                else:
                    end = content_size - 1
                t = Thread(target=download_chunk, args=(url, start, end, file, pbar, Lock(), 1024))
                threads.append(t)
                t.start()
                start += chunk_size

            for t in threads:
                t.join()

    # check downloaded file size and etag
    file_size = os.path.getsize(filepath)
    if file_size == content_size and etag == response.headers.get('etag'):
        end_time = time.time()
        logger.info('Download finished: %s times: %.2fs', filepath, (end_time - start_time))
        return True

    logger.error('Download failed or interrupted: %s', filepath)
    return False
