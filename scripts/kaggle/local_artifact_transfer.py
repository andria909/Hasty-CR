"""Retry terminal artifact downloads while preserving partial and cached files."""
import hashlib
from pathlib import Path
import shutil
import time


def file_sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def download_preserving(url,path):
    import requests
    path=Path(path)
    if path.exists():raise FileExistsError('Refusing to overwrite artifact destination')
    for attempt in range(4):
        partial=path.with_name(path.name+'.download_attempt_'+str(attempt))
        try:
            with requests.get(url,stream=True,timeout=(30,300)) as response:
                response.raise_for_status()
                with partial.open('xb') as stream:
                    for chunk in response.iter_content(1024*1024):
                        if chunk:stream.write(chunk)
            if path.exists():raise FileExistsError('Artifact destination appeared during transfer')
            partial.rename(path);return
        except requests.RequestException as error:
            status=getattr(getattr(error,'response',None),'status_code',None)
            if attempt==3 or status not in (None,404,408,429,500,502,503,504):raise
            time.sleep(2**attempt)


def copy_verified_cache(cache,files,name,destination):
    if cache is None or name not in files:return False
    cache=Path(cache);original=cache/name
    if original.is_symlink() or not original.resolve().is_relative_to(cache.resolve()):raise ValueError('Unsafe cached artifact path')
    if not original.is_file():return False
    if file_sha(original)!=files[name]['sha256']:raise ValueError('Cached artifact changed; refusing reuse')
    destination=Path(destination)
    if destination.exists():raise FileExistsError('Refusing to overwrite cached artifact destination')
    shutil.copyfile(original,destination);return True
