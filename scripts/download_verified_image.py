import hashlib,json,os,time,tarfile
from pathlib import Path
import requests
endpoint=os.environ['HEAR_IMAGE_TRANSFER_URL']
certificate='.codex/image-transfer-public.crt'
archive=Path('/tmp/hear-ai-production-candidate.tar')
assert requests.get(endpoint,verify=certificate,timeout=30).status_code==403
size=14871298048
for attempt in range(5):
    offset=archive.stat().st_size if archive.exists() else 0
    if offset==size:break
    try:
        token_response=requests.get(os.environ['ACTIONS_ID_TOKEN_REQUEST_URL'],params={'audience':'hear-ai-image-transfer'},headers={'Authorization':'Bearer '+os.environ['ACTIONS_ID_TOKEN_REQUEST_TOKEN']},timeout=30)
        token_response.raise_for_status()
        headers={'Authorization':'Bearer '+token_response.json()['value']}
        if offset:headers['Range']=f'bytes={offset}-'
        with requests.get(endpoint,headers=headers,verify=certificate,stream=True,timeout=(30,180)) as response:
            response.raise_for_status()
            assert response.status_code==(206 if offset else 200)
            with archive.open('ab' if offset else 'wb') as output:
                report=time.monotonic()
                for block in response.iter_content(8*1024*1024):
                    output.write(block);offset+=len(block)
                    if time.monotonic()-report>30:print(json.dumps({'archive_bytes_received':offset,'archive_bytes_total':size}),flush=True);report=time.monotonic()
        break
    except requests.RequestException as error:
        print(json.dumps({'transfer_retry':attempt+1,'error_type':type(error).__name__}),flush=True)
        if attempt==4:raise RuntimeError('Authenticated archive transfer failed') from None
        time.sleep(3)
assert archive.stat().st_size==size,'Incomplete archive'
digest=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()
assert digest=='9d590fe8a6c1a4093583f618632907fc594a8b9d8234815507bd4043ef5aa0f5','Archive checksum mismatch'
with tarfile.open(archive) as source:
    manifest=json.load(source.extractfile('manifest.json'))
    assert len(manifest)==1
    raw=source.extractfile(manifest[0]['Config']).read()
    assert hashlib.sha256(raw).hexdigest()=='559feefe3008f198b773cfc20c54375b5285e468fa1265cbbf7d97f1602e4818'
    config=json.loads(raw)
Path('/tmp/hear-ai-before.json').write_text(json.dumps({'rootfs':config['rootfs'],'runtime_configuration':config['config']}))
print(json.dumps({'archive_verified':True,'bytes':size,'sha256':digest}),flush=True)
