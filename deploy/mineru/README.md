# MinerU sidecar

This optional GPU sidecar is locked to MinerU `3.4.4`. It is intentionally
separate from TaskForge's default Compose stack: the image and model artifacts
are large and require an NVIDIA container runtime.

For this Windows workstation, the **native D-drive setup is the simplest
choice** because it does not require moving an existing Docker image disk:

```powershell
# Installs the venv, pip cache and all models under D:\TaskForge\mineru.
.\scripts\setup_mineru_d_drive.ps1 -Root D:\TaskForge\mineru `
  -ModelSource modelscope -DownloadModels

# Starts the exact 3.4.4 API on loopback.
.\scripts\start_mineru_d_drive.ps1 -Root D:\TaskForge\mineru -Port 8001
```

The setup redirects `TEMP`, `TMP`, pip, Hugging Face, ModelScope, Torch,
Python bytecode, MinerU config and parser output before installation. No
MinerU package or model is intentionally downloaded to C:.

## D-drive storage gate (Windows)

Do not build until Docker Desktop's **Disk image location** is on D:. Docker
image layers are stored inside that disk image, so bind-mounting only the model
cache is not sufficient. In Docker Desktop, open **Settings → Resources →
Advanced → Disk image location**, choose `D:\DockerDesktop`, and apply the
change. Do not move `docker_data.vhdx` manually while Docker is running.

Then run the fail-closed check:

```powershell
.\scripts\check_mineru_storage.ps1 -DataRoot D:\TaskForge\mineru
```

The current Compose file bind-mounts all runtime downloads and outputs below
`D:\TaskForge\mineru` by default:

- Hugging Face cache: `D:\TaskForge\mineru\cache\huggingface`;
- ModelScope cache: `D:\TaskForge\mineru\cache\modelscope`;
- MinerU configuration: `D:\TaskForge\mineru\cache\config\mineru.json`;
- parser output: `D:\TaskForge\mineru\output`.

```powershell
$env:TASKFORGE_MINERU_DATA_ROOT='D:/TaskForge/mineru'
docker compose -f deploy/mineru/compose.yaml --profile mineru build --pull
docker compose -f deploy/mineru/compose.yaml --profile mineru up -d
Invoke-RestMethod http://127.0.0.1:8001/health
```

Download only the pipeline bundle for the default TaskForge path; it covers
layout, OCR, tables and formulas without downloading MinerU's bundled VLM:

```powershell
docker run --rm --gpus all --ipc=host `
  -e MINERU_MODEL_SOURCE=modelscope `
  -e MODELSCOPE_CACHE=/opt/mineru-cache/modelscope `
  -e MINERU_TOOLS_CONFIG_JSON=/opt/mineru-cache/config/mineru.json `
  -v 'D:/TaskForge/mineru/cache:/opt/mineru-cache' `
  --entrypoint mineru-models-download mineru-mineru:latest `
  -s modelscope -m pipeline
```

After the cache and generated `mineru.json` are complete, set
`MINERU_MODEL_SOURCE=local` and `TASKFORGE_MINERU_BACKEND=pipeline` for a
locked evaluation. MinerU's VLM bundle is optional and intentionally deferred;
TaskForge A7 uses a separately configured visual extractor instead.

Configure a host-run TaskForge process with:

```dotenv
TASKFORGE_PDF_PARSER_BACKEND=auto
TASKFORGE_MINERU_BASE_URL=http://127.0.0.1:8001
TASKFORGE_MINERU_EXPECTED_VERSION=3.4.4
TASKFORGE_MINERU_BACKEND=pipeline
TASKFORGE_MINERU_PARSE_METHOD=auto
TASKFORGE_MINERU_EFFORT=high
```

The container is based on MinerU's official `mineru-3.4.4-released` Docker
recipe, but pins the Python package with `==3.4.4`. Both the container
healthcheck and TaskForge's client reject a mismatched runtime version.

MinerU is distributed under the MinerU Open Source License. Review
`LICENSE.md` for the locked release before deployment; its additional terms
include commercial-scale conditions and an attribution requirement for a
third-party online service.
