# Docker and Singularity runtime

The scientific environment remains `zhaochang07/myubuntu:neuro-v2` unless an
explicit image is supplied. No tool version is automatically upgraded. Its
original Dockerfile is unavailable; the new Dockerfile is a **compatibility layer
over a digest-pinned base**, not a reproducible reconstruction of that base.
It checks the required command/Python environment without installing packages.
Image builds and real Singularity execution have not yet been verified.

## What the existing 79 GB image contains (measured 2026-09-24)

`docker images` reports 79.3 GB because the containerd image store counts the
compressed blobs and the unpacked layers; the layers themselves add up to about
49 GB and the final filesystem to about 40 GB. `docker history` and `du` inside
the image show where the space goes and what this pipeline never uses:

| Item | Size | Used by the pipeline? |
|---|---|---|
| `COPY freesurfer-...tar.gz` layer (extracted in a later layer) | 9.5 GB | no - the tarball stays in the image forever |
| `/opt/mcr` (MATLAB Runtime R2019b) | 5.8 GB | no (FreeSurfer subfield/thalamus tools only) |
| `/opt/fsl/pkgs` (conda package cache, not hard-linked part) | ~1.8 GB | no |
| `/opt/micromamba/pkgs` | 0.4 GB | no |
| `/opt/freesurfer/trctrain` (TRACULA training data) | 1.4 GB | no |
| `/opt/freesurfer/subjects/{bert,cvs_avg35*,V1_average,fsaverage_sym}` | ~1.3 GB | no (keep `fsaverage*`) |
| `/opt/freesurfer/average/mult-comp-cor`, `average/samseg` | 1.8 GB | no |
| FreeSurfer python `site-packages/nvidia` (CUDA) | 2.6 GB | no GPU is used; test `mri_synthstrip` (PyTorch) after removing |
| `/opt/mrtrix3` | 0.2 GB | no |

Recommendations, in order of value:

1. **Non-root users cannot run AFNI.** AFNI is installed in `/root/abin` and
   `/root` is `drwx------`; Singularity/Apptainer run as the calling user, so
   every `3d*` program is "not found" on a cluster. `docker/Dockerfile` now runs
   `chmod 755 /root` and checks the toolchain as user `nobody` (verified: with
   `/root` opened, all tools and Python imports work for uid 65534). A rebuild
   should install AFNI under `/opt/afni` instead.
2. **Never `COPY` an archive and extract it in a later layer**: download and
   extract in one `RUN` (or use a BuildKit bind mount), otherwise the archive
   stays in the image (9.5 GB here). An existing image can be flattened with
   `docker export <container> | docker import - <tag>` (re-add `ENV`/`CMD` with
   `--change`), which also drops files deleted in later layers.
3. Remove what the pipeline does not use (table above, ~13-15 GB) and clean the
   conda caches (`conda clean -afy`, `micromamba clean -afy`) in the same `RUN`.
4. Bake or mount the TemplateFlow/HCP/Schaefer resources (`stages/fetch_resources.sh`,
   ~50 MB) so cluster jobs run offline.
5. Keep `bash -lc` (tools are set up in `/etc/profile.d`), or put the full
   environment in `ENV` so that `singularity exec` works without a login shell.
6. Record the image digest with every run (the pipeline writes tool versions to
   `logs/tool_versions.json` and the code to `logs/code_<run>/`).

Local Docker Desktop (Windows, Hyper-V backend): a 12 GB VM runs the smoke test
and two concurrent recon-all jobs on this 31 GB host; 20 GB failed to allocate.
If Docker Desktop reports `remove ...\dockerInference` or `...\engine.sock: The
file cannot be accessed by the system`, a previous crash left AF_UNIX socket files
that Windows cannot delete: quit Docker Desktop, rename
`%LOCALAPPDATA%\Docker\run` (and `%LOCALAPPDATA%\docker-secrets-engine`), start
it again; the renamed folders can be deleted after a reboot. Never choose
"Reset to factory defaults" there: it deletes all images.

## Next environment: clean build candidate

`generate_runtime.sh` produces a Dockerfile on stdout using an **already installed
Neurodocker 2.1.2**. It neither installs Neurodocker nor invokes Docker/downloads.
This is the proposed replacement for the 79 GB image; `docker/Dockerfile` below
remains a compatibility transition for the existing image. Environment upgrade is
not complete until this candidate builds and passes real-data comparison.
`Dockerfile.clean` is the actual generated output; regenerate it only with
`generate_runtime.sh`. It is provided so the next build machine does not need to
install the generator. Real Neurodocker 2.1.2 generation has passed locally,
including shell syntax checks for all 10 generated RUN instructions and inspection
of container paths. This does not establish build success.

The candidate fixes Ubuntu 22.04, FreeSurfer 7.4.1, ANTs 2.6.2, FSL 6.0.7.22,
Python 3.11.11 and direct Python package versions. It uses binary releases, not
compilers. FreeSurfer's Neurodocker defaults exclude CUDA/Qt libraries and example
subjects while retaining fsaverage and model resources. No GPU is requested and
CUDA devices are hidden; packaged upstream Python libraries are not aggressively
stripped. No AFNI R packages or separate desktop environment are installed.
Its final size is unknown until built.
The fixed Miniconda installer is `py311_25.7.0-2`; it replaces the earlier proposed
24.11 installer because Neurodocker 2.1.2 calls `conda tos accept`. Environment
Python remains 3.11.11, and environment package resolution explicitly uses
conda-forge. The standard template's terms handling is retained.

AFNI and Workbench use explicit HTTPS binary URLs plus verified SHA256 build args.
AFNI's rolling download must first be inspected on the preparation machine and its
actual `afni -ver` supplied as `AFNI_VERSION` (e.g. `26.2.09` only if that is what
the downloaded binary reports). Workbench **2.2.1** is the candidate, not the
previous 2.1 series. The build verifies both reported versions. Do not substitute
the AFNI source archive for the binary archive. Obtain downloads from
[AFNI](https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/background_install/download_links.html)
and [Workbench](https://www.humanconnectome.org/software/get-connectome-workbench).

```bash
# Run where Neurodocker 2.1.2 is already installed; no image execution here.
bash docker/generate_runtime.sh > /tmp/fmriproc-clean.Dockerfile
# Optional: supply BASE_IMAGE=ubuntu:22.04@sha256:<resolved digest> above.
# Fill these shell variables from the real downloaded binaries and SHA256 values.
: "${AFNI_URL:?}" "${AFNI_SHA256:?}" "${AFNI_VERSION:?}"
: "${WORKBENCH_URL:?}" "${WORKBENCH_SHA256:?}"
docker build --platform linux/amd64 -t fmriproc:clean-candidate \
  --build-arg AFNI_URL --build-arg AFNI_SHA256 --build-arg AFNI_VERSION \
  --build-arg WORKBENCH_URL --build-arg WORKBENCH_SHA256 \
  - < /tmp/fmriproc-clean.Dockerfile
```

Export those variables before `docker build` so Docker can read them. The generated
recipe rejects missing/malformed checksums and verifies archive bytes; no checksum
is fabricated in the repository. These version pins are not a full transitive or
OS package lock: apt resolution and Python transitive packages can evolve. Each
built candidate records conda explicit packages, pip freeze, and binary versions
under `/opt/runtime-record`; retain its image ID and generated recipe for reuse.
Neurodocker's selected template versions must be supported; generation fails
instead of silently substituting another version.

Build command/import checks are necessary but insufficient: on the next machine,
run a representative image through volume and surface branches, including
SynthStrip/SynthSeg model loading, registration, censoring and ROI/FC extraction.
Compare against the existing environment with identical inputs/configuration and
inspect QC before adopting changed AFNI/FSL/ANTs/Workbench/Python versions.
The current host cannot build because Docker fails during local socket
initialization; no endpoint reset or Docker data reset was attempted here.

Sources: [Neurodocker CLI and supported versions](https://repronim.org/neurodocker/user_guide/cli.html),
[Neurodocker 2.1.2 release](https://github.com/ReproNim/neurodocker/releases/tag/2.1.2),
[official Miniconda installer versions](https://repo.anaconda.com/miniconda/).

## Build on a machine with enough disk and a working Docker engine

The existing image is documented as approximately 79 GB; inspect its actual size
and available storage before pulling/building/exporting. Conversion needs storage
for the archive, extracted layers, temporary files, and final SIF simultaneously.
Do not run these operations on a paid cluster login node.

```bash
# If already local, inspect without downloading anything.
docker image inspect zhaochang07/myubuntu:neuro-v2 --format '{{json .RepoDigests}}'
# Copy the actual repository@sha256:... value from the output. If absent, obtain
# an immutable registry digest before building; do not invent a digest.
docker build --build-arg BASE_IMAGE='zhaochang07/myubuntu@sha256:ACTUAL_DIGEST' \
  -t fmriproc:local - < docker/Dockerfile
docker image inspect fmriproc:local --format '{{.Id}}'
```

`ACTUAL_DIGEST` is a placeholder. Builds reject an unpinned base. A digest fixes
the chosen upstream image but does not recover its installation recipe. Keep
the resolved base digest and derived image ID with the run record. The repository
is mounted read-only at runtime, so its relevant code revision must also be recorded.
Do not include datasets or the FreeSurfer license in image layers.
The Bash stdin build above sends only the Dockerfile, with no repository/data
build context; use Git Bash for this command on Windows.

For local Docker, use the existing PowerShell/Bash launcher with
`-Image fmriproc:local` / `--image fmriproc:local`; use a distinct volume prefix
for each dataset. The default remains the existing image for continuity.

## Offline conversion on this or another Linux machine

```bash
docker save --output fmriproc-local.tar fmriproc:local
# On a Linux machine with SingularityCE 3.9+ (or replace singularity by apptainer):
mkdir -p ./sif-tmp ./sif-cache
export SINGULARITY_TMPDIR="$PWD/sif-tmp"
export SINGULARITY_CACHEDIR="$PWD/sif-cache"
singularity build fmriproc-local.sif docker-archive:fmriproc-local.tar
```

For Apptainer use `APPTAINER_TMPDIR` / `APPTAINER_CACHEDIR`. `docker save`, not
`docker export`, preserves image configuration and layers. The archive can be
copied to another Linux computer for conversion without accessing a registry.
Distribution or upload of project data is a separate authorized operation.

## Run a prepared local SIF

```bash
bash docker/run_singularity.sh \
  --image /project/images/fmriproc-local.sif \
  --data /project/input --out /project/output \
  --license /project/license.txt \
  --config config/datasets/abide_smoke.conf \
  --env NTHREADS=4 --env N_JOBS=1 --print
```

Remove `--print` only when ready to execute. Print mode neither creates
directories nor invokes a runtime. Inputs and image must already exist.
`--runtime apptainer` selects Apptainer; no remote image URI is accepted.
The code, input data, config and license are read-only mounts. Outputs, resources,
FreeSurfer results, home/cache and temporary files are explicitly writable binds;
the SIF itself stays immutable. `--work`, `--resources`, and `--freesurfer` can
select separate host directories. Writable paths overlapping raw data or the
repository are rejected. Paths containing commas, colons, or newlines are rejected
because of bind syntax. Run this launcher on Linux, not directly on Windows.

The wrapper retains `bash -lc` because the existing base initializes neuroimaging
tool paths in its login profile. `--cleanenv --containall` separates host module
settings and home from the run; explicit `env` arguments carry configuration
values without runtime comma parsing. Only Slurm CPU/memory limits and explicit
`CPU_BUDGET` / `MEMORY_BUDGET_GB` are forwarded automatically. Set per-subject
threads and subject concurrency in the configuration or with `--env`.

Prepare resources once on a connected machine using the existing
`stages/fetch_resources.sh`, then retain that resource directory for offline runs.
Inside the launcher shell, check the selected dataset's assets without downloading:

```bash
bash /opt/fmriproc/stages/fetch_resources.sh -c "$FMRIPROC_CONFIG" --check
```

No Slurm submission is performed or supplied here. A future BSCC run must check
the live runtime and allocation: the current project policy uses one `v6_384`
node, 96 CPUs, 328G with enough independent subjects to use the allocation.
This is not a claim that the server or SIF has been tested.

## Focused local verification

```bash
bash -n docker/run_singularity.sh
bash tests/test_container_launchers.sh
```

These stub tests exercise command construction and failure handling. They cannot
verify SIF mounts, software libraries, profile initialization, or image processing;
those require a real container run on suitable hardware.

Official references: [SingularityCE 3.9 Docker archive conversion and read-only SIF](https://docs.sylabs.io/guides/3.9/user-guide/singularity_and_docker.html),
[environment isolation](https://docs.sylabs.io/guides/3.9/user-guide/environment_and_metadata.html),
[bind mounts and containment](https://docs.sylabs.io/guides/3.9/user-guide/bind_paths_and_mounts.html),
[Apptainer OCI compatibility](https://apptainer.org/docs/user/latest/docker_and_oci.html).
