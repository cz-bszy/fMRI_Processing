#!/usr/bin/env pwsh
#Requires -Version 7.0
<#
.SYNOPSIS
    Run fMRI_Processing v2 inside the neuroimaging container (Windows 11 + Docker Desktop, PowerShell 7).

.DESCRIPTION
    Mounts
        <repo>      -> /opt/fmriproc                (read-only)
        -Data       -> /data                        (read-only)
        -Out        -> /out
        -License    -> /opt/freesurfer/license.txt  (read-only)
    and three Docker NAMED volumes, because Windows bind mounts are slow and cannot
    hold the symbolic links that recon-all creates:
        <prefix>_freesurfer -> /out/freesurfer
        <prefix>_work       -> /out/work
        <prefix>_resources  -> /out/resources
    then runs
        bash -lc "bash /opt/fmriproc/run_pipeline.sh -c /opt/fmriproc/<Config> <arguments>"
    The docker command is printed before it is executed. The exit status is the
    one of run_pipeline.sh (0 = ok, 1 = at least one failed stage, 2 = usage).

    Arguments of run_pipeline.sh: -Subjects / -Stages are translated to -s / --stages;
    anything else is passed through unchanged. Double-dash options (--jobs 2,
    --force, --dry-run, --list-subjects, --subjects-file F) pass through by
    themselves; single-dash ones must follow a bare "--" inside a PowerShell
    session, otherwise PowerShell binds them to the parameters of this script.
    ("pwsh -File run_docker.ps1 ..." does not understand the bare "--": leave it out.)

.PARAMETER Data
    Host directory with the raw dataset (INPUT_DIR=/data in the dataset conf).
.PARAMETER Out
    Host output directory (OUT_DIR=/out in the dataset conf). Created when missing.
.PARAMETER Config
    Dataset conf, relative to the repository (default config/datasets/abide_test.conf).
    An absolute path outside the repository is also accepted: its directory is
    mounted read-only at /config.
.PARAMETER License
    FreeSurfer license file (default $env:USERPROFILE\Desktop\license.txt).
.PARAMETER Image
    Container image (default zhaochang07/myubuntu:neuro-v2).
.PARAMETER VolumePrefix
    Prefix of the named volumes (default fmriproc). Use one prefix per dataset.
.PARAMETER Cpus
    Optional docker --cpus value.
.PARAMETER Memory
    Optional docker --memory value, e.g. 16g.
.PARAMETER Env
    Configuration overrides passed as environment variables: -Env 'SURFACE=no','NTHREADS=8'
    inside a PowerShell session, or -Env SURFACE=no,NTHREADS=8 (one comma-separated
    string) from "pwsh -File", which does not parse array syntax.
.PARAMETER MinMemoryGB
    Warn when the Docker engine has less memory than this (default 6). recon-all,
    ANTs SyN and SynthSeg --robust want 8-16 GB; the pipeline itself refuses heavy
    stages below MIN_MEM_GB of the dataset conf.
.PARAMETER Subjects
    Subjects to process ("sub-A sub-B"); becomes run_pipeline.sh -s.
.PARAMETER Stages
    Stages to run ("ingest anat_recon ..."); becomes run_pipeline.sh --stages.
.PARAMETER Shell
    Open an interactive login shell with the same mounts instead of running the pipeline.
.PARAMETER ExportFreesurfer
    Copy the content of the <prefix>_freesurfer volume to <Out>\freesurfer_export
    (tar through a throw-away container; symbolic links are replaced by the files
    they point to, the fsaverage link is left out). Named volumes are not visible
    from Windows Explorer.
.PARAMETER PrintOnly
    Print the docker command and exit without running anything.
.PARAMETER PipelineArgs
    Remaining arguments, passed to run_pipeline.sh.

.EXAMPLE
    .\docker\run_docker.ps1 -Data E:\ASD\test_abide_ASD -Out E:\ASD\out --list-subjects
.EXAMPLE
    .\docker\run_docker.ps1 -Data E:\ASD\test_abide_ASD -Out E:\ASD\out -Config config/datasets/abide_smoke.conf -VolumePrefix smoke
.EXAMPLE
    .\docker\run_docker.ps1 -Data E:\ASD\test_abide_ASD -Out E:\ASD\out -Subjects "sub-0050952" -Stages "ingest anat_recon anat_prep"
.EXAMPLE
    .\docker\run_docker.ps1 -Data E:\ASD\test_abide_ASD -Out E:\ASD\out -Env 'SURFACE=no','ANAT_MODE=synth' -- --jobs 2 --force
.EXAMPLE
    .\docker\run_docker.ps1 -Out E:\ASD\out -ExportFreesurfer
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Data,
    [string]$Out,
    [string]$Config = 'config/datasets/abide_test.conf',
    [string]$License = $(if ($env:USERPROFILE) { Join-Path $env:USERPROFILE 'Desktop\license.txt' } else { Join-Path $HOME 'license.txt' }),
    [string]$Image = 'zhaochang07/myubuntu:neuro-v2',
    [string]$VolumePrefix = 'fmriproc',
    [string]$Cpus,
    [string]$Memory,
    [string[]]$Env = @(),
    [double]$MinMemoryGB = 6,
    [Alias('s')]
    [string]$Subjects,
    [string]$Stages,
    [switch]$Shell,
    [switch]$ExportFreesurfer,
    [switch]$PrintOnly,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PipelineArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoDir = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

function Stop-WithMessage([string]$Message) {
    Write-Host "run_docker.ps1: $Message" -ForegroundColor Red
    exit 2
}

function Resolve-HostDirectory([string]$Path, [string]$What, [switch]$Create) {
    if ([string]::IsNullOrWhiteSpace($Path)) {
        Stop-WithMessage "$What is required (see Get-Help $PSCommandPath -Detailed)"
    }
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        if (-not $Create) { Stop-WithMessage "$What not found: $Path" }
        if (-not $PrintOnly) { New-Item -ItemType Directory -Force -Path $Path | Out-Null }
    }
    $full = [System.IO.Path]::GetFullPath($Path)
    if ($full -match ',') { Stop-WithMessage "$What must not contain a comma (docker --mount syntax): $full" }
    return $full
}

# single-quote for the bash -lc command string
function ConvertTo-BashWord([string]$Text) {
    return "'" + $Text.Replace("'", "'\''") + "'"
}

# display only: quote what PowerShell would split
function Format-Word([string]$Text) {
    if ($Text -match '[\s''"`$&|<>;(){}]' -or $Text -eq '') { return "'" + $Text.Replace("'", "''") + "'" }
    return $Text
}

function Get-ContainerConfig {
    # returns @{ Path = <container path>; Mount = <host directory to mount at /config, or $null> }
    if ([System.IO.Path]::IsPathRooted($Config)) {
        if (-not (Test-Path -LiteralPath $Config -PathType Leaf)) { Stop-WithMessage "config not found: $Config" }
        $full = [System.IO.Path]::GetFullPath($Config)
        $repoPrefix = $RepoDir.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
        if ($full.StartsWith($repoPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
            $relative = $full.Substring($repoPrefix.Length).Replace('\', '/')
            return @{ Path = "/opt/fmriproc/$relative"; Mount = $null }
        }
        return @{ Path = '/config/' + (Split-Path -Leaf $full); Mount = (Split-Path -Parent $full) }
    }
    $relative = $Config.Replace('\', '/') -replace '^(\./)+', ''
    if (-not (Test-Path -LiteralPath (Join-Path $RepoDir $relative) -PathType Leaf)) {
        Stop-WithMessage "config not found in the repository: $relative (give a path relative to $RepoDir)"
    }
    return @{ Path = "/opt/fmriproc/$relative"; Mount = $null }
}

function Test-LineEndings {
    # A checkout with CRLF line endings breaks every bash script inside the container.
    foreach ($name in 'run_pipeline.sh', 'lib/common.sh') {
        $file = Join-Path $RepoDir $name
        if ((Test-Path -LiteralPath $file) -and ([System.IO.File]::ReadAllBytes($file) -contains 13)) {
            Write-Warning "$name has CRLF line endings; bash will fail with '\r: command not found'. Re-checkout with LF (README: troubleshooting)."
            return
        }
    }
}

function Test-DockerMemory {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { Stop-WithMessage 'docker was not found on PATH' }
    # A stopped engine still answers with exit status 0, MemTotal 0 and no server version.
    $raw = "$(& docker info --format '{{.MemTotal}}|{{.ServerVersion}}' 2>$null | Select-Object -Last 1)".Trim()
    if ($LASTEXITCODE -ne 0 -or $raw -notmatch '^[1-9]\d*\|.+$') {
        Stop-WithMessage "the Docker engine does not answer ('docker info') - start Docker Desktop first"
    }
    $gb = [math]::Round([double]($raw.Split('|')[0]) / 1GB, 1)
    Write-Host "Docker engine memory: $gb GB"
    if ($gb -lt $MinMemoryGB) {
        Write-Warning ("Docker sees only $gb GB RAM (threshold -MinMemoryGB $MinMemoryGB). recon-all, ANTs SyN and SynthSeg --robust want 8-16 GB. " +
            "Hyper-V backend: Docker Desktop > Settings > Resources > Advanced > Memory, then Apply & restart. " +
            "WSL2 backend: 'memory=16GB' under [wsl2] in $env:USERPROFILE\.wslconfig, 'wsl --shutdown', restart Docker Desktop. " +
            "The pipeline refuses heavy stages below MIN_MEM_GB of the dataset conf.")
    }
}

function Invoke-Docker([string[]]$DockerArgs) {
    Write-Host ''
    Write-Host ('docker ' + (($DockerArgs | ForEach-Object { Format-Word $_ }) -join ' ')) -ForegroundColor Cyan
    Write-Host ''
    if ($PrintOnly) { exit 0 }
    & docker @DockerArgs
    exit $LASTEXITCODE
}

# ----------------------------- export mode ----------------------------------

if ($ExportFreesurfer) {
    $outDir = Resolve-HostDirectory $Out '-Out' -Create
    if (-not $PrintOnly) { Test-DockerMemory }
    $script = 'set -o pipefail; mkdir -p /to/freesurfer_export && ' +
        'tar -C /from --exclude=./fsaverage -chf - . | ' +
        'tar -C /to/freesurfer_export --no-same-owner --no-same-permissions -xf - && ' +
        'echo exported: $(ls /to/freesurfer_export | wc -l) entries in freesurfer_export'
    Write-Host "Copying volume ${VolumePrefix}_freesurfer to $(Join-Path $outDir 'freesurfer_export')"
    Invoke-Docker @(
        'run', '--rm',
        '--mount', "type=volume,source=${VolumePrefix}_freesurfer,target=/from,readonly",
        '--mount', "type=bind,source=$outDir,target=/to",
        $Image, 'bash', '-lc', $script)
}

# ----------------------------- run / shell mode -----------------------------

$outDir = Resolve-HostDirectory $Out '-Out' -Create
if (-not (Test-Path -LiteralPath $License -PathType Leaf)) {
    Stop-WithMessage "FreeSurfer license not found: $License (use -License <file>)"
}
$licenseFile = [System.IO.Path]::GetFullPath($License)
$conf = Get-ContainerConfig

$dockerArgs = @('run', '--rm', '--init')
if ($Shell) { $dockerArgs += '-it' }
if ($Cpus) { $dockerArgs += @('--cpus', $Cpus) }
if ($Memory) { $dockerArgs += @('--memory', $Memory) }
$dockerArgs += @('--mount', "type=bind,source=$RepoDir,target=/opt/fmriproc,readonly")
if ($Data -or -not $Shell) {
    $dataDir = Resolve-HostDirectory $Data '-Data'
    $dockerArgs += @('--mount', "type=bind,source=$dataDir,target=/data,readonly")
}
$dockerArgs += @(
    '--mount', "type=bind,source=$outDir,target=/out",
    '--mount', "type=bind,source=$licenseFile,target=/opt/freesurfer/license.txt,readonly",
    '--mount', "type=volume,source=${VolumePrefix}_freesurfer,target=/out/freesurfer",
    '--mount', "type=volume,source=${VolumePrefix}_work,target=/out/work",
    '--mount', "type=volume,source=${VolumePrefix}_resources,target=/out/resources")
if ($conf.Mount) {
    $dockerArgs += @('--mount', "type=bind,source=$($conf.Mount),target=/config,readonly")
}
# "pwsh -File" hands -Env 'A=1','B=2' over as one literal string: split it here.
$envPairs = @()
foreach ($item in $Env) {
    foreach ($piece in ($item -split ',')) {
        $pair = $piece.Trim().Trim("'").Trim('"')
        if ($pair -eq '') { continue }
        if ($pair -notmatch '^[A-Za-z_][A-Za-z0-9_]*=') { Stop-WithMessage "-Env expects KEY=VALUE, got: $pair" }
        $envPairs += $pair
    }
}
foreach ($pair in $envPairs) {
    $dockerArgs += @('-e', $pair)
}
$dockerArgs += @('-e', "FMRIPROC_CONFIG=$($conf.Path)", '-e', 'FS_LICENSE=/opt/freesurfer/license.txt')

Test-LineEndings
if (-not $PrintOnly) { Test-DockerMemory }

if ($Shell) {
    Write-Host 'Interactive shell; the pipeline is /opt/fmriproc/run_pipeline.sh (FMRIPROC_CONFIG is set).'
    Invoke-Docker ($dockerArgs + @($Image, 'bash', '-l'))
}

$words = @('bash', '/opt/fmriproc/run_pipeline.sh', '-c', (ConvertTo-BashWord $conf.Path))
if ($Subjects) { $words += @('-s', (ConvertTo-BashWord $Subjects)) }
if ($Stages) { $words += @('--stages', (ConvertTo-BashWord $Stages)) }
foreach ($word in $PipelineArgs) {
    if ($word -match '"') { Stop-WithMessage "double quotes cannot be passed through reliably: $word" }
    $words += (ConvertTo-BashWord $word)
}
Invoke-Docker ($dockerArgs + @($Image, 'bash', '-lc', ($words -join ' ')))
