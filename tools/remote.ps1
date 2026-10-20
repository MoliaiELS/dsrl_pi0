[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('doctor', 'install-env', 'sync', 'env', 'smoke', 'eval', 'train', 'probe-smoke', 'probe', 'outcome-smoke', 'outcome', 'status', 'logs')]
    [string]$Command = 'doctor',
    [Parameter(Position=1)][string]$JobId='',
    [string]$ConfigPath = '', [string]$Commit='', [int]$Seed=0,
    [string]$ExperimentConfig='',
    [string]$SmokeJobId='', [string]$BaseJobId='', [string]$RestorePath='',
    [string]$DependencyJobId=''
)
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'remote_jobs.ps1') @PSBoundParameters
exit
