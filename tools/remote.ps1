[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('doctor', 'install-env', 'sync', 'env', 'smoke', 'eval', 'train', 'probe-smoke', 'probe', 'outcome-smoke', 'outcome', 'research-smoke', 'research', 'status', 'logs','sft-data','sft-align','sft-smoke','sft-train','sft-eval')]
    [string]$Command = 'doctor',
    [Parameter(Position=1)][string]$JobId='',
    [string]$ConfigPath = '', [string]$Commit='', [int]$Seed=0,
    [string]$ExperimentConfig='',
    [string]$SmokeJobId='', [string]$BaseJobId='', [string]$RestorePath='',
    [string]$DependencyJobId='', [switch]$DeferredSmokeGate
)
$ErrorActionPreference = 'Stop'
if ($Command.StartsWith('sft-')) {
    & (Join-Path $PSScriptRoot 'sft_remote.ps1') -Command $Command -ConfigPath $ConfigPath -Commit $Commit -ExperimentConfig $ExperimentConfig -SmokeJobId $SmokeJobId -RestorePath $RestorePath -DependencyJobId $DependencyJobId
    exit
}
& (Join-Path $PSScriptRoot 'remote_jobs.ps1') @PSBoundParameters
exit
