[CmdletBinding()]
param([string]$Command,[string]$ConfigPath='',[string]$Commit='',
      [string]$ExperimentConfig='',[string]$SmokeJobId='',[string]$RestorePath='',[string]$DependencyJobId='')
$ErrorActionPreference='Stop'
if (-not $ConfigPath) { $ConfigPath=Join-Path $PSScriptRoot 'remote.sft.json' }
if (-not $ExperimentConfig) { $ExperimentConfig='configs/sft_a_v1.json' }
$machine=Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
$stage=@{'sft-data'='data';'sft-align'='align';'sft-smoke'='profile';'sft-train'='train';'sft-eval'='evaluate'}[$Command]
if (-not $stage) { throw 'Unknown SFT stage.' }
$required=if ($stage -eq 'data') {'sft_data.py'} else {'sft_native.py'}
if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot $required))) { throw "SFT stage $stage is not implemented." }
if (-not $Commit) { $Commit=(git rev-parse HEAD).Trim(); if ($LASTEXITCODE -ne 0) { throw 'Cannot read commit.' } }
if ($Commit -notmatch '^[a-f0-9]{40}$') { throw 'Commit must be a full SHA.' }
if ($stage -ne 'data' -and $SmokeJobId -notmatch '^\d+$') { throw 'Provide the successful prerequisite job with -SmokeJobId.' }
if ($DependencyJobId -and $DependencyJobId -notmatch '^\d+$') { throw 'DependencyJobId must be numeric.' }
& (Join-Path $PSScriptRoot 'remote.ps1') sync -ConfigPath $ConfigPath -Commit $Commit
if ($LASTEXITCODE -ne 0) { throw 'Code synchronization failed.' }
function Q([string]$Value) { return "'"+$Value.Replace("'", "'\''")+"'" }
$repo=$machine.WorktreeRoot+'/'+$Commit.Substring(0,12)
$slurm=$machine.Slurm
$argsList=@('sbatch','--parsable',('--partition='+$slurm.Partition),('--account='+$slurm.Account),('--qos='+$slurm.Qos),('--gres='+$slurm.Gres),('--cpus-per-task='+$slurm.Cpus),('--time='+$machine.SftTimes.$stage),('--job-name=sft-a-'+$stage),('--output='+$machine.DataRoot+'/logs/slurm/%j.log'))
if ($slurm.Exclude) { $argsList+=('--exclude='+$slurm.Exclude) }
if ($slurm.NodeList) {
    if ($slurm.NodeList -match '4090node1' -and $stage -notin @('data','profile')) { throw 'node1 is allowed only for short data/profile debugging.' }
    $argsList+=('--nodelist='+$slurm.NodeList)
}
if ($DependencyJobId) { $argsList+=('--dependency=afterok:'+$DependencyJobId) }
$argsList+=@(($repo+'/tools/sft_job.sh'),$repo,$machine.CondaRoot,$machine.CondaEnv,$machine.DataRoot,$stage,$ExperimentConfig,$machine.SftEnvironmentPrefix,$machine.RLinfRoot,$SmokeJobId,$RestorePath,$machine.BashSetupPath)
$lines=@('set -euo pipefail',('source '+(Q $machine.BashSetupPath)),('cd '+(Q $repo)),('mkdir -p '+(Q ($machine.DataRoot+'/logs/slurm'))),'bash -n tools/sft_job.sh', (($argsList | ForEach-Object {Q $_}) -join ' '))
$encoded=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes(($lines -join "`n")+"`n"))
& ssh -o BatchMode=yes -o ConnectTimeout=15 $machine.SshHost "echo $encoded | base64 -d | bash"
if ($LASTEXITCODE -ne 0) { throw 'SFT submission failed.' }
