[CmdletBinding()]
param(
    [Parameter(Position=0)]
    [ValidateSet('doctor','install-env','sync','env','smoke','eval','train','status','logs')]
    [string]$Command='doctor',
    [Parameter(Position=1)][string]$JobId='',
    [string]$ConfigPath='', [string]$Commit='', [int]$Seed=0,
    [string]$ExperimentConfig='configs/libero_baseline.json',
    [string]$SmokeJobId='', [string]$BaseJobId='', [string]$RestorePath='',
    [switch]$PrintScript
)
$ErrorActionPreference='Stop'
if (-not $ConfigPath) { $ConfigPath=Join-Path $PSScriptRoot 'remote.json' }
$config=Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
function Quote-Bash([string]$Value) { return "'"+$Value.Replace("'", "'\''")+"'" }
$script=@('set -euo pipefail')
if ($Command -in @('status','logs')) {
    if ($JobId -notmatch '^\d+$') { throw 'Provide an explicit numeric job ID.' }
    $script += 'record='+ (Quote-Bash ($config.DataRoot+'/runs/job-'+$JobId))
    if ($Command -eq 'status') {
        $script += @(
            ('squeue -j '+$JobId+' -o "%.18i %.10T %.12M %.20R"'),
            ('sacct -j '+$JobId+' --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES -P'),
            'for f in run.json latest.json SUCCESS.json FAILED.json; do if test -f "$record/$f"; then echo "$f"; cat "$record/$f"; fi; done',
            'if test -f "$record/resources.jsonl"; then tail -2 "$record/resources.jsonl"; fi'
        )
    } else {
        $script += ('tail -80 '+(Quote-Bash ($config.DataRoot+'/logs/slurm/'+$JobId+'.log')))
    }
} else {
    if (-not $Commit) { $Commit=(& git -C (Split-Path $PSScriptRoot) rev-parse HEAD).Trim() }
    if ($Commit -notmatch '^[a-f0-9]{40}$') { throw 'Commit must be a full SHA.' }
    $repo=$config.WorktreeRoot+'/'+$Commit.Substring(0,12)
    $script += @(
        ('source '+(Quote-Bash ($config.CondaRoot+'/etc/profile.d/conda.sh'))),
        ('conda activate '+(Quote-Bash $config.CondaEnv)),
        ('export DSRL_DATA_ROOT='+(Quote-Bash $config.DataRoot))
    )
    if ($Command -eq 'sync') {
        $script += @(
            ('cd '+(Quote-Bash $config.SourceRepo)),
            'git status --short',
            ('git fetch '+(Quote-Bash $config.RemoteRepoUrl)+' '+(Quote-Bash $Commit)),
            ('test "$(git rev-parse FETCH_HEAD)" = '+(Quote-Bash $Commit)),
            ('if ! test -d '+(Quote-Bash $repo)+'; then git worktree add --detach '+(Quote-Bash $repo)+' '+(Quote-Bash $Commit)+'; fi'),
            ('cd '+(Quote-Bash $repo)),
            'test -z "$(git status --porcelain --untracked-files=no)"',
            ('git -c protocol.file.allow=always -c '+(Quote-Bash ('submodule.LIBERO.url='+$config.RepoDir+'/LIBERO'))+' -c '+(Quote-Bash ('submodule.openpi.url='+$config.RepoDir+'/openpi'))+' submodule update --init'),
            ('git -C openpi -c protocol.file.allow=always -c '+(Quote-Bash ('submodule.third_party/aloha.url='+$config.RepoDir+'/openpi/third_party/aloha'))+' -c '+(Quote-Bash ('submodule.third_party/libero.url='+$config.RepoDir+'/openpi/third_party/libero'))+' submodule update --init'),
            'git rev-parse HEAD', 'git submodule status --recursive', 'bash -n tools/job.sh tools/remote-env.sh',
            'python -m compileall -q tools examples jaxrl2'
        )
    } elseif ($Command -in @('doctor','install-env')) {
        $script += @(('cd '+(Quote-Bash $repo)), 'source tools/remote-env.sh')
        if ($Command -eq 'doctor') { $script += 'python -u tools/check_env.py' }
        else { $script += 'bash tools/install-env.sh' }
    } else {
        if ($Command -eq 'train') {
            if ($SmokeJobId -notmatch '^\d+$' -or $BaseJobId -notmatch '^\d+$') { throw 'Train requires -SmokeJobId and -BaseJobId.' }
            foreach ($gate in @($SmokeJobId,$BaseJobId)) {
                $script += 'test -f '+(Quote-Bash ($config.DataRoot+'/runs/job-'+$gate+'/SUCCESS.json'))
                $script += 'python -c '+(Quote-Bash ('import json; from pathlib import Path; d=json.loads(Path("'+$config.DataRoot+'/runs/job-'+$gate+'/run.json").read_text()); assert d["commit"]=="'+$Commit+'",d'))
            }
        }
        $mode=$Command
        if ($mode -eq 'eval') { $mode=if ($RestorePath) {'eval'} else {'base'} }
        $limit=if ($Command -in @('env','smoke')) {$config.Slurm.SmokeTime} elseif ($Command -eq 'eval') {$config.Slurm.EvalTime} else {$config.Slurm.TrainTime}
        $script += 'mkdir -p '+(Quote-Bash ($config.DataRoot+'/logs/slurm'))+' '+(Quote-Bash ($config.DataRoot+'/runs'))
        $arguments=@('sbatch --parsable', ('--partition='+(Quote-Bash $config.Slurm.Partition)),
            ('--account='+(Quote-Bash $config.Slurm.Account)), ('--qos='+(Quote-Bash $config.Slurm.Qos)),
            ('--gres='+(Quote-Bash $config.Slurm.Gres)), ('--cpus-per-task='+$config.Slurm.Cpus),
            ('--exclude='+(Quote-Bash $config.Slurm.Exclude)), ('--time='+(Quote-Bash $limit)),
            ('--job-name='+(Quote-Bash ('dsrl-'+$mode+'-s'+$Seed))),
            ('--output='+(Quote-Bash ($config.DataRoot+'/logs/slurm/%j.log'))),
            (Quote-Bash ($repo+'/tools/job.sh')), (Quote-Bash $repo),
            (Quote-Bash $config.CondaRoot), (Quote-Bash $config.CondaEnv),
            (Quote-Bash $config.DataRoot), (Quote-Bash $mode),
            (Quote-Bash $ExperimentConfig), [string]$Seed, (Quote-Bash $RestorePath))
        $script += 'job=$('+($arguments -join ' ')+')'
        $script += 'echo "JOB_ID=$job"'
        $script += 'echo "LOG='+$config.DataRoot+'/logs/slurm/$job.log"'
        $script += 'echo "RESULT='+$config.DataRoot+'/runs/job-$job"'
    }
}
if ($PrintScript) { $script; exit }
$payload=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes(($script -join "`n")+"`n"))
& ssh -o BatchMode=yes -o ConnectTimeout=15 $config.SshHost "echo $payload | base64 -d | bash"
if ($LASTEXITCODE -ne 0) { throw "Remote $Command failed (exit $LASTEXITCODE)." }
