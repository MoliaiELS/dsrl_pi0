[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('doctor', 'install-env')]
    [string]$Command = 'doctor',
    [string]$ConfigPath = ''
)
$ErrorActionPreference = 'Stop'
if (-not $ConfigPath) {
    $ConfigPath = Join-Path $PSScriptRoot 'remote.json'
}
$config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json

function Quote-Bash([string]$Value) {
    return "'" + $Value.Replace("'", "'\''") + "'"
}

$script = @(
    'set -euo pipefail',
    ('source ' + (Quote-Bash ($config.CondaRoot + '/etc/profile.d/conda.sh'))),
    ('conda activate ' + (Quote-Bash $config.CondaEnv)),
    ('cd -- ' + (Quote-Bash $config.RepoDir)),
    ('export DSRL_DATA_ROOT=' + (Quote-Bash $config.DataRoot)),
    'source tools/remote-env.sh'
)
if ($Command -eq 'install-env') {
    $script += 'bash tools/install-env.sh'
} else {
    $script += @(
        'record="$DSRL_DATA_ROOT/logs/environment/doctor-$(date +%Y%m%dT%H%M%S)"',
        'mkdir -p "$record"',
        'echo "Environment record: $record"',
        'python -u tools/check_env.py 2>&1 | tee "$record/doctor.log"',
        'python -m pip freeze --all > "$record/pip-freeze.txt"',
        'git rev-parse HEAD > "$record/commit.txt"',
        'git submodule status > "$record/submodules.txt"'
    )
}
$payload = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes(($script -join "`n") + "`n"))
& ssh -o BatchMode=yes -o ConnectTimeout=15 $config.SshHost "echo $payload | base64 -d | bash"
if ($LASTEXITCODE -ne 0) {
    throw "Remote $Command failed (exit $LASTEXITCODE)."
}
