# Local fallback for servers without reliable access to the public S3 bucket.
param([string]$DownloadBase='https://openpi-assets.storage.googleapis.com')
$ErrorActionPreference='Stop'
$taskRoot=Split-Path $PSScriptRoot
$stage=Join-Path $taskRoot 'tmp/pi0_libero.partial'
New-Item -ItemType Directory -Force -Path $stage | Out-Null
$listing=Join-Path $taskRoot 'tmp/pi0-libero-list.xml'
& curl.exe -sSfL --connect-timeout 10 --max-time 60 --retry 2 -o $listing 'https://openpi-assets.s3.amazonaws.com/?list-type=2&prefix=checkpoints/pi0_libero/&max-keys=1000'
if ($LASTEXITCODE -ne 0) { throw 'Cannot retrieve official model file list' }
[xml]$xml=Get-Content -LiteralPath $listing -Raw
if ($xml.ListBucketResult.IsTruncated -eq 'true') { throw 'Unexpected truncated checkpoint listing' }
$pending=New-Object System.Collections.Queue
foreach ($entry in ($xml.ListBucketResult.Contents | Sort-Object {[long]$_.Size} -Descending)) {
    $key=[string]$entry.Key
    if (-not $key.StartsWith('checkpoints/pi0_libero/')) { throw 'Unexpected object prefix' }
    $relative=$key.Substring('checkpoints/pi0_libero/'.Length)
    if ($relative -notmatch '^[A-Za-z0-9_./-]+$' -or $relative.Contains('..')) { throw 'Unsafe object path' }
    $destination=Join-Path $stage $relative
    New-Item -ItemType Directory -Force -Path (Split-Path $destination) | Out-Null
    $pending.Enqueue(@{Entry=$entry;Relative=$relative;Destination=$destination})
}
$running=New-Object System.Collections.ArrayList
$verified=New-Object System.Collections.ArrayList
while ($pending.Count -or $running.Count) {
    while ($pending.Count -and $running.Count -lt 4) {
        $item=$pending.Dequeue()
        $arguments='-sSfL --connect-timeout 15 --max-time 7200 --retry 4 -C - --dump-header "'+$item.Destination+'.headers" -o "'+$item.Destination+'" "'+$DownloadBase+'/'+$item.Entry.Key+'"'
        $item.Process=Start-Process -FilePath curl.exe -ArgumentList $arguments -WindowStyle Hidden -PassThru
        [void]$running.Add($item)
        Write-Output ('DOWNLOADING '+$item.Relative)
    }
    foreach ($item in @($running.ToArray())) {
        if (-not $item.Process.HasExited) { continue }
        $item.Process.WaitForExit()
        if ($item.Process.ExitCode -ne 0) { throw ('Download failed '+$item.Relative) }
        $file=Get-Item -LiteralPath $item.Destination
        if ($file.Length -ne [long]$item.Entry.Size) { throw ('Size mismatch '+$item.Relative) }
        $sha=(Get-FileHash -Algorithm SHA256 -LiteralPath $item.Destination).Hash.ToLowerInvariant()
        $etag=([string]$item.Entry.ETag).Trim('"')
        $mirrorMd5=''
        if ($DownloadBase -like '*storage.googleapis.com*') {
            $headers=Get-Content -LiteralPath ($item.Destination+'.headers') -Raw
            $origin=[regex]::Matches($headers,'(?im)^x-goog-meta-x-goog-source-etag:\s*"([^"\r\n]+)"')
            if (-not $origin.Count -or $origin[$origin.Count-1].Groups[1].Value -ne $etag) { throw ('Mirror origin mismatch '+$item.Relative) }
            $mirror=[regex]::Matches($headers,'(?im)^ETag:\s*"([^"\r\n]+)"')
            if (-not $mirror.Count) { throw 'Missing mirror checksum' }
            $mirrorMd5=$mirror[$mirror.Count-1].Groups[1].Value
            $md5=(Get-FileHash -Algorithm MD5 -LiteralPath $item.Destination).Hash.ToLowerInvariant()
            if ($md5 -ne $mirrorMd5) { throw ('Mirror MD5 mismatch '+$item.Relative) }
        } elseif ($etag.Length -eq 32) {
            $md5=(Get-FileHash -Algorithm MD5 -LiteralPath $item.Destination).Hash.ToLowerInvariant()
            if ($md5 -ne $etag) { throw ('Official ETag mismatch '+$item.Relative) }
        }
        [void]$verified.Add(@{key=[string]$item.Entry.Key;path=$item.Relative;size=$file.Length;sha256=$sha;etag=$etag;mirror_md5=$mirrorMd5})
        [void]$running.Remove($item)
        Write-Output ('VERIFIED '+$item.Relative+' '+$file.Length)
    }
    if ($running.Count) { Start-Sleep -Seconds 1 }
}
$manifest=@{source='s3://openpi-assets/checkpoints/pi0_libero';download_base=$DownloadBase;files=$verified.ToArray();total_bytes=($verified | Measure-Object -Property size -Sum).Sum}
$manifestPath=Join-Path $stage 'download_manifest.json'
[IO.File]::WriteAllText($manifestPath,($manifest | ConvertTo-Json -Depth 5),(New-Object Text.UTF8Encoding($false)))
Write-Output ('DOWNLOAD_COMPLETE '+$stage)
