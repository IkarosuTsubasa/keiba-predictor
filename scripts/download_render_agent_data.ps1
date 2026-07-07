param(
    [string]$WorkspaceRoot = "D:\keiba\new",
    [string]$SnapshotName = "",
    [string]$SshTarget = "srv-d6qm9j9aae7s739j7kkg@ssh.oregon.render.com",
    [string]$RemoteProjectRoot = "/opt/render/project/src",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if (-not $SnapshotName) {
    $SnapshotName = "render_agent_snapshot_" + (Get-Date -Format "yyyyMMdd_HHmmss")
}

$workspace = Resolve-Path -LiteralPath $WorkspaceRoot
$snapshotRoot = Join-Path $workspace "analysis_outputs"
$snapshotDir = Join-Path $snapshotRoot $SnapshotName

if (Test-Path -LiteralPath $snapshotDir) {
    $existing = Get-ChildItem -LiteralPath $snapshotDir -Force -ErrorAction SilentlyContinue
    if ($existing.Count -gt 0) {
        throw "Snapshot directory already exists and is not empty: $snapshotDir"
    }
}

$items = @(
    @{ Remote = "pipeline/data/agent_predictions"; Local = "agent_predictions"; Optional = $false },
    @{ Remote = "pipeline/data/agent_results"; Local = "agent_results"; Optional = $false },
    @{ Remote = "pipeline/data/agent_backtest_audit"; Local = "agent_backtest_audit"; Optional = $true }
)

if ($DryRun) {
    Write-Output "DRY RUN snapshot_dir=$snapshotDir"
} else {
    New-Item -ItemType Directory -Force -Path $snapshotDir | Out-Null
}

foreach ($item in $items) {
    $remotePath = "$RemoteProjectRoot/$($item.Remote)"
    $localPath = Join-Path $snapshotDir $item.Local
    $existsCommand = "test -d '$remotePath' && echo exists || echo missing"
    $sshArgs = @("-o", "BatchMode=yes", "-o", "UpdateHostKeys=no", $SshTarget, $existsCommand)

    if ($DryRun) {
        Write-Output ("ssh " + ($sshArgs -join " "))
        Write-Output "scp -o BatchMode=yes -o UpdateHostKeys=no -r ${SshTarget}:$remotePath $snapshotDir"
        continue
    }

    $existsOutput = & ssh @sshArgs
    if (($existsOutput -join "`n") -notmatch "exists") {
        if ([bool]$item.Optional) {
            Write-Warning "Remote optional directory missing: $remotePath"
            continue
        }
        throw "Remote required directory missing: $remotePath"
    }

    if (Test-Path -LiteralPath $localPath) {
        throw "Local target already exists; refusing to overwrite: $localPath"
    }

    $scpArgs = @("-o", "BatchMode=yes", "-o", "UpdateHostKeys=no", "-r", "${SshTarget}:$remotePath", $snapshotDir)
    & scp @scpArgs
    if ($LASTEXITCODE -ne 0) {
        throw "scp failed for $remotePath"
    }
}

Write-Output "snapshot_dir=$snapshotDir"
