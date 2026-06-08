param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("predict", "review", "validate", "backtest")]
    [string]$Action,

    [string]$InputFile,

    [string]$FromDate,

    [string]$ToDate,

    [ValidateSet("central", "central_turf", "central_dirt", "local")]
    [string]$ScopeKey,

    [int]$SleepSeconds = 5,

    [int]$RecentRunLimit = 5,

    [ValidateSet("mock", "openai")]
    [string]$LlmProvider = "mock",

    [switch]$ForceRefresh,

    [switch]$SkipReport,

    [switch]$EnableReport,

    [switch]$SkipSocial,

    [switch]$StopOnError
)

$ErrorActionPreference = "Stop"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptRoot
Set-Location $ProjectRoot

function Get-InputLines {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path $Path)) {
        throw "Input file not found: $Path"
    }

    return Get-Content $Path |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ -and (-not $_.StartsWith("#")) }
}

function Convert-ToRaceId {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    if ($Value -match '^\d{12}$') {
        return $Value
    }
    if ($Value -match 'race_id=(\d{12})') {
        return $Matches[1]
    }
    throw "Invalid race input. Use netkeiba URL with race_id or 12-digit race_id: $Value"
}

function Get-NetkeibaSourceFromRaceId {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RaceId
    )

    $venueCode = $RaceId.Substring(4, 2)
    $centralCodes = @("01", "02", "03", "04", "05", "06", "07", "08", "09", "10")
    $localCodes = @("30", "35", "36", "42", "43", "44", "45", "46", "47", "48", "50", "51", "54", "55")
    if ($centralCodes -contains $venueCode) {
        return "central"
    }
    if ($localCodes -contains $venueCode) {
        return "local"
    }
    throw "Cannot infer netkeiba source from race_id venue code '$venueCode': $RaceId"
}

function Convert-ToShutubaUrl {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    if ($Value -match '^https?://') {
        return $Value
    }
    if ($Value -match '^\d{12}$') {
        $source = Get-NetkeibaSourceFromRaceId -RaceId $Value
        if ($source -eq "local") {
            return "https://nar.netkeiba.com/race/shutuba.html?race_id=$Value"
        }
        return "https://race.netkeiba.com/race/shutuba.html?race_id=$Value"
    }
    throw "Invalid predict input. Use netkeiba shutuba URL or 12-digit race_id: $Value"
}

function Convert-ToResultUrl {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    if ($Value -match '^https?://') {
        return $Value
    }
    if ($Value -match '^\d{12}$') {
        $source = Get-NetkeibaSourceFromRaceId -RaceId $Value
        if ($source -eq "local") {
            return "https://nar.netkeiba.com/race/result.html?race_id=$Value"
        }
        return "https://race.netkeiba.com/race/result.html?race_id=$Value"
    }
    throw "Invalid review input. Use netkeiba result URL or 12-digit race_id: $Value"
}

function Get-RaceIdsFromRaceData {
    $raceDataDir = Join-Path $ProjectRoot "keiba_llm_agent\data\race_data"
    if (-not (Test-Path $raceDataDir)) {
        return @()
    }

    return Get-ChildItem $raceDataDir -Filter *.json |
        Sort-Object Name |
        ForEach-Object { $_.BaseName }
}

function Invoke-KeibaCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    Write-Host ("[RUN] python " + ($Arguments -join " ")) -ForegroundColor Cyan
    & python @Arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        if ($StopOnError) {
            throw "Command failed: python $($Arguments -join ' ')"
        }
        Write-Warning "Command failed: python $($Arguments -join ' ')"
    }
}

function Invoke-BulkPredict {
    if (-not $InputFile) {
        throw "predict requires -InputFile."
    }

    $entries = Get-InputLines -Path $InputFile
    foreach ($entry in $entries) {
        $url = Convert-ToShutubaUrl -Value $entry
        $commandArgs = @(
            "-m", "keiba_llm_agent.main",
            "predict-race",
            "--url", $url,
            "--recent-run-limit", $RecentRunLimit.ToString(),
            "--llm-provider", $LlmProvider
        )
        if ($ForceRefresh) { $commandArgs += "--force-refresh" }
        if (-not $EnableReport) { $commandArgs += "--skip-report" }
        elseif ($SkipReport) { $commandArgs += "--skip-report" }
        if ($SkipSocial) { $commandArgs += "--skip-social" }
        Invoke-KeibaCommand -Arguments $commandArgs
        Start-Sleep -Seconds $SleepSeconds
    }
}

function Invoke-BulkReview {
    if (-not $InputFile) {
        throw "review requires -InputFile."
    }

    $entries = Get-InputLines -Path $InputFile
    foreach ($entry in $entries) {
        $url = Convert-ToResultUrl -Value $entry
        $commandArgs = @(
            "-m", "keiba_llm_agent.main",
            "review-race",
            "--url", $url,
            "--llm-provider", $LlmProvider
        )
        if ($ForceRefresh) { $commandArgs += "--force-refresh" }
        if (-not $EnableReport) { $commandArgs += "--skip-report" }
        elseif ($SkipReport) { $commandArgs += "--skip-report" }
        if ($SkipSocial) { $commandArgs += "--skip-social" }
        Invoke-KeibaCommand -Arguments $commandArgs
        Start-Sleep -Seconds $SleepSeconds
    }
}

function Invoke-BulkValidate {
    $raceIds = @()
    if ($InputFile) {
        $raceIds = Get-InputLines -Path $InputFile
    } else {
        $raceIds = Get-RaceIdsFromRaceData
    }

    foreach ($entry in $raceIds) {
        $raceId = Convert-ToRaceId -Value $entry
        Invoke-KeibaCommand -Arguments @(
            "-m", "keiba_llm_agent.main",
            "validate-race-data",
            "--race-id", $raceId
        )
    }
}

function Invoke-Backtest {
    if (-not $FromDate -or -not $ToDate) {
        throw "backtest requires -FromDate and -ToDate."
    }

    $commandArgs = @(
        "-m", "keiba_llm_agent.main",
        "backtest",
        "--from-date", $FromDate,
        "--to-date", $ToDate
    )
    if ($ScopeKey) {
        $commandArgs += @("--scope-key", $ScopeKey)
    }
    Invoke-KeibaCommand -Arguments $commandArgs
}

switch ($Action) {
    "predict" { Invoke-BulkPredict }
    "review" { Invoke-BulkReview }
    "validate" { Invoke-BulkValidate }
    "backtest" { Invoke-Backtest }
    default { throw "Unsupported action: $Action" }
}
