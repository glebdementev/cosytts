$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$configPath = Join-Path $repoRoot "deploy\config.json"
if (-not (Test-Path $configPath)) {
  throw "Missing deploy config: $configPath."
}

$cfg = Get-Content $configPath -Raw | ConvertFrom-Json

function Get-RequiredValue([string]$name, $value) {
  if ($null -eq $value -or ($value -is [string] -and [string]::IsNullOrWhiteSpace($value))) {
    throw "Missing required config value: $name"
  }
  return $value
}

function Invoke-Step([string]$label, [scriptblock]$action) {
  Write-Host "==> $label"
  $start = Get-Date
  & $action
  $elapsed = (Get-Date) - $start
  $seconds = [Math]::Round($elapsed.TotalSeconds, 2)
  Write-Host "==> Done: $label (${seconds}s)"
}

function Resolve-LocalPath([string]$path) {
  $expanded = [Environment]::ExpandEnvironmentVariables($path)
  if ($expanded.StartsWith("~")) {
    if ([string]::IsNullOrWhiteSpace($HOME)) { throw "HOME is not set; cannot expand '~' in key_path" }
    $expanded = Join-Path $HOME $expanded.Substring(1).TrimStart("\", "/")
  }

  $resolved = Resolve-Path -Path $expanded -ErrorAction Stop
  return $resolved.ProviderPath
}

function ConvertTo-UnixLineEndings([string]$path) {
  if (-not (Test-Path $path)) {
    throw "Missing file to normalize: $path"
  }
  $content = [IO.File]::ReadAllText($path)
  $normalized = $content.Replace("`r`n", "`n").Replace("`r", "`n")
  if ($normalized -ne $content) {
    $utf8NoBom = New-Object System.Text.UTF8Encoding $false
    [IO.File]::WriteAllText($path, $normalized, $utf8NoBom)
    Write-Host "Normalized line endings: $path"
  }
}

function Resolve-PythonCommand {
  $candidates = @("python", "python3")
  foreach ($name in $candidates) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if ($cmd) {
      return @($cmd.Source)
    }
  }

  $py = Get-Command "py" -ErrorAction SilentlyContinue
  if ($py) {
    return @($py.Source, "-3")
  }

  throw "Python not found. Install it or ensure python/python3/py is on PATH."
}

$SshHost = Get-RequiredValue "ssh.host" $cfg.ssh.host
$SshUser = Get-RequiredValue "ssh.user" $cfg.ssh.user
$SshPort = [int](Get-RequiredValue "ssh.port" $cfg.ssh.port)
$KeyPath = $cfg.ssh.key_path
$RemoteDir = Get-RequiredValue "remote.dir" $cfg.remote.dir
$RemoteComposeFile = "api/docker-compose.streaming.yml"
$ApiPort = $env:TTS_API_PORT
if ([string]::IsNullOrWhiteSpace($ApiPort)) {
  $ApiPort = "8093"
}

$target = "${SshUser}@${SshHost}"

Write-Host "Deploying to ${target}:${RemoteDir} ..."
Write-Host "Using config: $configPath"

$sshBase = @("-o", "StrictHostKeyChecking=no", "-p", "$SshPort")
$scpArgs = @("-o", "StrictHostKeyChecking=no", "-P", "$SshPort")
if ($KeyPath) {
  $resolvedKeyPath = Resolve-LocalPath $KeyPath
  $sshBase += @("-i", $resolvedKeyPath)
  $scpArgs += @("-i", $resolvedKeyPath)
}
# -n prevents SSH from reading stdin, which causes hangs on Windows
$sshArgs = @("-n") + $sshBase

Invoke-Step "Check SSH connectivity" {
  & ssh @sshArgs $target "echo ok"
  if ($LASTEXITCODE -ne 0) { throw "Cannot connect to ${target} via SSH." }
}

$archivePath = Join-Path $repoRoot "deploy.tgz"
if (Test-Path $archivePath) { Remove-Item $archivePath -Force }

Invoke-Step "Normalize deploy scripts line endings" {
  $bootstrapPath = Join-Path $repoRoot "deploy\remote_bootstrap.sh"
  ConvertTo-UnixLineEndings $bootstrapPath
}

Invoke-Step "Create archive" {
  Push-Location $repoRoot
  try {
    $tarVersion = & tar --version 2>$null | Select-Object -First 1
    $tarArgs = @(
      "-czf", $archivePath,
      "--exclude", ".git",
      "--exclude", "deploy/config.json",
      "--exclude", ".venv",
      "--exclude", "__pycache__",
      "--exclude", "deploy.tgz",
      "--exclude", "models",
      "--exclude", "*.onnx",
      "--exclude", "*.plan",
      "--exclude", "asr_example",
      "--exclude", "deprecated",
      "."
    )
    if ($tarVersion -match "GNU tar") {
      $tarArgs = @("--warning=no-file-changed", "--ignore-failed-read") + $tarArgs
    }
    & tar @tarArgs
    if ($LASTEXITCODE -ne 0) { throw "tar failed" }
  } finally {
    Pop-Location
  }
}

Invoke-Step "Prepare remote directory" {
  # Preserve models directory to avoid re-downloading large model files
  & ssh @sshArgs $target "find $RemoteDir -mindepth 1 -maxdepth 1 ! -name 'models' -exec rm -rf {} + 2>/dev/null; mkdir -p $RemoteDir"
  if ($LASTEXITCODE -ne 0) { throw "ssh mkdir failed" }
}

Invoke-Step "Upload archive" {
  & scp @scpArgs $archivePath "${target}:${RemoteDir}/deploy.tgz"
  if ($LASTEXITCODE -ne 0) { throw "scp failed" }
}

Invoke-Step "Extract and bootstrap on remote (may take a while)" {
  Write-Host "Remote output follows:"
  & ssh @sshArgs $target "cd $RemoteDir && tar -xzf deploy.tgz && rm -f deploy.tgz && bash deploy/remote_bootstrap.sh"
  if ($LASTEXITCODE -ne 0) { throw "remote bootstrap failed" }
}

Invoke-Step "Verify health endpoint (remote localhost)" {
  $healthUrl = "http://localhost:${ApiPort}/health"
  for ($attempt = 1; $attempt -le 20; $attempt++) {
    $response = & ssh @sshArgs $target "curl -fsS $healthUrl"
    if ($LASTEXITCODE -eq 0 -and $response -match '"status"\s*:\s*"ok"') {
      Write-Host "Health OK (remote): $healthUrl"
      return
    }
    Start-Sleep -Seconds 3
  }
  Write-Host "Health check failed (remote): $healthUrl"
  Write-Host "Remote compose status:"
  & ssh @sshArgs $target "cd $RemoteDir && docker compose -f $RemoteComposeFile ps"
  Write-Host "Remote compose logs (api):"
  & ssh @sshArgs $target "cd $RemoteDir && docker compose -f $RemoteComposeFile logs --tail=200 api"
  throw "Service did not become healthy."
}

Invoke-Step "Verify streaming synthesis endpoint (remote from local)" {
  $voiceName = $env:TTS_TEST_VOICE
  if ([string]::IsNullOrWhiteSpace($voiceName)) {
    $voiceName = "polina"
  }
  $requestBody = @{
    text = "Streaming synthesis test from deploy script."
    voice = $voiceName
  } | ConvertTo-Json -Depth 5 -Compress

  $requestJsonPath = Join-Path $repoRoot "deploy\_stream_test_request.json"
  $outputPcmPath = Join-Path $repoRoot "deploy\_stream_test_output.pcm"

  try {
    [IO.File]::WriteAllText($requestJsonPath, $requestBody, [System.Text.Encoding]::UTF8)
    if (Test-Path $outputPcmPath) {
      Remove-Item $outputPcmPath -Force
    }

    $streamUrl = "http://${SshHost}:${ApiPort}/synthesize/stream"
    & curl.exe -fsS `
      -X POST $streamUrl `
      -H "Content-Type: application/json" `
      --data-binary "@$requestJsonPath" `
      --output $outputPcmPath
    if ($LASTEXITCODE -ne 0) { throw "Streaming synthesis request failed." }

    $outInfo = Get-Item $outputPcmPath
    if ($outInfo.Length -le 0) {
      throw "Streaming synthesis produced empty output."
    }
    Write-Host "Streaming synthesis OK: wrote $($outInfo.Length) bytes to $outputPcmPath"
  }
  finally {
    if (Test-Path $requestJsonPath) {
      Remove-Item $requestJsonPath -Force
    }
  }
}
