@echo off
setlocal
set "A32_FIX_SCRIPT=%~f0"
set "A32_PS=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%A32_PS%" set "A32_PS=%SystemRoot%\Sysnative\WindowsPowerShell\v1.0\powershell.exe"
if not exist "%A32_PS%" (
  echo Windows PowerShell is missing. No SSH configuration was changed.
  exit /b 1
)
"%A32_PS%" -NoLogo -NoProfile -Command "$s=[IO.File]::ReadAllText($env:A32_FIX_SCRIPT); $m='# A32_POWERSHELL_BEGIN'; & ([scriptblock]::Create($s.Substring($s.LastIndexOf($m)+$m.Length)))"
exit /b %ERRORLEVEL%
# A32_POWERSHELL_BEGIN
$ErrorActionPreference = 'Stop'
$temporaryConfig = $null
try {
    function Find-OpenSSH([string]$Name) {
        $found = Get-Command ($Name + '.exe') -ErrorAction SilentlyContinue
        if ($found) { return $found.Source }
        foreach ($base in @('System32', 'Sysnative')) {
            $candidate = Join-Path $env:SystemRoot ($base + '\OpenSSH\' + $Name + '.exe')
            if (Test-Path -LiteralPath $candidate) { return $candidate }
        }
        throw ($Name + '.exe was not found.')
    }
    $ssh = Find-OpenSSH 'ssh'
    $keygen = Find-OpenSSH 'ssh-keygen'
    $directory = Join-Path $env:USERPROFILE '.ssh'
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
    $key = Join-Path $directory 'a100key'
    if (!(Test-Path -LiteralPath $key)) {
        Write-Host 'Creating a dedicated SSH key. Existing keys are preserved.'
        $arguments = '-t ed25519 -f "' + $key + '" -N "" -C a100-windows'
        $process = Start-Process -FilePath $keygen -ArgumentList $arguments -Wait -NoNewWindow -PassThru
        if ($process.ExitCode -ne 0) { throw 'SSH key generation failed.' }
    }
    if (!(Test-Path -LiteralPath ($key + '.pub'))) {
        throw ('The existing private key has no matching public file: ' + $key + '.pub')
    }
    $publicKey = (Get-Content -LiteralPath ($key + '.pub') -Raw).Trim()
    if ($publicKey -notmatch '^ssh-ed25519 [A-Za-z0-9+/]+={0,3}(\s|$)') {
        throw 'The selected public key is not a valid Ed25519 public key.'
    }
    Write-Host 'Enter the A100 server password when prompted. Only the public key is uploaded.'
    $install = "umask 077; mkdir -p .ssh; chmod 700 .ssh; tmp=`$(mktemp .ssh/.a100-public.XXXXXX) || exit 1; tr -d '\r' > `$tmp; test -s `$tmp || { rm -f `$tmp; exit 1; }; touch .ssh/authorized_keys; chmod 600 .ssh/authorized_keys; grep -Fqx -f `$tmp .ssh/authorized_keys || cat `$tmp >> .ssh/authorized_keys; result=`$?; rm -f `$tmp; exit `$result"
    $publicKey | & $ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -l houj 115.191.75.48 $install
    if ($LASTEXITCODE -ne 0) { throw 'Public-key installation failed. The Windows SSH config was not changed.' }

    $config = Join-Path $directory 'config'
    $original = ''
    if (Test-Path -LiteralPath $config) {
        $original = [IO.File]::ReadAllText($config)
        $backup = $config + '.before-a100-' + (Get-Date -Format yyyyMMdd-HHmmss-fff)
        Copy-Item -LiteralPath $config -Destination $backup
        Write-Host ('Existing config backed up to: ' + $backup)
    }
    # Remove broken IdentityFile directives only in blocks exclusively naming
    # these two A100 aliases. Preserve all other hosts and configuration.
    $aliasesToFix = @('volc-a32', 'volc-a32-key')
    $inA100Block = $false
    $kept = New-Object 'System.Collections.Generic.List[string]'
    foreach ($line in ($original -split '\r?\n')) {
        if ($line -match '^\s*Host\s+(.+)$') {
            $aliases = @(($Matches[1] -replace '\s+#.*$', '').Trim() -split '\s+')
            $inA100Block = $aliases.Count -gt 0
            foreach ($alias in $aliases) {
                if ($aliasesToFix -notcontains $alias) { $inA100Block = $false }
            }
        } elseif ($line -match '^\s*Match\s+') {
            $inA100Block = $false
        }
        if ($inA100Block -and $line -match '^\s*IdentityFile\s+') { continue }
        $kept.Add($line)
    }
    $identity = $key.Replace('\', '/')
    $header = @(
        '# A100 connection repaired by fix_windows_ssh.cmd',
        'Host volc-a32 volc-a32-key',
        '    HostName 115.191.75.48',
        '    User houj',
        '    Port 22',
        ('    IdentityFile "' + $identity + '"'),
        '    IdentitiesOnly yes',
        '    ServerAliveInterval 30',
        '    ServerAliveCountMax 3',
        '',
        'Host *',
        ''
    ) -join "`r`n"
    $updated = $header + ($kept -join "`r`n") + "`r`n"
    $temporaryConfig = Join-Path $directory ('a100-test-' + [guid]::NewGuid().ToString('N'))
    $encoding = New-Object System.Text.UTF8Encoding -ArgumentList $false
    [IO.File]::WriteAllText($temporaryConfig, $updated, $encoding)
    Write-Host 'Verifying public-key login before replacing the SSH config...'
    $result = & $ssh -F $temporaryConfig -o BatchMode=yes -o ConnectTimeout=15 volc-a32 whoami
    if ($LASTEXITCODE -ne 0 -or ($result -join "`n").Trim() -ne 'houj') {
        throw 'SSH verification failed. Your original config is preserved.'
    }
    [IO.File]::WriteAllText($config, $updated, $encoding)
    Write-Host 'SUCCESS: SSH public-key login verified as houj.'
    Write-Host 'In Codex, use SSH host alias: volc-a32 (without the word ssh).'
} catch {
    Write-Host ('ERROR: ' + $_.Exception.Message) -ForegroundColor Red
    exit 1
} finally {
    if ($temporaryConfig -and (Test-Path -LiteralPath $temporaryConfig)) {
        Remove-Item -LiteralPath $temporaryConfig
    }
}
