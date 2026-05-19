# Load .env file into PowerShell environment
$envFile = ".env"

if (Test-Path $envFile) {
    Write-Host "Loading environment variables from $envFile..." -ForegroundColor Green
    
    $content = Get-Content $envFile | Where-Object { $_ -and -not $_.StartsWith("#") }
    
    foreach ($line in $content) {
        if ($line -like "*=*") {
            $key, $value = $line -split "=", 2
            $key = $key.Trim()
            $value = $value.Trim() -replace '^["'']|["'']$'
            
            if ($key) {
                Set-Item -Path "env:$key" -Value $value
                Write-Host "Set $key" -ForegroundColor Yellow
            }
        }
    }
    Write-Host ".env variables loaded successfully!" -ForegroundColor Green
} else {
    Write-Host "Warning: .env file not found" -ForegroundColor Red
}
