param(
    [ValidateSet('build','run','clean','rebuild')][string]$Action = 'build',
    [ValidateSet('Debug','Release')][string]$Configuration = 'Release',
    [string]$RunArgs = ''
)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$buildRoot = Join-Path $projectRoot 'build'
$buildDir = Join-Path $buildRoot $Configuration.ToLowerInvariant()
$appName = (Split-Path $projectRoot -Leaf) -replace '^\d{2}_',''
$executable = Join-Path $projectRoot "$appName.exe"

# Remove only this script's generated configurations and the known app executable.
if ($Action -in @('clean','rebuild')) {
    foreach ($configName in @('debug','release')) {
        $target = [IO.Path]::GetFullPath((Join-Path $buildRoot $configName))
        if (-not $target.StartsWith(([IO.Path]::GetFullPath($buildRoot) + '\'), [StringComparison]::OrdinalIgnoreCase)) {
            throw 'Clean target escaped build directory'
        }
        if (Test-Path -LiteralPath $target) { Remove-Item -LiteralPath $target -Recurse -Force }
    }
    if (Test-Path -LiteralPath $executable) { Remove-Item -LiteralPath $executable -Force }
    if ($Action -eq 'clean') { exit 0 }
}

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw 'MSVC C++ build tools are required' }
$vcvars = Join-Path $vs 'VC\Auxiliary\Build\vcvars64.bat'
$environmentLines = & cmd.exe /d /c "call `"$vcvars`" >nul && set"
if ($LASTEXITCODE -ne 0) { throw 'Failed to initialize MSVC environment' }
foreach ($line in $environmentLines) {
    if ($line -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') }
}
$ninja = Join-Path $vs 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
if (-not (Test-Path $ninja)) { throw 'Visual Studio Ninja component is required' }
$qtRoot = Join-Path $projectRoot 'externals\Qt\6.8.3\msvc2022_64'
$env:PATH = "$qtRoot\bin;$env:PATH"
$env:QT_PLUGIN_PATH = "$qtRoot\plugins"
New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
$ErrorActionPreference = 'Continue'
& cmake -S $projectRoot -B $buildDir -G Ninja "-DCMAKE_MAKE_PROGRAM=$ninja" '-DCMAKE_CXX_COMPILER=cl' "-DCMAKE_BUILD_TYPE=$Configuration" '-DCMAKE_CUDA_ARCHITECTURES=native' 2>&1 | Tee-Object (Join-Path $buildDir 'configure.log')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& cmake --build $buildDir --parallel 4 2>&1 | Tee-Object (Join-Path $buildDir 'build.log')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if ($Action -eq 'run') {
    if ($RunArgs) { & $executable ($RunArgs -split ' ') } else { & $executable }
    exit $LASTEXITCODE
}
