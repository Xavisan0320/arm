# Builds the Python environment for uroarm6 (Windows, PowerShell).
#
#   powershell -ExecutionPolicy Bypass -File setup_venv.ps1
#
# Options:
#   -Python  <path>   Python 3.11 to build the venv from (default: "py -3.11")
#   -VenvDir <dir>    where to create it (default: venv, next to this script)
#
# See SETUP.md for what each step is for.

param(
    [string]$Python = "",
    [string]$VenvDir = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $VenvDir) { $VenvDir = Join-Path $root "venv" }

function Run($exe, [string[]]$argList) {
    Write-Host ">> $exe $($argList -join ' ')" -ForegroundColor Cyan
    & $exe @argList
    if ($LASTEXITCODE -ne 0) { throw "failed ($LASTEXITCODE): $exe $($argList -join ' ')" }
}

# ---- 1. Python 3.11. Plain "py" may pick a newer Python (3.12+) that the
#         pinned packages were never tested on, so ask for 3.11 explicitly.
if ($Python) {
    $pyExe = $Python; $pyArgs = @()
} else {
    $pyExe = "py"; $pyArgs = @("-3.11")
}
$ver = & $pyExe @($pyArgs + @("-c", "import sys; print('%d.%d' % sys.version_info[:2])"))
if ($ver -ne "3.11") { throw "Python 3.11 is required (found $ver). Install it from python.org, or pass -Python <path to python.exe 3.11>." }

# ---- 2. the venv
if (Test-Path $VenvDir) { throw "$VenvDir already exists - remove it first, or pass -VenvDir <new dir>." }

# PyTorch installs files ~190 characters deep inside the venv, and Windows
# refuses paths over 260 characters unless long paths are enabled - pip then
# fails half-way with "[WinError 206] (file name too long)". Check before
# downloading anything.
$fullVenv = [System.IO.Path]::GetFullPath($VenvDir)
$longPaths = (Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
              -Name LongPathsEnabled -ErrorAction SilentlyContinue).LongPathsEnabled
if ($longPaths -ne 1 -and $fullVenv.Length -gt 50) {
    throw ("The venv path is too long for Windows ($($fullVenv.Length) chars: $fullVenv).`n" +
           "Put the project in a short folder (e.g. C:\arm\uroarm6) or pass -VenvDir C:\venvs\uroarm6,`n" +
           "or enable Windows long paths (see SETUP.md).")
}
Run $pyExe ($pyArgs + @("-m", "venv", $VenvDir))
$py = Join-Path $VenvDir "Scripts\python.exe"
Run $py @("-m", "pip", "install", "--upgrade", "pip")

# ---- 3. torch first, CPU build, from the PyTorch index (the default index can
#         hand out a much larger CUDA build that this app does not use).
#         numpy is pinned here too, or torchvision pulls the newest numpy 2.x
#         only for step 4 to downgrade it again.
Run $py @("-m", "pip", "install", "torch==2.13.0", "torchvision==0.28.0", "numpy==1.26.4",
          "--index-url", "https://download.pytorch.org/whl/cpu",
          "--extra-index-url", "https://pypi.org/simple")

# ---- 4. everything else, pinned
Run $py @("-m", "pip", "install", "-r", (Join-Path $root "requirements.txt"))

# ---- 5. the OpenCV swap. ultralytics installs opencv-python, which shares the
#         cv2 package with opencv-contrib-python and hides the old ArUco API
#         marker.py needs. Remove it, then reinstall contrib 4.5.5 over the top
#         (uninstalling opencv-python deletes cv2 files contrib also owns).
Run $py @("-m", "pip", "uninstall", "-y", "opencv-python", "opencv-python-headless")
Run $py @("-m", "pip", "install", "--force-reinstall", "--no-deps", "opencv-contrib-python==4.5.5.64")

# ---- 6. check (written to a file: multi-line "python -c" arguments do not
#         survive Windows PowerShell 5.1's argument quoting reliably)
$checkFile = Join-Path $env:TEMP "uroarm6_check_env.py"
@"
import sys, cv2, numpy, sklearn, serial, PySimpleGUI, torch, ultralytics
from cv2 import aruco
assert cv2.__version__.startswith('4.5.5'), cv2.__version__
assert numpy.__version__.startswith('1.'), numpy.__version__
assert hasattr(aruco, 'estimatePoseSingleMarkers'), 'old ArUco API missing - the OpenCV swap failed'
print('OK  python', sys.version.split()[0], '| cv2', cv2.__version__, '| numpy', numpy.__version__,
      '| torch', torch.__version__, '| ultralytics', ultralytics.__version__,
      '| PySimpleGUI', PySimpleGUI.__version__, '| scikit-learn', sklearn.__version__)
"@ | Set-Content -Path $checkFile -Encoding ascii
Run $py @($checkFile)
Write-Host ""
Write-Host "Done. Start the app with:  $py arm.py" -ForegroundColor Green
Write-Host "(pip check will still say 'ultralytics requires opencv-python' - that is expected, see SETUP.md)"
