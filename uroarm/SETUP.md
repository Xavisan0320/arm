# uroarm セットアップ手順書

6サーボ（5軸＋グリッパー）の仕分けアームを、Windows PC で動かすための環境構築手順です。
（同じリポジトリの `uroarm6` は、この版に J4（前腕ロール）を足した7サーボ版です。Python の環境はまったく同じです。）

## 動作確認済みの環境

| 項目 | バージョン |
|---|---|
| OS | Windows 11 |
| Python | **3.11.9**（3.11 系が必須） |
| OpenCV | opencv-contrib-python **4.5.5.64**（新しい版は不可、後述） |
| numpy | **1.26.4**（2.x は不可） |
| PyTorch | 2.13.0（CPU 版） |
| ultralytics（YOLO） | 8.4.102 |
| PySimpleGUI | 6.2（LGPL 版。ライセンスキー不要） |
| scikit-learn | 1.9.0 |

すべてのパッケージの正確な一覧は `requirements-lock.txt`（参考用）にあります。

## 必要なもの

- アーム本体：サーボ6個（J1〜J5 が腕、**J6 がグリッパー**）、PCA9685 サーボドライバ、Raspberry Pi Pico（MicroPython）
- USB カメラ
- 配線：Pico の **GP0 = SDA、GP1 = SCL**（I2C0）を PCA9685 へ。PCA9685 のアドレスは **0x40**。サーボは J1〜J6 を**チャンネル 0〜5** の順につなぐ
- ArUco マーカー：`data/marker/marker.pdf` を印刷。**0〜2番を机の上（基準平面）**、**3番以降をグリッパー**に貼る

---

## 1. Python 3.11 をインストールする

1. python.org から **Python 3.11.9（Windows installer 64-bit）** をダウンロードしてインストールします。
   - インストール時に「Add python.exe to PATH」にチェックを入れます。
2. PowerShell で確認します。
   ```powershell
   py -0p
   ```
   一覧に `-V:3.11` があれば準備完了です。

> **注意**：他のバージョン（3.12 など）も入っていると、`py` だけで実行したときにそちらが使われます。この手順では必ず `py -3.11` で 3.11 を指定します。

## 2. ファイルを取得する（**短いパスの場所に**）

```powershell
cd C:\
git clone https://github.com/Xavisan0320/arm.git
cd C:\arm\uroarm
```

> **重要**：`C:\arm` のような**短いパスの場所**に置いてください。PyTorch は venv の中に190文字近い深さのファイルを作るので、Windows の「パスは260文字まで」という制限に引っかかります。「ドキュメント」の奥などに置くと、インストールが途中で `[WinError 206] ファイル名または拡張子が長すぎます` で失敗します。`setup_venv.ps1` は、venv のパスが50文字を超えていると最初に止まって知らせます（Windows の長いパスが有効なら止まりません）。詳しくは「5. 注意点」を参照。

## 3. 仮想環境（venv）を作る（自動）

`uroarm` フォルダで次を実行します。数分かかります（PyTorch などを約1GBダウンロードします）。

```powershell
powershell -ExecutionPolicy Bypass -File setup_venv.ps1
```

最後に次のような行が出れば成功です。

```
OK  python 3.11.9 | cv2 4.5.5 | numpy 1.26.4 | torch 2.13.0+cpu | ultralytics 8.4.102 | PySimpleGUI 6.2 | scikit-learn 1.9.0
```

オプション：
- `-Python <python.exe のパス>`：`py -3.11` の代わりに、使う Python 3.11 を直接指定
- `-VenvDir <フォルダ>`：venv を作る場所（既定は `uroarm\venv`）

### 手動で行う場合（スクリプトと同じ内容）

```powershell
py -3.11 -m venv venv
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install torch==2.13.0 torchvision==0.28.0 numpy==1.26.4 --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe -m pip uninstall -y opencv-python opencv-python-headless
venv\Scripts\python.exe -m pip install --force-reinstall --no-deps opencv-contrib-python==4.5.5.64
```

**最後の2行（OpenCV の入れ替え）は必ず実行してください**。理由は「5. 注意点」を参照。

## 4. 起動と初期設定

### 4-1. Pico にファームウェアを書き込む

1. Pico に MicroPython を書き込みます（Thonny などを使用）。
2. `pico/main.py` を、Pico 上に **`main.py`** という名前で保存します。追加のライブラリは不要です。

### 4-2. 設定ファイル `data/arm.json`

リポジトリの `data/arm.json` は、**開発に使った機体の較正データ**です。別の機体では値が合いません。最低限、次を自分の環境に合わせます。

| キー | 内容 |
|---|---|
| `COM` | Pico がつながっている COM ポート（デバイスマネージャーで確認。例：`COM4`） |
| `camera-index` | カメラの番号（このデータでは `1`。カメラが1台だけなら通常 `0`） |

※ `data/arm.json` は、`arm.py` を閉じるときに上書きされます。編集するときはアプリを閉じてください。

### 4-3. 起動

```powershell
venv\Scripts\python.exe arm.py
```

> `py arm.py` や、別フォルダの venv の `python` で起動しないでください。必要なライブラリが入っておらず、エラーになります。

### 4-4. 新しい機体での較正と使い方

1. **`angle.py`**：サーボの角度とアームの関節角の対応を、関節ごとに較正します。
2. **`arm.py`** を起動して、次の順に行います。
   1. 「**Reset**」：机の上の基準平面（マーカー0〜2）を測り直します。
   2. 「**Adjust XY**」：カメラとアームの位置関係を較正します（ハンドアイ較正）。
   3. 「**Test XY**」：画面上の点にアームを動かして、較正を確認します。
3. 物体をつかむ：
   - 「**AI**」または「**AI (red_cube)**」などを押すと、YOLO が起動して、その種類の物体を検出するようになります（「AI」はすべての種類）。
   - 「**Grab**」で、検出した物体をつかみに行きます。
   - 「**Ready**」で待機姿勢に戻ります。

---

## 5. 注意点・トラブルシューティング

### OpenCV は 4.5.5 の contrib 版だけにする
`marker.py` は、ArUco の古い関数 `aruco.estimatePoseSingleMarkers` を使っています。この関数は OpenCV 4.7 以降で削除されました。
一方、`ultralytics` をインストールすると、新しい `opencv-python` も一緒に入ります。これと `opencv-contrib-python` は同じ `cv2` を共有しているので、両方あると壊れます。そのため、`opencv-python` を削除してから contrib 4.5.5 を入れ直します。
- この構成では、`pip check` に「ultralytics requires opencv-python, which is not installed」と出ますが、**意図どおりなので問題ありません**。
- 症状：`AttributeError: module 'cv2.aruco' has no attribute 'estimatePoseSingleMarkers'` が出たら、3章の最後の2行をもう一度実行してください。

### numpy は 1.x に固定
OpenCV 4.5.5 は numpy 2 に対応していません。
- 症状：`numpy.core.multiarray failed to import` や `_ARRAY_API not found`
- 対処：`venv\Scripts\python.exe -m pip install numpy==1.26.4`

### パスが長すぎてインストールが失敗する（WinError 206）
`ERROR: Could not install packages due to an OSError: [WinError 206] ファイル名または拡張子が長すぎます` と出た場合の対処です。次のどちらかを行ってください。
- **フォルダを短い場所へ移す（おすすめ）**：`C:\arm\uroarm` など。途中まで作られた `venv` フォルダは削除してから作り直します。venv だけを別の短い場所に作ることもできます（`-VenvDir C:\venvs\uroarm`）。
- **Windows の長いパスを有効にする（管理者権限が必要）**：管理者の PowerShell で次を実行し、PC を再起動します。
  ```powershell
  New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force
  ```

### venv はコピー・移動しない
venv の中には、作った場所のパスが記録されています。別のフォルダや PC へコピーしても正しく動きません。必ずその場所で作り直してください。

### スクリプトが実行できない（実行ポリシー）
「このシステムではスクリプトの実行が無効になっている」と出たら、手順どおり `powershell -ExecutionPolicy Bypass -File setup_venv.ps1` で実行してください（そのときだけ許可されます）。

### 「DLL load failed … アプリケーション制御ポリシーによってこのファイルがブロックされました」
開発中、**別フォルダの古い venv** でこのエラーが出たことがあります（`sklearn` の読み込み時）。まず、起動に使っている Python が `uroarm\venv\Scripts\python.exe` かを確認してください。それでも出る場合は、Windows のアプリ制御（スマート アプリ コントロール等）が原因です。PC の管理者に確認してください。

---

## ファイル構成

| ファイル | 内容 |
|---|---|
| `arm.py` | メインアプリ（GUI、検出、つかむ、較正） |
| `kinematics.py` | 順運動学・逆運動学（5軸） |
| `servo.py` / `util.py` | サーボ通信・角度変換、共通処理 |
| `camera.py` / `marker.py` / `infer.py` | カメラ、ArUco マーカー、YOLO 検出 |
| `angle.py` / `yoko.py` | 関節角の較正、キーボード操作 |
| `board.py` / `print.py` | カメラ較正用ボード・マーカーの生成 |
| `pico/main.py` | Pico のファームウェア（PCA9685 制御） |
| `contest.pt` | YOLO モデル |
| `data/arm.json` | 設定と較正データ（この機体用） |
| `setup_venv.ps1` / `requirements.txt` | 環境構築（本手順書） |
| `requirements-lock.txt` | 動作確認済み環境の全パッケージ一覧（参考用） |
