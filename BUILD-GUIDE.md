# spout_ntsc ビルド手順（Windows Visual Studio版）

spout_ntscをWindows上でVisual Studioを使ってビルドする完全な手順です。
CUDA GPU加速とQt6 GUIプレビューに対応しています。

---

## 前提条件のインストール

### 1. Visual Studio 2022

**ダウンロード:**
https://visualstudio.microsoft.com/ja/downloads/

**必要なコンポーネント:**
- C++によるデスクトップ開発
- Windows 10 SDK

**インストール後の確認:**

Developer Command Prompt for VS 2022 を開いて:
```cmd
cl
```

以下のような出力が表示されればOK:
```
Microsoft (R) C/C++ Optimizing Compiler Version 19.xx.xxxxx.x for x64
```

---

### 2. CMake

**ダウンロード:**
https://cmake.org/download/

**インストール後の確認:**
```cmd
cmake --version
```

出力例:
```
cmake version 3.30.0
```

---

### 3. FFmpeg開発ライブラリ

#### ダウンロード:

ブラウザで https://www.gyan.dev/ffmpeg/builds/ を開く

**ダウンロードするファイル:**
- `ffmpeg-release-full-shared.7z` (約150MB)

#### 解凍:

7-Zipなどで `C:\ffmpeg` に解凍してください。

**解凍後のディレクトリ構造（重要）:**
```
C:\ffmpeg\
  ├── bin\       (ffmpeg.exe, *.dll)
  ├── include\   (libavcodec/, libavformat/, libavutil/, libswscale/, libswresample/)
  └── lib\       (*.lib インポートライブラリ)
```

#### 確認:

エクスプローラーで以下が存在することを確認:
```
C:\ffmpeg\include\libavcodec\avcodec.h
C:\ffmpeg\include\libavformat\avformat.h
C:\ffmpeg\include\libavutil\avutil.h
C:\ffmpeg\include\libswscale\swscale.h
C:\ffmpeg\include\libswresample\swresample.h
C:\ffmpeg\lib\avcodec.lib
C:\ffmpeg\lib\avformat.lib
C:\ffmpeg\lib\avutil.lib
C:\ffmpeg\lib\swscale.lib
C:\ffmpeg\lib\swresample.lib
```

---

### 4. NVIDIA CUDA Toolkit（GPU加速用・オプション）

**ダウンロード:**
https://developer.nvidia.com/cuda-downloads

**推奨バージョン:** CUDA 12.5 以上

**インストール:**
- デフォルト設定でインストール
- Visual Studio Integration を有効にする

**インストール後の確認:**
```cmd
nvcc --version
```

出力例:
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Wed_Apr_17_19:19:55_Pacific_Daylight_Time_2024
Cuda compilation tools, release 12.5, V12.5.40
```

**自分のGPUのCompute Capabilityを確認:**
```cmd
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

出力例:
```
name, compute_cap
NVIDIA GeForce RTX 4070 Super, 8.9
```

#### GPU設定の調整（重要）:

`CMakeLists.txt` の22行目を自分のGPUに合わせて編集:

エクスプローラーで `E:\Repos\composite-video-simulator\CMakeLists.txt` を開き、
22行目を確認:

```cmake
set(CMAKE_CUDA_ARCHITECTURES 89)
```

**主要なGPUの設定値:**

| GPU | Compute Capability | 設定値 |
|-----|-------------------|-------|
| RTX 4090 / 4080 / 4070 / 4060 (Ada Lovelace) | 8.9 | 89 |
| RTX 3090 / 3080 / 3070 / 3060 (Ampere) | 8.6 | 86 |
| RTX 2080 / 2070 / 2060 (Turing) | 7.5 | 75 |
| GTX 1080 / 1070 / 1060 (Pascal) | 6.1 | 61 |

**複数GPU対応（推奨）:**
```cmake
set(CMAKE_CUDA_ARCHITECTURES 75;86;89)
```

**CUDA無しでビルドする場合:**

この手順をスキップし、後のCMake configureで `-DENABLE_CUDA=OFF` を指定してください。

---

### 5. Qt6（GUIプレビュー用・オプション）

**ダウンロード:**
https://www.qt.io/download-qt-installer

**インストール:**
- Qt Online Installer を実行
- Qt 6.7.x 以上を選択
- コンポーネントで `MSVC 2019 64-bit` または `MSVC 2022 64-bit` を選択
- インストール先を確認（例: `C:\Qt\6.7.0\msvc2019_64`）

**インストール後の確認:**

エクスプローラーで以下が存在することを確認:
```
C:\Qt\6.7.0\msvc2019_64\bin\Qt6Core.dll
C:\Qt\6.7.0\msvc2019_64\bin\Qt6Gui.dll
C:\Qt\6.7.0\msvc2019_64\bin\Qt6Widgets.dll
C:\Qt\6.7.0\lib\cmake\Qt6\Qt6Config.cmake
```

**Qt無しでビルドする場合:**

この手順をスキップし、後のCMake configureで `-DENABLE_QT_GUI=OFF` を指定してください（これがデフォルト）。

---

## ビルド手順

### ステップ1: x64 Native Tools Command Promptを開く

```
"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
```

**重要:** 必ず **x64 版** を使用してください。x86版では動作しません。

`for x86` と表示された場合は、x64版のプロンプトを開き直してください。

### ステップ2: リポジトリに移動

```cmd
cd E:\Repos\composite-video-simulator
```

### ステップ3: ビルドディレクトリを作成

```cmd
mkdir build
cd build
```

### ステップ4: CMake Configure

以下のいずれかのコマンドを実行してください。

#### 構成A: CUDA有効 + Qt無し（最小構成）

```cmd
cmake -G "Visual Studio 17 2022" -A x64 -DFFMPEG_ROOT=C:/ffmpeg ..
```

#### 構成B: CUDA有効 + Qt有効（フル構成）

```cmd
cmake -G "Visual Studio 17 2022" -A x64 -DFFMPEG_ROOT=C:/ffmpeg -DENABLE_QT_GUI=ON -DQt6_DIR=C:/Qt/6.7.0/msvc2019_64 ..
```

Qt6のパスが異なる場合は適宜変更してください（例: `C:/Qt/6.8.0/msvc2022_64`）

#### 構成C: CUDA無効 + Qt無し（CPU版）

```cmd
cmake -G "Visual Studio 17 2022" -A x64 -DFFMPEG_ROOT=C:/ffmpeg -DENABLE_CUDA=OFF ..
```

#### 構成D: CUDA無効 + Qt有効

```cmd
cmake -G "Visual Studio 17 2022" -A x64 -DFFMPEG_ROOT=C:/ffmpeg -DENABLE_CUDA=OFF -DENABLE_QT_GUI=ON -DQt6_DIR=C:/Qt/6.7.0/msvc2019_64 ..
```

**成功すると以下のような出力が表示されます:**

```
-- Selecting Windows SDK version 10.0.xxxxx.0 to target Windows 10.0.xxxxx.
-- The CXX compiler identification is MSVC 19.xx.xxxxx.x
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.xx.xxxxx/bin/Hostx64/x64/cl.exe - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- CUDA enabled: 12.5.40
-- Using FFMPEG_ROOT = C:/ffmpeg
  Found avcodec: C:/ffmpeg/lib/avcodec.lib
  Found avformat: C:/ffmpeg/lib/avformat.lib
  Found avutil: C:/ffmpeg/lib/avutil.lib
  Found swscale: C:/ffmpeg/lib/swscale.lib
  Found swresample: C:/ffmpeg/lib/swresample.lib
-- Configuring done
-- Generating done
-- Build files have been written to: E:/Repos/composite-video-simulator/build
```

Qt有効の場合は追加で:
```
-- spout_ntsc: Qt6 GUI enabled
```

### ステップ5: ビルド実行

```cmd
cmake --build . --config Release --target spout_ntsc --parallel 8
```

パラメータ説明:
- `--config Release`: Release構成でビルド（最適化有効）
- `--target spout_ntsc`: spout_ntsc のみビルド（他のツールはスキップ）
- `--parallel 8`: 8コア並列ビルド（CPUコア数に合わせて調整可能）

**ビルド成功例:**

```
Microsoft (R) Build Engine version 17.x.x.xxxxx for .NET Framework
Copyright (C) Microsoft Corporation. All rights reserved.

  Checking Build System
  Building NVCC (Device) object CMakeFiles/ntsc_cuda.dir/ntsc_cuda_generated_ntsc_cuda.cu.obj
  Building Custom Rule E:/Repos/composite-video-simulator/CMakeLists.txt
  ntsc_cuda.cu
  ntsc_cuda.vcxproj -> E:\Repos\composite-video-simulator\build\Release\ntsc_cuda.lib
  Building Custom Rule E:/Repos/composite-video-simulator/CMakeLists.txt
  spout_ntsc.cpp
  Automatic MOC for target spout_ntsc
  spout_ntsc.vcxproj -> E:\Repos\composite-video-simulator\build\Release\spout_ntsc.exe
  Building Custom Rule E:/Repos/composite-video-simulator/CMakeLists.txt

Build succeeded.
    0 Warning(s)
    0 Error(s)

Time Elapsed 00:01:23.45
```

### ステップ6: 実行ファイルの確認

```cmd
dir Release\spout_ntsc.exe
```

出力例:
```
2026/03/17  18:30           234,567 spout_ntsc.exe
```

---

## 実行に必要なDLLのコピー

ビルドは成功しましたが、実行するには追加のDLLファイルが必要です。

### FFmpeg DLLのコピー

```cmd
copy C:\ffmpeg\bin\avcodec-*.dll Release\
copy C:\ffmpeg\bin\avformat-*.dll Release\
copy C:\ffmpeg\bin\avutil-*.dll Release\
copy C:\ffmpeg\bin\swscale-*.dll Release\
copy C:\ffmpeg\bin\swresample-*.dll Release\
copy C:\ffmpeg\bin\avdevice-*.dll Release\
copy C:\ffmpeg\bin\avfilter-*.dll Release\
copy C:\ffmpeg\bin\postproc-*.dll Release\
```

または一括コピー:
```cmd
copy C:\ffmpeg\bin\*.dll Release\
```

### Qt DLLのコピー（Qt有効でビルドした場合のみ）

```cmd
copy C:\Qt\6.7.0\msvc2019_64\bin\Qt6Core.dll Release\
copy C:\Qt\6.7.0\msvc2019_64\bin\Qt6Gui.dll Release\
copy C:\Qt\6.7.0\msvc2019_64\bin\Qt6Widgets.dll Release\
```

Qt6のプラットフォームプラグインもコピー:
```cmd
mkdir Release\platforms
copy C:\Qt\6.7.0\msvc2019_64\plugins\platforms\qwindows.dll Release\platforms\
```

---

## 動作確認

### ヘルプ表示

```cmd
Release\spout_ntsc.exe -h
```

出力例:
```
spout_ntsc [options]

 Spout2 options:
   -receiver <name>           Spout2 source to receive from (default: active sender)
   -sender <name>             Spout2 sender name (default: NTSC-Output)
   -fps <n>                   Target frame rate (default: 30)
   -gui                       Show Qt preview window

 Video standard:
   -tvstd <pal|ntsc>          TV standard (default: ntsc)
   -width <n>                 Processing width (default: 720)

 NTSC/Composite options:
   -vhs                       Enable VHS emulation
   ...
```

### 起動テスト

```cmd
Release\spout_ntsc.exe -sender TestOutput
```

出力例（CUDA有効の場合）:
```
spout_ntsc — Real-time NTSC composite video simulation proxy
Standard: NTSC | Resolution: 720x480 | Composite: ON | VHS: OFF
Spout receiver: connecting to active sender
Spout sender: broadcasting as 'TestOutput'
Processing resolution: 720x480 (NTSC)
Target FPS: 30
NTSC CUDA acceleration enabled
Processing loop started. Press Ctrl+C to stop.
Waiting for Spout2 sender...
```

Spout2送信元（OBS、TouchDesigner、Resolume等）を起動すると映像処理が開始されます。

Ctrl+C で終了。

---

## トラブルシューティング

### エラー: "Could not find FFmpeg library: avcodec"

**原因:** FFMPEG_ROOTのパスが間違っている、またはinclude/lib/が存在しない

**解決:**

1. パスを確認:
```cmd
dir C:\ffmpeg\include\libavcodec
dir C:\ffmpeg\lib\avcodec.lib
```

2. 両方存在することを確認

3. 再度CMake configure:
```cmd
cd E:\Repos\composite-video-simulator\build
del CMakeCache.txt
cmake -G "Visual Studio 17 2022" -A x64 -DFFMPEG_ROOT=C:/ffmpeg ..
```

### エラー: "CUDA compiler not found"

**原因:** nvccがPATHに無い、またはCUDA Toolkitが未インストール

**解決:**

1. CUDA Toolkitのインストール確認:
```cmd
nvcc --version
```

2. インストールされていない場合:
   - https://developer.nvidia.com/cuda-downloads からダウンロード
   - インストール後、Developer Command Promptを再起動

3. インストール済みでも認識されない場合:
```cmd
set PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin
nvcc --version
```

4. または CUDA無効でビルド:
```cmd
cmake -G "Visual Studio 17 2022" -A x64 -DFFMPEG_ROOT=C:/ffmpeg -DENABLE_CUDA=OFF ..
```

### エラー: "CMAKE_CUDA_ARCHITECTURES 89: No NVIDIA GPU architecture"

**原因:** 古いCUDA Toolkitがsm_89をサポートしていない

**解決:**

1. CMakeLists.txtの22行目を編集:
```cmake
set(CMAKE_CUDA_ARCHITECTURES 86)
```

または自分のGPUに合った値に変更（前述のGPU対応表参照）

2. 再度CMake configure & build:
```cmd
cd E:\Repos\composite-video-simulator\build
del CMakeCache.txt
cmake -G "Visual Studio 17 2022" -A x64 -DFFMPEG_ROOT=C:/ffmpeg ..
cmake --build . --config Release --target spout_ntsc --parallel 8
```

### エラー: "Could not find Qt6"

**原因:** Qt6のパスが間違っている、またはQt6が未インストール

**解決:**

1. Qt6インストール確認:
```cmd
dir C:\Qt\6.7.0\msvc2019_64\lib\cmake\Qt6\Qt6Config.cmake
```

2. パスが異なる場合、正しいパスを指定:
```cmd
cmake -G "Visual Studio 17 2022" -A x64 -DFFMPEG_ROOT=C:/ffmpeg -DENABLE_QT_GUI=ON -DQt6_DIR=C:/Qt/6.8.0/msvc2022_64 ..
```

3. または Qt無効でビルド:
```cmd
cmake -G "Visual Studio 17 2022" -A x64 -DFFMPEG_ROOT=C:/ffmpeg -DENABLE_QT_GUI=OFF ..
```

### 実行時エラー: "プロシージャエントリポイントが見つかりません"

**原因:** FFmpeg DLLが見つからない、またはバージョン不一致

**解決:**

1. DLLがコピーされているか確認:
```cmd
dir Release\*.dll
```

2. 足りない場合は再度コピー:
```cmd
copy C:\ffmpeg\bin\*.dll Release\
```

3. 環境変数PATHにC:\ffmpeg\binを追加（代替案）:
```cmd
set PATH=%PATH%;C:\ffmpeg\bin
Release\spout_ntsc.exe -h
```

### 実行時エラー: "Failed to open DirectX 11 for receiver"

**原因:** DirectXランタイムの問題、またはGPU非対応

**解決:**

1. DirectX End-User Runtimesをインストール:
   https://www.microsoft.com/ja-jp/download/details.aspx?id=35

2. グラフィックドライバーを最新に更新

---

## まとめ: 全コマンド一覧（コピペ用）

### 構成A: CUDA有効 + Qt無し（標準構成）

```cmd
cd E:\Repos\composite-video-simulator
mkdir build
cd build
cmake -G "Visual Studio 17 2022" -A x64 -DFFMPEG_ROOT=C:/ffmpeg ..
cmake --build . --config Release --target spout_ntsc --parallel 8
copy C:\ffmpeg\bin\*.dll Release\
Release\spout_ntsc.exe -h
```

### 構成B: CUDA有効 + Qt有効（フル構成）

```cmd
cd E:\Repos\composite-video-simulator
mkdir build
cd build
cmake -G "Visual Studio 17 2022" -A x64 -DFFMPEG_ROOT=C:/ffmpeg -DENABLE_QT_GUI=ON -DQt6_DIR=C:/Qt/6.7.0/msvc2019_64 ..
cmake --build . --config Release --target spout_ntsc --parallel 8
copy C:\ffmpeg\bin\*.dll Release\
copy C:\Qt\6.7.0\msvc2019_64\bin\Qt6Core.dll Release\
copy C:\Qt\6.7.0\msvc2019_64\bin\Qt6Gui.dll Release\
copy C:\Qt\6.7.0\msvc2019_64\bin\Qt6Widgets.dll Release\
mkdir Release\platforms
copy C:\Qt\6.7.0\msvc2019_64\plugins\platforms\qwindows.dll Release\platforms\
Release\spout_ntsc.exe -gui -h
```

### 構成C: CUDA無効 + Qt無し（CPU版）

```cmd
cd E:\Repos\composite-video-simulator
mkdir build
cd build
cmake -G "Visual Studio 17 2022" -A x64 -DFFMPEG_ROOT=C:/ffmpeg -DENABLE_CUDA=OFF ..
cmake --build . --config Release --target spout_ntsc --parallel 8
copy C:\ffmpeg\bin\*.dll Release\
Release\spout_ntsc.exe -h
```

---

## 配布用パッケージの作成

他のPCで実行可能なパッケージを作成する場合:

```cmd
cd E:\Repos\composite-video-simulator\build
mkdir spout_ntsc_package
copy Release\spout_ntsc.exe spout_ntsc_package\
copy C:\ffmpeg\bin\*.dll spout_ntsc_package\
```

Qt有効の場合は追加で:
```cmd
copy C:\Qt\6.7.0\msvc2019_64\bin\Qt6Core.dll spout_ntsc_package\
copy C:\Qt\6.7.0\msvc2019_64\bin\Qt6Gui.dll spout_ntsc_package\
copy C:\Qt\6.7.0\msvc2019_64\bin\Qt6Widgets.dll spout_ntsc_package\
mkdir spout_ntsc_package\platforms
copy C:\Qt\6.7.0\msvc2019_64\plugins\platforms\qwindows.dll spout_ntsc_package\platforms\
```

`spout_ntsc_package` フォルダをZIP圧縮して配布。
