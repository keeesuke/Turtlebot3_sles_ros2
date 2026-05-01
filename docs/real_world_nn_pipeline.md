# Real-World NN Planner: Data Collection & Training Pipeline

## 目次

1. [概要と目的](#1-概要と目的)
2. [NNモデルの入出力仕様](#2-nnモデルの入出力仕様)
3. [Simulation vs Real-World の違い](#3-simulation-vs-real-world-の違い)
4. [データ形式の選択：JSON vs ROS Bag](#4-データ形式の選択json-vs-ros-bag)
5. [Start / Goal 戦略](#5-start--goal-戦略)
6. [データ収集スクリプトの設計](#6-データ収集スクリプトの設計)
7. [前処理パイプライン（JSON → NPZ）](#7-前処理パイプラインjson--npz)
8. [学習パイプライン（NPZ → Model）](#8-学習パイプラインnpz--model)
9. [モデルのデプロイ方法](#9-モデルのデプロイ方法)
10. [完全パイプライン フローチャート](#10-完全パイプライン-フローチャート)
11. [ファイル・ディレクトリ構成](#11-ファイルディレクトリ構成)

---

## 1. 概要と目的

このドキュメントは、**TurtleBot3 Waffle Pi 実機環境における NN Planner の実世界データ収集 → 前処理 → 学習 → デプロイまでの完全なパイプライン**を設計・解説するものです。

NN Planner（`planner_nn_real_world.py`）は **模倣学習（Imitation Learning）** に基づいており、エキスパートコントローラ（MPPI / HAA プランナー）が生成した制御コマンドをデータとして収集し、そのデモンストレーションをニューラルネットワークに学習させます。

学習済みモデルは推論時に 50 Hz で稼働し、リアルタイムで `/cmd_vel` を発行します。

---

## 2. NNモデルの入出力仕様

### 2.1 入力ベクトル（364次元）

```
input = [v, ω, goal_x_robot, goal_y_robot, lidar_0, lidar_1, ..., lidar_359]
         ↑  ↑   ←── ロボット座標系でのゴール ───→   ←── 360 rays ──→
         2次元  +         2次元              +         360次元
                           = 合計 364次元
```

| インデックス | 要素           | 単位    | 備考                                        |
|------------|--------------|---------|---------------------------------------------|
| 0          | v            | m/s     | ロボット前進速度（EMAフィルタ済み）             |
| 1          | ω (omega)    | rad/s   | 回転角速度（EMAフィルタ済み）                  |
| 2          | goal_x_robot | m       | ゴールのロボット座標系 X（前方方向）             |
| 3          | goal_y_robot | m       | ゴールのロボット座標系 Y（左方向）               |
| 4–363      | lidar[0–359] | m       | 0〜360°、各光線の障害物距離（0〜lidar_max_range） |

**重要な前処理（LiDAR）:**
- `inf`, `nan`, `<= 0` の値 → `lidar_max_range`（1.0m）に置換
- 360レイ以外のセンサ → `np.interp` でリサンプリング
- `[0.0, lidar_max_range]` の範囲にクリップ

### 2.2 出力ベクトル（2次元）

```
output = [v_cmd, ω_cmd]
```

| インデックス | 要素   | 単位   | クリップ範囲                                        |
|------------|--------|--------|----------------------------------------------------|
| 0          | v_cmd  | m/s    | `[0.0, v_limit_haa]`（デフォルト 0.2 m/s）           |
| 1          | ω_cmd  | rad/s  | `[-omega_limit_haa, omega_limit_haa]`（デフォルト ±0.9） |

### 2.3 ネットワーク構造（MLP）

```
Input(364)
  └─ Linear(364 → 256) + BatchNorm1d + ReLU + Dropout(0.1)
  └─ Linear(256 → 128) + BatchNorm1d + ReLU + Dropout(0.1)
  └─ Linear(128 →  64) + BatchNorm1d + ReLU + Dropout(0.1)
  └─ Linear( 64 →   2)  # 出力層（活性化なし）
Output(2)
```

- 損失関数: MSELoss
- オプティマイザ: Adam（lr=1e-3）
- スケジューラ: ReduceLROnPlateau（patience=5, factor=0.5）
- バッチサイズ: 256
- エポック数: 20（最良モデルを保存）

### 2.4 チェックポイント形式

```python
{
    'epoch': int,
    'model_state_dict': OrderedDict(...),  # model.state_dict()
    'optimizer_state_dict': ...,
    'val_loss': float,
    'val_metrics': dict,
    'config': dict
}
```

ロード方法（`weights_only=False` が必要 — PyTorch 2.x 以降）:
```python
checkpoint = torch.load('best_model.pth', map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
```

---

## 3. Simulation vs Real-World の違い

実世界でのデータ収集・推論はシミュレーションと複数の点で異なります。

### 3.1 状態推定の違い

| 項目                   | Simulation                              | Real World                                    |
|----------------------|----------------------------------------|-----------------------------------------------|
| ロボット位置・姿勢      | `/gazebo/model_states`（Ground Truth）  | TF2: `map → base_footprint`                   |
| 速度推定               | Gazebo が直接提供                       | 300ms スライディングウィンドウ + EMA (α=0.5)    |
| 更新レート             | 20 Hz（`/gazebo/model_states`）         | 100 Hz（TF2ルックアップタイマー）                |
| 精度                  | 完全（ノイズなし）                       | SLAM 精度に依存（Cartographer）                 |
| ドリフト              | なし                                    | あり（長時間走行でズレる）                       |

**速度推定の仕組み（Real World）:**
```
300ms ウィンドウ内の最初と最後のポーズを使い
vx_w = (x_new - x_old) / dt
vy_w = (y_new - y_old) / dt
v_body = vx_w * cos(θ) + vy_w * sin(θ)   # ワールド速度 → ロボット前進速度
v_filtered = 0.5 * v_body + 0.5 * v_prev  # EMAフィルタ
```

### 3.2 LiDARの違い

| 項目           | Simulation                         | Real World（LDS-01）                     |
|--------------|-----------------------------------|----------------------------------------|
| トピック名     | `/simulated_scan` (カスタム)       | `/scan`                                |
| レイ数         | 360（固定）                        | 360（LDS-01）または可変                  |
| 無応答値       | `-1.0`（慣習）                     | `float('inf')` または `nan`             |
| ノイズ         | なし                               | あり（反射面、照明条件など）               |
| 最大距離       | 1.0m（シミュレーション設定）        | 3.5m（LDS-01仕様）→ 1.0mにクリップ必要  |
| レイ数変換     | 不要                               | 360以外 → `np.interp`でリサンプリング    |

### 3.3 ゴール指定の違い

| 項目           | Simulation                         | Real World                              |
|--------------|-----------------------------------|-----------------------------------------|
| ゴール設定     | 起動時パラメータ（固定）            | RViz2 "2D Goal Pose" → `/move_base_simple/goal` |
| 形式           | `[x, y, theta]`（起動引数）         | `geometry_msgs/PoseStamped`             |
| 動的変更       | 再起動が必要                        | リアルタイムで変更可能                   |

### 3.4 必要なROS2スタック

**Simulation:**
```
Gazebo → /gazebo/model_states, /simulated_scan
```

**Real World:**
```
turtlebot3_node       → /scan, TF: odom→base_footprint
turtlebot3_cartographer → /map, TF: map→odom
(上記の合成で map→base_footprint が利用可能)
```

---

## 4. データ形式の選択：JSON vs ROS Bag

### 4.1 選択肢の比較

| 観点               | JSON (JSONL)                            | ROS Bag (.db3)                           |
|------------------|----------------------------------------|------------------------------------------|
| 実装難易度         | 低（標準ライブラリのみ）                 | 中（`rosbag2_py` が必要）                 |
| ポータビリティ      | 高（Python のみで読める）               | 低（ROS2 環境が必要）                     |
| ストレージ効率      | 低（テキスト、冗長）                    | 高（バイナリ圧縮可）                      |
| データ同期の柔軟性  | 高（後処理で自由に同期）               | 中（`rosbag2` の再生が前提）              |
| クラッシュ耐性      | 高（1行ずつ書き込み）                   | 高（同様）                                |
| 前処理の透明性      | 高（ステップを自由に設計）             | 低（`rosbag2` の API に縛られる）         |
| Mac での確認       | 容易（`jq`, テキストエディタ）         | 難（ROS2 が必要）                        |
| 既存パイプライン    | 対応済み（`convert_json_to_npz.py`）   | 未整備（新規開発が必要）                  |

### 4.2 推奨：JSON Lines (JSONL) 形式

**理由:**
1. **既存の `robot_data_recorder_mpc.py` が JSONL を採用** している（`robot_states_*.jsonl`, `control_inputs_*.jsonl`, `lidar_scans_*.jsonl`）
2. **`convert_json_to_npz.py` が既に整備済み**で、変換パイプラインが動作確認されている
3. macOS での開発・確認が容易（ROS2 なしで処理可能）
4. クラッシュが発生しても既存データは失われない
5. 各データストリームを独立したファイルに保存するため、一部データが欠損しても残りは利用可能

**セッションフォルダ構成（JSONL）:**
```
~/robot_data/session_YYYYMMDD_HHMMSS_REAL/
├── robot_states_TIMESTAMP.jsonl     # TF2ポーズ + 速度（50 Hz）
├── control_inputs_TIMESTAMP.jsonl   # /cmd_vel コマンド
├── lidar_scans_TIMESTAMP.jsonl      # /scan（処理前の生データ）
└── session_info_TIMESTAMP.json      # メタデータ（start/goal, duration, etc.）
```

---

## 5. Start / Goal 戦略

### 5.1 基本方針：「どこでも開始、手動停止、事後的にゴールを決定」

シミュレーションでは起動時にゴール座標を固定します。実世界では以下の戦略を採用します：

```
1. ロボットを任意の場所に配置
2. MPPIプランナー（またはteleopを人間が操作）でロボットを走行させる
3. 十分に走行したら Ctrl+C で停止
4. 最初のフレーム = Start 位置（事後的に取得）
5. 最後のフレーム = Goal 位置（事後的に取得）
6. 全フレームの target_position = 「最終フレームのロボット位置」（ゴール）をロボット座標系に変換
```

### 5.2 事後的なゴール割り当て（post-hoc goal labeling）

収集したトラジェクトリを NPZ に変換する際、各タイムステップのゴール（`target_positions`）を次のように計算します：

```python
# 最終フレームの位置を「ゴール」とする
goal_x_world = positions[-1, 0]
goal_y_world = positions[-1, 1]

target_positions = []
for i in range(len(positions)):
    robot_x, robot_y, robot_theta = positions[i, 0], positions[i, 1], orientations[i]
    
    # ゴールをロボット座標系に変換
    dx = goal_x_world - robot_x
    dy = goal_y_world - robot_y
    cos_t = np.cos(-robot_theta)
    sin_t = np.sin(-robot_theta)
    goal_x_robot = dx * cos_t - dy * sin_t
    goal_y_robot = dx * sin_t + dy * cos_t
    
    target_positions.append([goal_x_robot, goal_y_robot])

target_positions = np.array(target_positions)  # (N, 2)
```

### 5.3 目標割り当て戦略の比較

| 戦略                          | 長所                                      | 短所                              |
|-----------------------------|------------------------------------------|----------------------------------|
| **最終フレーム = ゴール**（推奨） | シンプル、自然な軌道として成立             | 最初と最後が同じ場所だと意味をなさない |
| 固定ゴール（起動時に指定）       | 一貫性がある                              | 実世界では指定が煩雑               |
| 複数サブゴール（ウェイポイント）  | 多様なデータが集まる                      | 実装が複雑、変換スクリプト要修正   |

### 5.4 重複フレームの除去

走行の開始直後と終了直前はロボットが停止しているため、有効なデータがほとんどありません。変換時に次の条件でフレームを除外します：

```python
# 速度がほぼゼロのフレームを除外
valid_mask = (
    (np.abs(control_linear) > 0.01) |  # 直進速度がある
    (np.abs(control_angular) > 0.05)   # または角速度がある
)
```

---

## 6. データ収集スクリプトの設計

### 6.1 設計方針：1コマンドで開始・停止

実機でのデータ収集は**1つのスクリプトを起動して Ctrl+C で終了**するだけで完結する設計にします。

スクリプトが行うこと：
1. ROS2 ノードを起動
2. TF2（`map → base_footprint`）からロボット状態を取得
3. `/scan` からLiDARを取得
4. `/cmd_vel` を監視してMPPI制御コマンドを記録
5. すべてのデータを JSONL 形式でインクリメンタルに書き込む
6. Ctrl+C 時にセッション情報（開始/終了位置・時刻）を `session_info.json` に保存

### 6.2 新規スクリプト: `robot_data_recorder_real_world.py`

**ファイル配置:** `src/turtlebot3_sles_data/turtlebot3_sles_data/robot_data_recorder_real_world.py`

**既存 `robot_data_recorder_mpc.py` との違い:**

| 項目               | 既存（MPC/Sim）                    | 新規（Real World）                        |
|------------------|-----------------------------------|------------------------------------------|
| ロボット状態取得    | `/gazebo/model_states`            | TF2: `map → base_footprint`              |
| 速度取得           | Gazebo から直接                    | スライディングウィンドウ推定              |
| LiDAR トピック     | `/simulated_scan`                 | `/scan`                                  |
| ゴール指定         | パラメータで固定                   | 収集後に最終フレームから自動設定           |
| セッション名       | `session_*_MPC`                   | `session_*_REAL`                         |

**主要なパラメータ:**
```python
# 記録レート
ROBOT_STATE_WRITE_RATE_HZ = 50   # TF2ポーズ更新（50 Hz）
CMD_VEL_RATE_HZ = 50             # /cmd_vel の最大記録レート
LIDAR_RATE_HZ = 10               # LiDARスキャン記録レート（ストレージ節約）

# 速度推定
TF2_SLIDING_WINDOW_SEC = 0.30    # 300ms スライディングウィンドウ
TF2_EMA_ALPHA = 0.5              # EMAフィルタ係数
```

**起動方法（想定）:**
```bash
# MPPI プランナーと一緒に起動（MPPIが走行中にデータ収集）
ros2 run turtlebot3_sles_data robot_data_recorder_real_world
```

**記録停止・保存:**
```bash
Ctrl+C
# → session_info_TIMESTAMP.json が書き出される
# → ~/robot_data/session_YYYYMMDD_HHMMSS_REAL/ に全ファイルが保存される
```

### 6.3 記録されるJSONLフォーマット

**`robot_states_TIMESTAMP.jsonl`** (1行 = 1フレーム):
```json
{"timestamp": 1706300000.123, "position": [1.234, 0.567, 0.0], "yaw": 0.785, "linear_velocity": [0.15, 0.0, 0.0], "angular_velocity": [0.0, 0.0, 0.12]}
```

**`control_inputs_TIMESTAMP.jsonl`:**
```json
{"timestamp": 1706300000.130, "linear_x": 0.15, "angular_z": 0.12}
```

**`lidar_scans_TIMESTAMP.jsonl`:**
```json
{"timestamp": 1706300000.100, "angle_min": -3.14159, "angle_max": 3.14159, "angle_increment": 0.01745, "range_min": 0.12, "range_max": 3.5, "ranges": [0.85, 0.82, ...]}
```

**`session_info_TIMESTAMP.json`** (停止時に1回書き出し):
```json
{
  "session_type": "REAL",
  "start_time": 1706300000.0,
  "end_time": 1706300300.0,
  "duration_sec": 300.0,
  "start_position": [0.0, 0.0],
  "end_position": [2.34, 1.56],
  "goal_position": [2.34, 1.56],
  "total_robot_states": 15000,
  "total_control_inputs": 15000,
  "total_lidar_scans": 3000
}
```

---

## 7. 前処理パイプライン（JSON → NPZ）

### 7.1 変換の全ステップ

```
raw JSONL files
      ↓
Step 1: タイムスタンプ同期（最近傍補間）
      ↓
Step 2: ゴール割り当て（最終フレーム → target_positions）
      ↓
Step 3: LiDAR 正規化（inf/nan → max_range, リサンプリング）
      ↓
Step 4: 無効フレーム除去（速度ゼロ, LiDAR欠損）
      ↓
Step 5: Train / Val / Test 分割（8:1:1）
      ↓
train_dataset.npz / val_dataset.npz / test_dataset.npz
```

### 7.2 タイムスタンプ同期の必要性

3つのデータストリームは異なるレートで記録されます：

| ストリーム            | レート  | 備考                       |
|--------------------|--------|---------------------------|
| robot_states       | 50 Hz  | TF2ルックアップ             |
| control_inputs     | 50 Hz  | /cmd_vel コールバック        |
| lidar_scans        | 10 Hz  | ストレージ節約のため間引き   |

各ロボット状態フレームに対して「最も近い時刻のLiDARスキャン」と「最も近い制御コマンド」を紐付けます：

```python
# 各ロボット状態に対して最近傍のLiDARスキャンを対応付け
for i, state_ts in enumerate(robot_state_timestamps):
    lidar_idx = np.argmin(np.abs(lidar_timestamps - state_ts))
    time_diff = abs(lidar_timestamps[lidar_idx] - state_ts)
    if time_diff < 0.15:  # 150ms 以内のLiDARのみ使用
        matched_lidar[i] = lidar_scans[lidar_idx]
    else:
        valid_mask[i] = False  # LiDARが遠い場合はフレームを除外
```

### 7.3 NPZデータセット形式（train_mlp.py が要求する形式）

```python
np.savez('train_dataset.npz',
    states=np.zeros((N, 2)),           # [[v, omega], ...]
    target_positions=np.zeros((N, 2)), # [[goal_x_robot, goal_y_robot], ...] ← ロボット座標系
    lidar_scans=np.zeros((N, 360)),    # 正規化済み360レイLiDAR
    control_linear=np.zeros(N),        # v_cmd
    control_angular=np.zeros(N),       # omega_cmd
)
```

> **注意:** `target_positions` はワールド座標系ではなく**ロボット座標系**でなければなりません。  
> 変換式: `(dx * cos(-θ) - dy * sin(-θ), dx * sin(-θ) + dy * cos(-θ))`

### 7.4 実行コマンド（前処理）

```bash
# Step 1: JSONL → raw NPZ（既存スクリプトを拡張して使用）
python3 src/turtlebot3_sles_data/turtlebot3_sles_data/convert_json_to_npz.py \
    ~/robot_data/session_20260126_190311_REAL

# Step 2: raw NPZ → 学習用NPZ（ゴール割り当て + 同期 + 分割）
# ※ 新規スクリプト prepare_training_data.py が必要（後述）
python3 src/turtlebot3_sles_data/turtlebot3_sles_data/prepare_training_data.py \
    ~/robot_data/session_20260126_190311_REAL \
    --output-dir ~/robot_data/training_datasets/

# 結果:
# ~/robot_data/training_datasets/
# ├── train_dataset.npz
# ├── val_dataset.npz
# └── test_dataset.npz
```

### 7.5 前処理スクリプト: `prepare_training_data.py`（設計案）

このスクリプトは現在存在せず、新規作成が必要です。主な処理：

```python
def prepare_training_data(session_folder, output_dir):
    # 1. raw NPZ の読み込み
    data = np.load(f"{session_folder}/robot_data_TIMESTAMP.npz", allow_pickle=True)
    lidar_data = np.load(f"{session_folder}/lidar_ranges_TIMESTAMP.npz", allow_pickle=True)
    
    # 2. 最終フレームをゴールとして設定
    goal_x_world = data['positions'][-1, 0]
    goal_y_world = data['positions'][-1, 1]
    
    # 3. 各フレームのゴールをロボット座標系に変換
    target_positions = compute_goal_in_robot_frame(
        data['positions'], data['orientations'], goal_x_world, goal_y_world
    )
    
    # 4. LiDAR タイムスタンプ同期
    synced_lidar = sync_lidar_to_states(
        lidar_data, data['lidar_timestamps'], data['robot_state_timestamps']
    )
    
    # 5. LiDAR 正規化（inf/nan → 1.0m, リサンプリング）
    normalized_lidar = normalize_lidar(synced_lidar, max_range=1.0)
    
    # 6. 有効フレームのフィルタリング
    valid_mask = filter_valid_frames(
        data['control_linear'], data['control_angular'], synced_lidar
    )
    
    # 7. states ベクトル作成 [v, omega]
    states = np.stack([
        data['linear_velocities'][valid_mask],
        data['angular_velocities'][valid_mask]
    ], axis=1)
    
    # 8. Train/Val/Test 分割（時系列順を崩さない）
    split_and_save(states, target_positions[valid_mask],
                   normalized_lidar[valid_mask],
                   data['control_linear'][valid_mask],
                   data['control_angular'][valid_mask],
                   output_dir)
```

---

## 8. 学習パイプライン（NPZ → Model）

### 8.1 既存スクリプト: `train_mlp.py`

`src/turtlebot3_sles_learning/turtlebot3_sles_learning/train_mlp.py` は**変更なし**で使用できます。

実行前に必要なファイル：
```
./train_dataset.npz
./val_dataset.npz
./test_dataset.npz
```

**実行コマンド（macOS でも動作）:**
```bash
cd ~/robot_data/training_datasets/

python3 /path/to/Turtlebot3_sles_ros2/src/turtlebot3_sles_learning/turtlebot3_sles_learning/train_mlp.py
```

### 8.2 学習の設定（`train_mlp.py` デフォルト値）

```python
config = {
    'batch_size': 256,
    'learning_rate': 1e-3,
    'num_epochs': 20,
    'hidden_dims': [256, 128, 64],
    'dropout': 0.1,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': 'models'
}
```

### 8.3 出力ファイル

```
./models/
├── best_model.pth         # 最良バリデーションLossのモデル（デプロイ用）
└── training_curves.png    # Train/Val Loss グラフ
```

### 8.4 評価指標

学習完了後にテストセットで以下を報告：

| 指標     | Control v    | Control ω    |
|---------|-------------|-------------|
| MAE     | 小さいほど良い | 小さいほど良い |
| RMSE    | 小さいほど良い | 小さいほど良い |
| R²      | 1.0 に近いほど良い（≥ 0.85 が目安） | 同様 |

### 8.5 複数セッションのデータを統合する場合

```bash
# 複数セッションを変換・前処理した後、NPZ を結合
python3 - <<'EOF'
import numpy as np, glob

all_data = {k: [] for k in ['states','target_positions','lidar_scans','control_linear','control_angular']}

for npz_file in sorted(glob.glob('~/robot_data/training_datasets/session_*/train_dataset.npz')):
    d = np.load(npz_file)
    for k in all_data:
        all_data[k].append(d[k])

merged = {k: np.concatenate(v, axis=0) for k, v in all_data.items()}
np.savez('merged_train_dataset.npz', **merged)
print(f"Merged: {len(merged['states'])} samples")
EOF
```

---

## 9. モデルのデプロイ方法

### 9.1 モデルファイルの配置場所

ROS2 の `--symlink-install` ビルドでは `__file__` が `install/` ディレクトリを指すため、モデルファイルは以下の2箇所に配置します：

```
# 開発時（colcon build 前に手動コピー）
src/turtlebot3_sles_control/turtlebot3_sles_control/best_model.pth

# 実行時（ROS2 ノードが参照する場所）
install/turtlebot3_sles_control/lib/turtlebot3_sles_control/best_model.pth
```

`CMakeLists.txt` の設定により `colcon build` 時に自動的に `src/` から `install/` へコピーされます：
```cmake
install(FILES
    turtlebot3_sles_control/best_model.pth
    DESTINATION lib/${PROJECT_NAME}
)
```

### 9.2 新しいモデルのデプロイ手順

```bash
# 1. 学習済みモデルを src/ にコピー
cp ~/robot_data/training_datasets/models/best_model.pth \
   ~/Turtlebot3_sles_ros2/src/turtlebot3_sles_control/turtlebot3_sles_control/best_model.pth

# 2. install/ にも直接コピー（colcon build なしで即時反映）
cp ~/robot_data/training_datasets/models/best_model.pth \
   ~/Turtlebot3_sles_ros2/install/turtlebot3_sles_control/lib/turtlebot3_sles_control/best_model.pth

# 3. NN Planner を起動して動作確認
ros2 launch turtlebot3_sles_control turtlebot3_planner_NN_real_world.launch.py
```

### 9.3 モデルのバージョン管理（推奨）

```
~/robot_data/
└── models/
    ├── best_model_sim_v1.pth           # シミュレーションデータのみ
    ├── best_model_real_20260301.pth    # 実世界データ追加（2026-03-01）
    ├── best_model_real_20260315.pth    # データ追加後の再学習
    └── best_model.pth -> best_model_real_20260315.pth  # 現在使用中（symlink）
```

---

## 10. 完全パイプライン フローチャート

```
┌─────────────────────────────────────────────────────────────────┐
│                   PHASE 1: DATA COLLECTION                      │
│                                                                  │
│  Robot Hardware                  ROS2 Topics                    │
│  ┌──────────┐   /scan            ┌──────────────────────────┐   │
│  │TurtleBot3│──────────────────→ │robot_data_recorder_      │   │
│  │Waffle Pi │   TF2 map→footpr.  │  real_world.py           │   │
│  │          │──────────────────→ │                          │   │
│  │          │   /cmd_vel (MPPI)  │ 書き込み先:              │   │
│  │          │ ←──────────────── │ ~/robot_data/            │   │
│  └──────────┘                   │ session_YYYYMMDD_REAL/   │   │
│                                  │  ├ robot_states.jsonl    │   │
│   Ctrl+C で停止 ─────────────→  │  ├ control_inputs.jsonl  │   │
│                                  │  ├ lidar_scans.jsonl     │   │
│                                  │  └ session_info.json     │   │
│                                  └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│                   PHASE 2: PREPROCESSING                        │
│                                                                  │
│  convert_json_to_npz.py                                         │
│   JSONL → robot_data_TIMESTAMP.npz + lidar_ranges_TIMESTAMP.npz │
│                          ↓                                       │
│  prepare_training_data.py（新規作成が必要）                      │
│   ・ゴール割り当て（最終フレーム → target_positions）            │
│   ・LiDAR タイムスタンプ同期（最近傍補間）                       │
│   ・LiDAR 正規化（inf/nan → 1.0m, リサンプリング）               │
│   ・無効フレーム除去（速度ゼロ等）                               │
│   ・Train(80%) / Val(10%) / Test(10%) 分割                      │
│                          ↓                                       │
│  training_datasets/                                              │
│   ├ train_dataset.npz  (N_train × 364 + 2)                      │
│   ├ val_dataset.npz                                              │
│   └ test_dataset.npz                                             │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│                   PHASE 3: TRAINING                             │
│                                                                  │
│  train_mlp.py（既存・変更不要）                                  │
│                                                                  │
│  Input(364) → MLP [256, 128, 64] → Output(2)                    │
│  MSELoss + Adam(lr=1e-3) + ReduceLROnPlateau                    │
│  20 epochs, batch=256, dropout=0.1                              │
│                          ↓                                       │
│  models/                                                         │
│   ├ best_model.pth          ← 最良Val Lossで保存                 │
│   └ training_curves.png                                          │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│                   PHASE 4: DEPLOYMENT                           │
│                                                                  │
│  1. best_model.pth を src/ と install/ にコピー                  │
│  2. ros2 launch ... turtlebot3_planner_NN_real_world.launch.py  │
│  3. RViz2 で "2D Goal Pose" を設定                              │
│  4. NN が 50 Hz で /cmd_vel を発行                              │
│                                                                  │
│  planner_nn_real_world.py                                        │
│  State: TF2 map→base_footprint (100 Hz EMA)                     │
│  LiDAR: /scan → 360 ray, [0, 1.0m]                             │
│  Goal:  /move_base_simple/goal (RViz2)                          │
│  Ctrl:  /cmd_vel @ 50 Hz                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 11. ファイル・ディレクトリ構成

### 11.1 リポジトリ内のファイル

```
Turtlebot3_sles_ros2/
└── src/
    ├── turtlebot3_sles_data/
    │   └── turtlebot3_sles_data/
    │       ├── robot_data_recorder_mpc.py         # 既存（Sim/MPC用）
    │       ├── robot_data_recorder_real_world.py  # ★ 新規作成（Real World用）
    │       ├── convert_json_to_npz.py             # 既存（JSONL → raw NPZ）
    │       ├── prepare_training_data.py           # ★ 新規作成（raw NPZ → 学習用NPZ）
    │       └── analyze_robot_data.py              # 既存（可視化）
    ├── turtlebot3_sles_learning/
    │   └── turtlebot3_sles_learning/
    │       └── train_mlp.py                       # 既存（MLP 学習）
    └── turtlebot3_sles_control/
        ├── launch/
        │   └── turtlebot3_planner_NN_real_world.launch.py  # 既存（Real World起動）
        └── turtlebot3_sles_control/
            ├── planner_nn.py                      # 既存（Simulation用）
            ├── planner_nn_real_world.py           # 既存（Real World推論）
            └── best_model.pth                     # ★ 学習後にここに配置
```

### 11.2 ロボット上のデータ（`~/robot_data/`）

```
~/robot_data/
├── session_20260301_120000_REAL/       # 収集セッション1
│   ├── robot_states_20260301_120000.jsonl
│   ├── control_inputs_20260301_120000.jsonl
│   ├── lidar_scans_20260301_120000.jsonl
│   ├── session_info_20260301_120000.json
│   ├── robot_data_20260301_120000.npz      # convert_json_to_npz.py の出力
│   └── lidar_ranges_20260301_120000.npz
├── session_20260308_140000_REAL/       # 収集セッション2
│   └── ...
└── training_datasets/                  # prepare_training_data.py の出力
    ├── train_dataset.npz
    ├── val_dataset.npz
    ├── test_dataset.npz
    └── models/
        ├── best_model.pth              # train_mlp.py の出力
        └── training_curves.png
```

---

## まとめ：実施すべき作業リスト

### 必須（未実装）

- [ ] `robot_data_recorder_real_world.py` の実装（TF2ベースの実機データ収集）
- [ ] `prepare_training_data.py` の実装（ゴール割り当て + タイムスタンプ同期 + 分割）

### 任意（品質向上）

- [ ] `convert_json_to_npz.py` の実世界JSONL対応確認（フォーマット変更があれば修正）
- [ ] 複数セッションのデータ統合スクリプト
- [ ] データ品質チェックスクリプト（速度分布、LiDAR分布の可視化）
- [ ] 学習の自動化シェルスクリプト（collect → preprocess → train → deploy を1コマンドで）

### 既存（変更不要）

- [x] `train_mlp.py` — NPZ を受け取って学習する
- [x] `planner_nn_real_world.py` — 学習済みモデルを実機で使う
- [x] `convert_json_to_npz.py` — JSONL を raw NPZ に変換する
- [x] `turtlebot3_planner_NN_real_world.launch.py` — NN Planner の起動
