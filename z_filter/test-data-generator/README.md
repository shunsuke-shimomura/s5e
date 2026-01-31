# テストデータジェネレーター

このディレクトリには、z_filterライブラリ用のテストデータを生成するPythonスクリプトが含まれています。

## ファイル

- **`generate_test_data.py`**: テストデータ生成のメインスクリプト
- **`test_conditions.py`**: 様々なフィルタ設定でテスト条件を定義
- **`pyproject.toml`**: 依存関係を含むPythonプロジェクト設定
- **`uv.lock`**: 依存関係のロックファイル

## 使用方法

### 全てのテストデータを生成

```bash
python3 generate_test_data.py --all --output-dir ../test-data
```

### 特定のテスト条件を生成

```bash
python3 generate_test_data.py --condition butterworth_3rd_30hz --output-dir ../test-data
```

### 利用可能なテスト条件の一覧表示

```bash
python3 generate_test_data.py --list
```

## テスト条件

以下のテスト条件が利用可能です：

1. **butterworth_3rd_30hz**: 3次バターワース・ローパスフィルタ（30Hzカットオフ）
2. **bessel_5th_order**: 5次ベッセル・ローパスフィルタ（50Hzカットオフ）
3. **elliptic_4th_order**: 4次楕円・ローパスフィルタ（40Hzカットオフ、1dBリップル、60dBストップバンド）
4. **cascade_4th_order**: 2次セクションのカスケード4次フィルタ（25Hzカットオフ）
5. **custom_3rd_order**: 特定の極・零点を持つカスタム3次フィルタ
6. **custom_bandpass**: カスタムバンドパスフィルタ（10-80Hz）
7. **highpass_20hz**: 2次ハイパスフィルタ（20Hzカットオフ）
8. **lowpass_50hz_zeta07**: カスタム減衰係数を持つ2次ローパス（ζ=0.7）
9. **lowpass_100hz_prewarp**: プリワープ補正付き2次ローパス（100Hz）
10. **low_sample_rate_filter**: 低サンプルレート用1次フィルタ（10Hz）

## 出力構造

各テスト条件は以下を生成します：

```text
test-data/
└── condition_name/
    ├── config.json          # テスト条件設定
    ├── a_coeffs.csv         # 分母係数
    ├── b_coeffs.csv         # 分子係数
    ├── bode_check.png       # 検証用ボード線図
    └── signals/
        ├── impulse_xy.csv           # インパルス応答データ
        ├── impulse_processing.png   # インパルス処理前後のグラフ
        ├── step_xy.csv              # ステップ応答データ
        ├── step_processing.png      # ステップ処理前後のグラフ
        ├── sine30_xy.csv            # 30Hz正弦波応答データ
        ├── sine30_processing.png    # 正弦波処理前後のグラフ
        ├── chirp5to200_xy.csv       # チャープ信号応答データ（該当する場合）
        ├── chirp5to200_processing.png # チャープ処理前後のグラフ
        ├── noise_xy.csv             # 白色雑音応答データ
        └── noise_processing.png     # 雑音処理前後のグラフ
```

## 依存関係

uvを使用して依存関係をインストール：

```bash
uv sync
```

または手動でインストール：

- numpy
- scipy
- matplotlib

## 開発依存関係

コード品質チェック用：

```bash
uv sync --group dev
```

## コード品質チェック

### フォーマット

```bash
# コードをフォーマット
uv run black .

# フォーマットをチェック（変更なし）
uv run black --check .
```

### リント

```bash
# コードをリント
uv run flake8 .
```

## 実装詳細

- s領域からz領域への変換に`scipy.signal.bilinear`変換を使用
- サンプリング周波数：1000Hz（テスト条件ごとに設定可能）
- 信号長：ほとんどのテストで1000サンプル
- フィルタ実装はRustの直接型II転置構造と一致

## 生成されるグラフ

各信号に対して以下のような処理前後のグラフが自動生成されます：

### 時間領域グラフ
- **上部プロット**: 入力信号（青）と出力信号（赤）の時間波形
- **下部プロット**: 
  - インパルス・ノイズ信号: 周波数領域比較（FFT）
  - その他の信号: 拡大表示（最初の100サンプル）

### 視覚化の特徴
- 高解像度PNG形式（150 DPI）
- グリッド表示で読みやすい
- 入力と出力の色分け（青：入力、赤：出力）
- エラー処理により生成に失敗してもプロセスを継続