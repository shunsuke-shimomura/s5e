# Z-フィルタライブラリ

直接型II転置実装を使用したデジタルIIR（無限インパルス応答）フィルタリング用のno_std Rustライブラリ。

## 概要

このライブラリは組み込みシステム向けに効率的なデジタルフィルタリング機能を提供し、コンパイル時多項式係数管理で任意のフィルタ次数（0〜5次）をサポートします。

## 特徴

- **no_std互換性**: ヒープ割り当てなしで組み込み環境で動作
- **直接型II転置**: メモリ効率の良いフィルタ実装
- **汎用多項式次数**: 0次から5次のフィルタをサポート
- **型安全係数**: const genericsを使用したコンパイル時係数管理
- **包括的テスト**: SciPy参考実装に対して検証済み

## アーキテクチャ

```text
z_filter/
├── src/
│   └── lib.rs                 # コアフィルタ実装
├── tests/
│   └── integration_tests.rs   # 包括的結合テスト
├── test-data-generator/       # テストデータ生成用Pythonスクリプト
│   ├── generate_test_data.py  # メインテストデータジェネレーター
│   ├── test_conditions.py     # テスト条件定義
│   └── README.md             # ジェネレータードキュメント
└── test-data/                # 検証用生成テストデータ
    └── [condition_name]/     # 個別テスト条件
        ├── config.json       # テスト設定
        ├── a_coeffs.csv     # 分母係数
        ├── b_coeffs.csv     # 分子係数
        └── signals/         # テスト信号と期待出力
```

## 使用方法

### 基本フィルタセットアップ

```rust
use z_filter::{DirectFormIITransposed, Poly2};

// 2次フィルタを作成
let mut filter = DirectFormIITransposed::<f64, Poly2<f64>, [f64; 3]>::new(
    Poly2::new([b0, b1, b2]),    // 分子係数
    Poly2::new([1.0, a1, a2]),   // 分母係数 (a0 = 1.0)
);

// サンプルを処理
let output = filter.process_sample(input_sample);
```

### フィルタ次数

ライブラリは0次から5次のフィルタをサポートします：

```rust
use z_filter::{Poly0, Poly1, Poly2, Poly3, Poly4, Poly5};

// 0次（ゲインのみ）
DirectFormIITransposed::<f64, Poly0<f64>, [f64; 1]>

// 1次  
DirectFormIITransposed::<f64, Poly1<f64>, [f64; 2]>

// 2次
DirectFormIITransposed::<f64, Poly2<f64>, [f64; 3]>

// ... 5次まで
DirectFormIITransposed::<f64, Poly5<f64>, [f64; 6]>
```

### フィルタ状態のリセット

```rust
filter.reset();  // 内部遅延線をクリア
```

## テスト

### 結合テストの実行

```bash
cargo test
```

### 新しいテストデータの生成

```bash
cd test-data-generator
python3 generate_test_data.py --all --output-dir ../test-data
```

### テストカバレッジ

結合テストは10種類のテスト条件でフィルタ実装を検証します：

- **butterworth_3rd_30hz**: 3次バターワース・ローパス（30Hz）
- **bessel_5th_order**: 5次ベッセル・ローパス（50Hz）
- **elliptic_4th_order**: 4次楕円・ローパス（40Hz）
- **cascade_4th_order**: 4次カスケード実装
- **custom_3rd_order**: カスタム3次設計
- **custom_bandpass**: バンドパスフィルタ（10-80Hz）
- **highpass_20hz**: 2次ハイパス（20Hz）
- **lowpass_50hz_zeta07**: カスタム減衰係数（ζ=0.7）
- **lowpass_100hz_prewarp**: プリワープ補正付き
- **low_sample_rate_filter**: 低サンプルレート用1次

各条件で複数の信号タイプ（インパルス、ステップ、正弦波、チャープ、ノイズ）をテストし、SciPy参考に対して完璧な精度（0.00e0エラー）を達成しています。

## 実装詳細

### 直接型II転置

フィルタは直接型II転置構造を実装します：

```text
x[n] ──→ b₀ ──→ (+) ──→ y[n]
         │      ↑
         ↓      │
        [z⁻¹] ──┴── a₁
         │      ↑
         b₁ ──→ (+)
         │      ↑
         ↓      │
        [z⁻¹] ──┴── a₂
         │
         b₂
```

### メモリレイアウト

- **遅延線**: `[T; N]` ここでN = filter_order + 1
- **係数**: `Poly<T>`による コンパイル時定数配列
- **ゼロ割り当て**: 全メモリを静的に割り当て

### 数値的考慮事項

- s領域からz領域への変換にbilinear変換（Tustin法）を使用
- SciPyの`signal.bilinear`実装と一致
- 典型的な制御システムアプリケーションで数値安定性を維持

## 依存関係

### ランタイム

- 外部依存関係なし（no_std互換）

### 開発/テスト

- **std**: テストビルドでのみ有効（`#[cfg(test)]`）
- **serde_json**: テスト設定読み込み
- **csv**: テストデータI/O

## ライセンス

ArkEdge Space AOCS（姿勢軌道制御システム）プロジェクトの一部。