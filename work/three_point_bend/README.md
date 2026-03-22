# 三点曲げ動的接触 ワークスペース

[← README](../../README.md) | [status-224](../../docs/status/status-224.md)

## フォルダ構成

```
work/three_point_bend/
├── docs/                    技術ドキュメント
│   └── diagnosis.md         不収束原因の詳細診断
├── reports/                 実行結果サマリ
│   └── summary.md           全 Run の比較表
├── assets/                  計算用インプット条件 (YAML)
│   ├── run_01_E25_dynamic_kpen_statusfilt.yaml
│   ├── run_02_E100_dynamic_kpen_statusfilt.yaml
│   ├── run_03_E100_beam_kpen_statusfilt.yaml
│   ├── run_04_E25_beam_kpen_nofilt.yaml
│   ├── run_05_E100_beam_kpen_nofilt.yaml
│   └── run_06_E25_dynamic_kpen_revert.yaml
├── tools/                   実行・後処理スクリプト
│   └── run_dynamic_bend.py  汎用実行スクリプト
├── results/                 診断結果 (YAML)
│   ├── result_run01_E25.yaml
│   ├── result_run02_E100.yaml
│   ├── ...
│   └── result_run06_E25_revert.yaml
└── logs/                    計算ログ (git 管理外)
    ├── 01_E100_dynamic_kpen_statusfilt.log
    ├── ...
    └── 06_E25_dynamic_kpen_revert.log
```

## 使い方

```bash
# E=25 MPa で実行
python work/three_point_bend/tools/run_dynamic_bend.py --E 25 --push 30 --tag run07 2>&1 | tee /tmp/log-run07-$(date +%s).log

# E=100 MPa, k_pen 手動指定
python work/three_point_bend/tools/run_dynamic_bend.py --E 100 --k-pen 19.7 --tag run08 2>&1 | tee /tmp/log-run08-$(date +%s).log
```

## 現在の課題

- push=30mm に対し frac≈0.89（26.7mm）で NR 停滞
- 詳細は `docs/diagnosis.md` 参照
