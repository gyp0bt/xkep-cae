# ContactSetupProcess

[<- README](../../../../README.md)

## 概要

接触設定の PreProcess。ContactManager 初期化 + broadphase 探索を Process API として管理する。

## 入出力

- **入力**: `ContactSetupConfig` — メッシュ・ペナルティ・摩擦・broadphase パラメータ
- **出力**: `ContactSetupData` — ContactManager インスタンス + 接触パラメータ

## 依存

- `__xkep_cae_deprecated.contact.pair.ContactManager`（importlib 経由）

## 移行元

`__xkep_cae_deprecated/process/concrete/pre_contact.py` → status-183
